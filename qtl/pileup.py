import pandas as pd
import numpy as np
import glob
import os
import subprocess
import contextlib
import tempfile
from collections.abc import Iterable
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
from matplotlib.collections import PatchCollection
from matplotlib.colors import hsv_to_rgb, rgb2hex
import seaborn as sns
from cycler import cycler
import pyBigWig

from . import stats, annotation
from . import plot as qtl_plot
from . import genotype as gt
from . import core


@contextlib.contextmanager
def cd(cd_path):
    if cd_path is not None:
        saved_path = os.getcwd()
        os.chdir(cd_path)
        yield
        os.chdir(saved_path)
    else:
        yield


def _samtools_depth_wrapper(args):
    """
    Wrapper for `samtools depth`.

    For files on GCP, GCS_OAUTH_TOKEN must be set.
    This can be done with qtl.refresh_gcs_token().
    """
    bam_file, region_str, sample_id, bam_index_dir, flags, user_project = args

    cmd = f"samtools depth {flags} -r {region_str} {bam_file}"
    if user_project is not None:
        cmd += f"?userProject={user_project}"
    with cd(bam_index_dir):
        c = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')

    df = pd.DataFrame([i.split('\t') for i in c], columns=['chr', 'pos', sample_id])
    df.index = df['pos'].astype(np.int32)
    return df[sample_id].astype(np.int32)


def samtools_depth(region_str, bam_s, bam_index_dir=None, flags='-aa -Q 255 -d 100000',
                   num_threads=12, user_project=None, verbose=True):
    """
    Run samtools depth for a list of BAMs.

    Note: reads with the flags [UNMAP,SECONDARY,QCFAIL,DUP] are excluded by default;
    see documentation for `samtools depth` and http://www.htslib.org/doc/samtools-flags.html

    Parameters
    ----------
    region_str : str
        Genomic region as 'chr:start-end'
    bam_s : pd.Series or dict
        sample_id -> bam_path
    bam_index_dir: str
        Directory already containing local copies of the BAM/CRAM indexes, or target directory
    flags : str
        Flags passed to samtools depth
    num_threads : int
        Number of threads
    user_project : str
        User project for GCP

    Returns
    -------
    pileups_df : pd.DataFrame
        DataFrame of pileups (samples in columns)
    """
    pileups_df = []
    with mp.Pool(processes=num_threads) as pool:
        for k,r in enumerate(pool.imap(_samtools_depth_wrapper, [(i,region_str,j,bam_index_dir,flags,user_project) for j,i in bam_s.items()]), 1):
            if verbose:
                print(f'\r  * running samtools depth on region {region_str} for bam {k}/{len(bam_s)}', end='' if k < len(bam_s) else None)
            pileups_df.append(r)
    pileups_df = pd.concat(pileups_df, axis=1)
    pileups_df.index.name = 'position'
    return pileups_df


def read_regtools_junctions(junctions_file, convert_positions=True):
    """
    Read output from regtools junctions extract and
    convert start/end positions to intron starts/ends.
    """
    junctions_df = pd.read_csv(junctions_file, sep='\t', header=None,
                               usecols=[0, 1, 2, 4, 5, 10],
                               names=['chrom', 'start', 'end', 'count', 'strand', 'block_sizes'])
    if convert_positions:
        junctions_df['start'] += junctions_df['block_sizes'].apply(lambda x: int(x.split(',')[0])) + 1
        junctions_df['end'] -= junctions_df['block_sizes'].apply(lambda x: int(x.split(',')[1]))
        junctions_df.index = (junctions_df['chrom'] + ':' + junctions_df['start'].astype(str)
                              + '-' + junctions_df['end'].astype(str) + ':' + junctions_df['strand'])
    return junctions_df


def regtools_wrapper(args):
    """
    Wrapper for regtools junctions extract.
    Filters out secondary and supplementary alignments
    """
    bam_file, region_str, sample_id, bam_index_dir, strand, user_project = args
    with tempfile.TemporaryDirectory() as tempdir:
        filtered_bam = os.path.join(tempdir, 'filtered.bam')
        cmd = f"samtools view -b -F 2304 {bam_file}"
        if user_project is not None:
            cmd += f"?userProject={user_project}"
        cmd += f" {region_str} > {filtered_bam}"
        with cd(bam_index_dir):
            subprocess.check_call(cmd, shell=True)
        subprocess.check_call(f"samtools index {filtered_bam}", shell=True)
        junctions_file = os.path.join(tempdir, 'junctions.txt.gz')
        cmd = f"regtools junctions extract \
            -a 8 -m 50 -M 500000 -s {strand} \
            {filtered_bam} | gzip -c > {junctions_file}"
        subprocess.check_call(cmd, shell=True, stderr=subprocess.DEVNULL)
        junctions_df = read_regtools_junctions(junctions_file, convert_positions=True)
    junctions_df.index.name = sample_id
    return junctions_df


def regtools_extract_junctions(region_str, bam_s, bam_index_dir=None, strand=0, num_threads=12,
                               user_project=None, verbose=True):
    """
      region_str: string in 'chr:start-end' format
      bam_s: pd.Series or dict mapping sample_id->bam_path
      bam_index_dir: directory containing local copies of the BAM/CRAM indexes
    """
    core.check_dependency('regtools')

    junctions_df = []
    n = len(bam_s)
    with mp.Pool(processes=num_threads) as pool:
        for k,df in enumerate(pool.imap(regtools_wrapper, [(i,region_str,j,bam_index_dir,strand,user_project) for j,i in bam_s.items()]), 1):
            if verbose:
                print(f'\r  * running regtools junctions extract on region {region_str} for bam {k}/{n}', end='' if k < n else None)
            junctions_df.append(df['count'].rename(df.index.name))
    junctions_df = pd.concat(junctions_df, axis=1).infer_objects().fillna(0).astype(np.int32)
    junctions_df.index.name = 'junction_id'
    return junctions_df


def norm_pileups(pileups_df, libsize_s, covariates_df=None, id_map=lambda x: '-'.join(x.split('-')[:2])):
    """
      pileups_df: output from samtools_depth()
      libsize_s: pd.Series mapping sample_id->library size (total mapped reads)
    """
    # convert pileups to reads per million
    pileups_rpm_df = pileups_df / libsize_s[pileups_df.columns] * 1e6
    pileups_rpm_df.rename(columns=id_map, inplace=True)

    if covariates_df is not None:
        residualizer = stats.Residualizer(covariates_df)
        pileups_rpm_df = residualizer.transform(pileups_rpm_df)

    return pileups_rpm_df


def group_pileups(pileups_df, libsize_s, variant_id, genotypes, covariates_df=None,
                  id_map=lambda x: '-'.join(x.split('-')[:2])):
    """
      pileups_df: output from samtools_depth()
      libsize_s: pd.Series mapping sample_id->library size (total mapped reads)
    """
    pileups_rpm_df = norm_pileups(pileups_df, libsize_s, covariates_df=covariates_df, id_map=id_map)

    # get genotype dosages
    if isinstance(genotypes, str) and genotypes.endswith('.vcf.gz'):
        g = gt.get_genotype(variant_id, genotypes)[pileups_rpm_df.columns]
    elif isinstance(genotypes, pd.Series):
        g = genotypes
    else:
        raise ValueError('Unsupported format for genotypes.')

    # average pileups by genotype or category
    cols = np.unique(g[g.notnull()]).astype(int)
    df = pd.concat([pileups_rpm_df[g[g == i].index].mean(axis=1).rename(i) for i in cols], axis=1)
    return df


def plot(pileup_dfs, gene, mappability_bigwig=None, variant_id=None, order='additive', junction_dfs=None,
         title=None, plot_variants=None, annot_track=None, scores_df=None, scores_colors=None, max_intron=300, alpha=1, lw=0.5, junction_alpha=0.5, junction_lw=2,
         domains_df=None, highlight_introns=None, highlight_introns2=None, shade_regions=None, colors=['#0374B3', '#C84646', '#C69B3A'], junction_colors=None,
         legend_loc='upper left', dlegend=0.1, ymax=None, xlim=None, rasterized=False, outline=False, labels=None,
         pc_color='k', nc_color='darkgray', show_cds=True,
         dl=0.75, aw=4.5, dr=0.75, db=0.5, ah=1.5, dt=0.25, ds=0.2):
    """
      pileup_dfs:
    """
    if junction_colors is None:
        junction_colors = colors

    if isinstance(pileup_dfs, pd.DataFrame):
        pileup_dfs = [pileup_dfs]
    num_pileups = len(pileup_dfs)
    if isinstance(junction_dfs, pd.DataFrame):
        junction_dfs = [junction_dfs]

    # gene model axes
    nt = len(gene.transcripts)
    ahg = 0.08 * nt + 0.01*(nt-1)
    ds2 = 0.12
    ahs = 0.25  # track/score axis height

    # set up axes
    fw = dl + aw + dr
    fh = db + ahg + ds + (num_pileups-1)*ds2 + num_pileups*ah + dt
    if mappability_bigwig is not None:
        fh += ahs + ds2
    if scores_df is not None:
        nscores = scores_df.shape[1] - 3
        fh += nscores * (ahs + ds2)

    offset = db
    fig = plt.figure(facecolor='none', figsize=(fw,fh))
    axv = []
    # scores axes
    if scores_df is not None:
        axs = []
        for i in range(nscores):
            ax = fig.add_axes([dl/fw, (offset+i*(ds2+ahs))/fh, aw/fw, ahs/fh], sharex=axv[0] if axv else None)
            axv.append(ax)
            axs.append(ax)
        axs = axs[::-1]  # plot top to bottom
        offset += nscores*ahs + (nscores-1)*ds2 + ds
    # mappability track
    if mappability_bigwig is not None:
        mpax = fig.add_axes([dl/fw, offset/fh, aw/fw, ahs/fh], sharex=axv[0] if axv else None)
        axv.append(mpax)
        offset += ahs + ds
    # gene model axes
    gax = fig.add_axes([dl/fw, offset/fh, aw/fw, ahg/fh], sharex=axv[0] if axv else None)
    axv.append(gax)
    offset += ahg + ds
    for i in range(num_pileups):
        ax = fig.add_axes([dl/fw, (offset+i*(ah+ds2))/fh, aw/fw, ah/fh], facecolor='none', sharex=axv[0] if axv else None)
        axv.append(ax)
    axv = axv[::-1]

    # parse order
    s = pileup_dfs[0].sum()
    if len(s) > 1:
        if order == 'additive':  # assumes columns are dosages
            # assert s.sort_index().equals(s), f"TEST"  # e.g., [0, 1, 2]
            if s[s.index[0]] < s[s.index[-1]]:
                order = s.index.tolist()[::-1]  # plot dosage with highest coverage first
            else:
                order = s.index.tolist()
        elif order == 'sorted':
            order = np.argsort(s)[::-1]
        elif order == 'none' or order is None:
            order = s.index
        assert all([i in s for i in order])
    else:  # single tracks
        assert all([pileup_df.shape[1] == 1 for pileup_df in pileup_dfs])
        order = [pileup_df.columns[0] for pileup_df in pileup_dfs]

    # if variant_id is provided and inputs are dosages, rename
    gtlabel_dict = None
    if variant_id is not None:
        chrom, pos, ref, alt = variant_id.split('_')[:4]
        pos = int(pos)
        if np.issubdtype(pileup_dfs[0].columns.dtype, np.integer):  # infer that inputs are genotypes
            gtlabel_dict = {0: f'{ref}:{ref}', 1: f'{ref}:{alt}', 2: f'{alt}:{alt}'}
            pileup_dfs = [df.rename(columns=gtlabel_dict) for df in pileup_dfs]
            order = [gtlabel_dict[i] for i in order]
    else:
        pos = None

    # set colors
    if pileup_dfs[0].shape[1] <= 3 and isinstance(colors, list):
        cycler_colors = colors  # default: blue, red, gold
    else:  # use tab10 colors
        cycler_colors = [rgb2hex(i) for i in plt.cm.tab10(np.arange(10))]
    custom_cycler = cycler('color', cycler_colors)
    for i in range(num_pileups):
        axv[i].set_prop_cycle(custom_cycler)

    # plot pileups
    gene.set_plot_coords(max_intron=max_intron)
    for k,ax in enumerate(axv[:num_pileups]):
        xi = gene.map_pos(pileup_dfs[k].index)
        for j,i in enumerate(order):
            if i in pileup_dfs[k]:
                if colors is not None and isinstance(colors, (dict, pd.Series)):
                    c = colors[i]
                else:
                    c = cycler_colors[j]
                if outline:
                    ax.plot(xi, pileup_dfs[k][i], color=c, label=i, lw=lw, alpha=alpha, rasterized=rasterized)
                else:
                    ax.fill_between(xi, pileup_dfs[k][i], color=c, label=i, alpha=alpha, rasterized=rasterized)

    if labels is None:
        labels = ['Mean RPM'] * num_pileups
    # format
    for k,ax in enumerate(axv[:num_pileups]):
        ax.margins(0)
        ax.set_ylabel(labels[k], fontsize=12)
        qtl_plot.format_plot(ax, fontsize=10)
        ax.tick_params(axis='x', length=3, width=0.6, pad=1)
        ax.set_xticks(gene.map_pos(gene.get_collapsed_coords().reshape(1,-1)[0]))
        ax.set_xticklabels([])
        ax.spines['left'].set_position(('outward', 6))

    if xlim is not None:
        axv[0].set_xlim(xlim)
    if ymax is not None:
        for k in range(num_pileups):
            axv[k].set_ylim([0, ymax])

    # legend for pileups
    if pileup_dfs[0].shape[1] > 1:
        handles, legend_labels = axv[0].get_legend_handles_labels()
        leg = axv[0].legend(handles[::-1], legend_labels[::-1], loc=legend_loc, handlelength=0.75, handletextpad=0.5,
                            bbox_to_anchor=(1, 1), bbox_transform=axv[0].transAxes + mtransforms.ScaledTranslation(dlegend, 0, gax.figure.dpi_scale_trans),
                            labelspacing=0.2, borderaxespad=0, fontsize=10)
        for line in leg.get_lines():
            line.set_linewidth(1.5)
    else:
        for k, pileup_df in enumerate(pileup_dfs):
            axv[k].text(1.02, 0.5, pileup_df.columns[0], ha='left', va='center', fontsize=10, transform=axv[k].transAxes)
        qtl_plot.shared_y_label(axv[:num_pileups], labels[0], fontsize=12)
        for k in range(num_pileups):
            axv[k].set_ylabel(None)

    if plot_variants is not None and not isinstance(plot_variants, str) and len(plot_variants) > 1:
        axv[0].add_artist(leg)#, clip_on=False)

    if variant_id is not None and title is None:
        axv[0].set_title(f"{gene.name} :: {variant_id.split('_b')[0].replace('_',':',1).replace('_','-')}", fontsize=11)
    else:
        axv[0].set_title(title, fontsize=11)

    # plot variant(s)
    def _plot_variant(x, color='tab:red', ec='k', **kwargs):
        for ax in axv[:num_pileups]:
            xlim = np.diff(ax.get_xlim())[0]
            ylim = np.diff(ax.get_ylim())[0]
            h = 0.075 * ylim
            b = h/np.sqrt(3) * ah/aw * xlim/ylim
            v = np.array([[x-b, -h-0.01*ylim], [x+b, -h-0.01*ylim], [x, -0.01*ylim]])
            ax.add_patch(patches.Polygon(v, closed=True, color=color, ec=ec, lw=0.5, clip_on=False, zorder=10, **kwargs))

    if isinstance(plot_variants, str):
        x = gene.map_pos(int(plot_variants.split('_')[1]))
        _plot_variant(x)
    elif isinstance(plot_variants, Iterable):
        for i in plot_variants:
            ipos = int(i.split('_')[1])
            x = gene.map_pos(ipos)
            if pos is not None and ipos == pos:
                _plot_variant(x, color='tab:red')
            else:
                _plot_variant(x, color='tab:orange')
    elif plot_variants == True and pos is not None:
        x = gene.map_pos(pos)
        _plot_variant(x)

    ax = axv[0]
    if plot_variants is not None and not isinstance(plot_variants, str) and len(plot_variants) > 1:
        kwargs = {'ec':'k', 'lw':0.5, 's':20, 'marker':'^'}
        h1 = ax.scatter(np.nan, np.nan, fc='tab:red', **kwargs, label='Lead')
        h2 = ax.scatter(np.nan, np.nan, fc='tab:orange', **kwargs, label='Other')
        if len(plot_variants) > 1:
            ax.legend(handles=[h1,h2], loc='lower left', title='CS variants',
                      handlelength=1, handletextpad=0.5, borderaxespad=0,
                      bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes + mtransforms.ScaledTranslation(dlegend, 0, gax.figure.dpi_scale_trans))
    ax.set_ylim([0, ax.get_ylim()[1]])

    # plot highlight/shading
    if shade_regions is not None:
        for region, color in shade_regions.items():
            if isinstance(region, str):
                if ':' in region:
                    region = list(map(int, region.split(':')[1].split('-')))
                else:
                    region = list(map(int, region.split('-')))
            start, end = gene.map_pos(region)
            axv[0].add_patch(patches.Rectangle((start, 0), end-start, ax.get_ylim()[1], facecolor=color, zorder=-10))
            # axv[-1].add_patch(patches.Rectangle((shade_range[k][0], 0), shade_range[k][1]-shade_range[k][0], ax.get_ylim()[1],
            #                   facecolor=[0.8]*3, zorder=-10))
            # axv[-1].add_patch(patches.Rectangle((shade_range[k], 0), shade_range[k+1]-shade_range[k], ax.get_ylim()[1],
            #                   facecolor=[0.8]*3 if k % 2 == 0 else [0.9]*3, zorder=-10))

        # if isinstance(shade_range, str):
        #     shade_range = np.array([shade_range.split(':')[-1].split('-')]).astype(int)
        # shade_range = gene.map_pos(shade_range)
        # for k in range(len(shade_range)):
        #     axv[-1].add_patch(patches.Rectangle((shade_range[k][0], 0), shade_range[k][1]-shade_range[k][0], ax.get_ylim()[1],
        #                       facecolor=[0.8]*3, zorder=-10))
            # axv[-1].add_patch(patches.Rectangle((shade_range[k], 0), shade_range[k+1]-shade_range[k], ax.get_ylim()[1],
            #                   facecolor=[0.8]*3 if k % 2 == 0 else [0.9]*3, zorder=-10))

    # add gene model
    gene.plot(ax=gax, max_intron=max_intron, wx=0.2, domains_df=domains_df, highlight_introns=highlight_introns,
              highlight_introns2=highlight_introns2, ec='none', clip_on=True,
              pc_color=pc_color, nc_color=nc_color, show_cds=show_cds)
    if domains_df is not None:
        gax.legend(loc='lower left', borderaxespad=0, bbox_to_anchor=(1, 0),
                   bbox_transform=gax.transAxes + mtransforms.ScaledTranslation(dlegend, 0, gax.figure.dpi_scale_trans),
                   title='Domains', title_fontsize=10, handlelength=1, handletextpad=0.5)
    gax.set_title('')

    if nt < 3:
        gax.set_ylabel('Isoforms', fontsize=10, rotation=0, ha='right', va='center')
    else:
        gax.set_ylabel('Isoforms', fontsize=10, labelpad=15)
    plt.setp(gax.get_xticklabels(), visible=False)
    plt.setp(gax.get_yticklabels(), visible=False)
    for s in ['top', 'right', 'bottom', 'left']:
        gax.spines[s].set_visible(False)
    gax.tick_params(length=0, labelbottom=False)

    if mappability_bigwig is not None:  # add mappability
        xi = gene.map_pos(pileup_dfs[0].index)
        # c = gene.get_coverage(mappability_bigwig)
        with pyBigWig.open(mappability_bigwig) as bw:
            c = bw.values(gene.chr, int(pileup_dfs[0].index[0]-1), int(pileup_dfs[0].index[-1]), numpy=True)
        mpax.fill_between(xi, c, color=3*[0.6], lw=1, interpolate=False, rasterized=rasterized)
        for i in ['top', 'right']:
            mpax.spines[i].set_visible(False)
            mpax.spines[i].set_linewidth(0.6)
        mpax.set_ylabel('Map.', fontsize=10, rotation=0, ha='right', va='center')
        mpax.tick_params(axis='x', length=0, labelbottom=False)
        mpax.tick_params(axis='y', labelsize=8)
        mpax.spines['left'].set_position(('outward', 6))
        mpax.set_ylim([0,1])
        plt.sca(axv[0])

    if xlim is None:
        xlim = gene.map_pos([pileup_dfs[0].index[0], pileup_dfs[0].index[-1]])
    ax.set_xlim(xlim)

    if scores_df is not None:
        scores_df = scores_df.copy()
        scores_df['start'] = gene.map_pos(scores_df['start'])
        scores_df['end'] = gene.map_pos(scores_df['end'])
        for k,ax in enumerate(axs):
            qtl_plot.format_plot(ax, hide=['top', 'right'], y_offset=6)
            col = scores_df.columns[k+3]
            rects = [patches.Rectangle((s, 0), e - s, v) for s, e, v in scores_df[['start', 'end', col]].itertuples(index=False)]
            ax.add_collection(PatchCollection(rects, facecolor=scores_colors[col] if scores_colors is not None and col in scores_colors else "tab:blue",
                                              edgecolor="none", alpha=0.7))
            ax.text(1, 0.5, col, transform=ax.transAxes, ha='left', va='center', fontsize=10)
            ax.set_xticks(xlim)  # this sets x ticks for all axes, but looks cleaner

    # if annot_track is not None:
    #     tax = fig.add_axes([dl/fw, 0/fh, aw/fw, da2/fh], sharex=axv[0])
    #     gene.plot_coverage(coverage=annot_track, ax=tax, max_intron=max_intron)
    #     tax.tick_params(length=0, labelbottom=False)

    # plot last since in a separate set of axes
    if junction_dfs is not None:
        for k, junctions_df in enumerate(junction_dfs):
            junctions_df = junctions_df.copy()
            junctions_df['start'] = junctions_df.index.map(lambda x: int(x.split(':')[-1].split('-')[0]))
            junctions_df['end'] = junctions_df.index.map(lambda x: int(x.split(':')[-1].split('-')[1]))
            if gtlabel_dict is not None:
                junctions_df.rename(columns=gtlabel_dict, inplace=True)

            # filter junctions by start/end of coverage
            junctions_df = junctions_df[(junctions_df['start'] >= pileup_dfs[0].index[0])
                                        & (junctions_df['end'] <= pileup_dfs[0].index[-1])]
            for j,i in enumerate(order):
                if i in junctions_df:
                    s = pileup_dfs[k][i].copy()
                    if junction_colors is not None and isinstance(junction_colors, (dict, pd.Series)):
                        ec = junction_colors[i]
                    else:
                        ec = cycler_colors[j]
                    gene.plot_junctions(axv[k], junctions_df, s, show_counts=False, align='minimum', count_col=i,
                                        h=0.3, lw=junction_lw, lw_fct=np.sqrt, ec=ec, alpha=junction_alpha, clip_on=True)

    return axv
