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
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
from cycler import cycler
import pyBigWig

from . import stats, annotation
from . import plot as qtl_plot
from . import genotype as gt


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


def _read_regtools_junctions(junctions_file, convert_positions=True):
    """
    Read output from regtools junctions extract
    and convert start/end positions to intron starts/ends.
    """
    junctions_df = pd.read_csv(junctions_file, sep='\t', header=None,
                               usecols=[0, 1, 2, 4, 5, 10],
                               names=['chrom', 'start', 'end', 'count', 'strand', 'block_sizes'])
    if convert_positions:
        junctions_df['start'] += junctions_df['block_sizes'].apply(lambda x: int(x.split(',')[0])) + 1
        junctions_df['end'] -= junctions_df['block_sizes'].apply(lambda x: int(x.split(',')[1]))
        junctions_df.index = junctions_df['chrom'] + ':' + junctions_df['start'].astype(str) + '-' + junctions_df['end'].astype(str) + ':' + junctions_df['strand']
        return junctions_df['count']
    return junctions_df


def regtools_wrapper(args):
    """Wrapper for regtools junctions extract"""
    bam_file, region_str, sample_id, bam_index_dir, strand = args
    with tempfile.TemporaryDirectory() as tempdir:
        filtered_bam = os.path.join(tempdir, 'filtered.bam')
        with cd(bam_index_dir):
            subprocess.check_call(f"samtools view -b -F 2304 {bam_file} {region_str} > {filtered_bam}", shell=True)
        subprocess.check_call(f"samtools index {filtered_bam}", shell=True)
        junctions_file = os.path.join(tempdir, 'junctions.txt.gz')
        cmd = f"regtools junctions extract \
            -a 8 -m 50 -M 500000 -s {strand} \
            {filtered_bam} | gzip -c > {junctions_file}"
        subprocess.check_call(cmd, shell=True, stderr=subprocess.DEVNULL)
        junctions_s = _read_regtools_junctions(junctions_file, convert_positions=True).rename(sample_id)
    return junctions_s


def regtools_extract_junctions(region_str, bam_s, bam_index_dir=None, strand=0, num_threads=12):
    """
      region_str: string in 'chr:start-end' format
      bam_s: pd.Series or dict mapping sample_id->bam_path
      bam_index_dir: directory containing local copies of the BAM/CRAM indexes
    """
    junctions_df = []
    with mp.Pool(processes=num_threads) as pool:
        for k,r in enumerate(pool.imap(regtools_wrapper, [(i,region_str,j,bam_index_dir,strand) for j,i in bam_s.items()]), 1):
            print(f'\r  * running regtools junctions extract on region {region_str} for bam {k}/{len(bam_s)}', end='')
            junctions_df.append(r)
        print()
    junctions_df = pd.concat(junctions_df, axis=1).fillna(0).astype(int)
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


def plot(pileup_dfs, gene, mappability_bigwig=None, variant_id=None, order='additive',
         title=None, plot_variants=None, annot_track=None, max_intron=300, alpha=1, lw=0.5,
         highlight_introns=None, highlight_introns2=None, shade_range=None,
         ymax=None, xlim=None, rasterized=False, outline=False, labels=None,
         dl=0.75, aw=4.5, dr=0.75, db=0.5, ah=1.5, dt=0.25, ds=0.2):
    """
      pileup_dfs:
    """

    if isinstance(pileup_dfs, pd.DataFrame):
        pileup_dfs = [pileup_dfs]
    num_pileups = len(pileup_dfs)

    nt = len(gene.transcripts)
    da = 0.08 * nt + 0.01*(nt-1)
    da2 = 0.12

    fw = dl + aw + dr
    fh = db + da + ds + (num_pileups-1)*da2 + num_pileups*ah + dt
    if mappability_bigwig is not None:
        fh += da2

    if variant_id is not None:
        chrom, pos, ref, alt = variant_id.split('_')[:4]
        pos = int(pos)
        if np.issubdtype(pileup_dfs[0].columns.dtype, np.integer):  # assume that inputs are genotypes
            gtlabels = np.array([
                f'{ref}:{ref}',
                f'{ref}:{alt}',
                f'{alt}:{alt}',
            ])
        else:
            gtlabels = None
    else:
        pos = None
        gtlabels = None

    if pileup_dfs[0].shape[1] <= 3:
        custom_cycler = cycler('color', [
            # hsv_to_rgb([0.55, 0.75, 0.8]),  #(0.2, 0.65, 0.8),  # blue
            # hsv_to_rgb([0.08, 1, 1]),  #(1.0, 0.5, 0.0),   # orange
            # hsv_to_rgb([0.3, 0.7, 0.7]),  #(0.2, 0.6, 0.17),  # green
            '#0374B3',  # blue
            '#C84646',  # red
            '#C69B3A',  # gold
        ])
    else:
        custom_cycler = None

    fig = plt.figure(facecolor=(1,1,1), figsize=(fw,fh))
    ax = fig.add_axes([dl/fw, (db+da+ds)/fh, aw/fw, ah/fh])
    ax.set_prop_cycle(custom_cycler)
    axv = [ax]
    for i in range(1, num_pileups):
        ax = fig.add_axes([dl/fw, (db+da+ds+i*(da2+ah))/fh, aw/fw, ah/fh], sharex=axv[0])
        ax.set_prop_cycle(custom_cycler)
        axv.append(ax)

    s = pileup_dfs[0].sum()
    if isinstance(order, list):
        sorder = order
    elif order == 'additive':
        sorder = s.index
        if s[sorder[0]] < s[sorder[-1]]:
            sorder = sorder[::-1]
    elif order == 'sorted':
        sorder = np.argsort(s)[::-1]
    elif order == 'none':
        sorder = s.index

    gene.set_plot_coords(max_intron=max_intron)
    for k,ax in enumerate(axv):
        xi = gene.map_pos(pileup_dfs[k].index)
        for i in sorder:
            if i in pileup_dfs[k]:
                if outline:
                    ax.plot(xi, pileup_dfs[k][i], label=i, lw=lw, alpha=alpha, rasterized=rasterized)
                else:
                    ax.fill_between(xi, pileup_dfs[k][i], label=i, alpha=alpha, rasterized=rasterized)

    if labels is None:
        labels = ['Mean RPM'] * num_pileups
    for k,ax in enumerate(axv):
        ax.margins(0)
        ax.set_ylabel(labels[k], fontsize=12)
        qtl_plot.format_plot(ax, fontsize=10, lw=0.6)
        ax.tick_params(axis='x', length=3, width=0.6, pad=1)
        ax.set_xticks(gene.map_pos(gene.get_collapsed_coords().reshape(1,-1)[0]))
        ax.set_xticklabels([])
        ax.spines['left'].set_position(('outward', 6))

    if xlim is not None:
        ax.set_xlim(xlim)
    if ymax is not None:
        ax.set_ylim([0, ymax])

    if gtlabels is not None:
        gtlabels = gtlabels[sorder]
    leg = axv[-1].legend(loc='upper left', handlelength=1, bbox_to_anchor=(1.02,1),
                         labelspacing=0.2, borderaxespad=0, labels=gtlabels)

    for line in leg.get_lines():
        line.set_linewidth(1)

    if variant_id is not None and title is None:
        axv[-1].set_title(f"{gene.name} :: {variant_id.split('_b')[0].replace('_',':',1).replace('_','-')}", fontsize=11)
    else:
        axv[-1].set_title(title, fontsize=11)

    # plot variant(s)
    def _plot_variant(x):
        for ax in axv:
            xlim = np.diff(ax.get_xlim())[0]
            ylim = np.diff(ax.get_ylim())[0]
            h = 0.04 * ylim
            b = h/np.sqrt(3) * ah/aw * xlim/ylim
            v = np.array([[x-b, -h-0.01*ylim], [x+b, -h-0.01*ylim], [x, -0.01*ylim]])
            ax.add_patch(patches.Polygon(v, closed=True, color='r', clip_on=False, zorder=10))

    if isinstance(plot_variants, str):
        x = gene.map_pos(int(plot_variants.split('_')[1]))
        _plot_variant(x)
    elif isinstance(plot_variants, Iterable):
        for i in plot_variants:
            x = gene.map_pos(int(i.split('_')[1]))
            _plot_variant(x)
    elif plot_variants == True and pos is not None:
        x = gene.map_pos(pos)
        _plot_variant(x)

    # plot highlight/shading
    if shade_range is not None:
        if isinstance(shade_range, str):
            shade_range = shade_range.split(':')[-1].split('-')
        shade_range = np.array(shade_range).astype(int)
        shade_range = gene.map_pos(shade_range)
        for k in range(len(shade_range)-1):
            axv[-1].add_patch(patches.Rectangle((shade_range[k], 0), shade_range[k+1]-shade_range[k], ax.get_ylim()[1],
                              facecolor=[0.8]*3 if k % 2 == 0 else [0.9]*3, zorder=-10))

    # add gene model
    gax = fig.add_axes([dl/fw, db/fh, aw/fw, da/fh], sharex=axv[0])
    gene.plot(ax=gax, max_intron=max_intron, wx=0.2, highlight_introns=highlight_introns,
              highlight_introns2=highlight_introns2, fc='k', ec='none', clip_on=True)
    gax.set_title('')
    gax.set_ylabel('Isoforms', fontsize=10, rotation=0, ha='right', va='center')
    plt.setp(gax.get_xticklabels(), visible=False)
    plt.setp(gax.get_yticklabels(), visible=False)
    for s in ['top', 'right', 'bottom', 'left']:
        gax.spines[s].set_visible(False)
    gax.tick_params(length=0, labelbottom=False)
    axv.append(gax)

    if mappability_bigwig is not None:  # add mappability
        xi = gene.map_pos(pileup_dfs[0].index)
        # c = gene.get_coverage(mappability_bigwig)
        with pyBigWig.open(mappability_bigwig) as bw:
            c = bw.values(gene.chr, int(pileup_dfs[0].index[0]-1), int(pileup_dfs[0].index[-1]), numpy=True)
        mpax = fig.add_axes([dl/fw, 0.25/fh, aw/fw, da2/fh], sharex=axv[0])
        mpax.fill_between(xi, c, color=3*[0.6], lw=1, interpolate=False, rasterized=rasterized)
        for i in ['top', 'right']:
            mpax.spines[i].set_visible(False)
            mpax.spines[i].set_linewidth(0.6)
        mpax.set_ylabel('Map.', fontsize=10, rotation=0, ha='right', va='center')
        mpax.tick_params(length=0, labelbottom=False)
        axv.append(mpax)
        plt.sca(axv[0])

    if annot_track is not None:
        tax = fig.add_axes([dl/fw, 0/fh, aw/fw, da2/fh], sharex=axv[0])
        gene.plot_coverage(coverage=annot_track, ax=tax, max_intron=max_intron)
        tax.tick_params(length=0, labelbottom=False)
    # axv[-1].set_xlabel(f'Exon coordinates on {gene.chr}', fontsize=12)

    return axv
