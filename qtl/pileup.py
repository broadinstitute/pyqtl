import pandas as pd
import numpy as np
import glob
import os
import subprocess
import contextlib
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

from . import stats, annotation
from . import plot as qtl_plot
from . import genotype as gt


@contextlib.contextmanager
def cd(cd_path):
    saved_path = os.getcwd()
    os.chdir(cd_path)
    yield
    os.chdir(saved_path)


def refresh_gcs_token():
    t = subprocess.check_output('gcloud auth application-default print-access-token', shell=True)
    os.environ.putenv('GCS_OAUTH_TOKEN', t)


def _samtools_depth_wrapper(args, d=100000, user_project=None):
    bam_file, region_str, sample_id, bam_index_dir = args

    cmd = 'samtools depth -a -a -d {} -Q 255 -r {} {}'.format(d, region_str, bam_file)
    with cd(bam_index_dir):
        c = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')

    df = pd.DataFrame([i.split('\t') for i in c], columns=['chr', 'pos', sample_id])
    df.index = df['chr']+'_'+df['pos']
    return df[sample_id].astype(np.int32)


def samtools_depth(region_str, bam_s, bam_index_dir, d=100000, num_threads=12):
    """
      region_str: string in 'chr:start-end' format
      bam_s: pd.Series or dict mapping sample_id->bam_path
      bam_index_dir: directory containing local copies of the BAM/CRAM indexes
    """
    pileups_df = []
    with mp.Pool(processes=num_threads) as pool:
        for k,r in enumerate(pool.imap(_samtools_depth_wrapper, [(i,region_str,j,bam_index_dir) for j,i in bam_s.items()]), 1):
            print('\r  * running samtools depth on region {} for bam {}/{}'.format(region_str, k,len(bam_s)), end='')
            pileups_df.append(r)
        print()
    pileups_df = pd.concat(pileups_df, axis=1)
    pileups_df.index.name = 'position'
    return pileups_df


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


def group_pileups(pileups_df, libsize_s, variant_id, vcf, covariates_df=None, id_map=lambda x: '-'.join(x.split('-')[:2])):
    """
      pileups_df: output from samtools_depth()
      libsize_s: pd.Series mapping sample_id->library size (total mapped reads)
    """
    pileups_rpm_df = norm_pileups(pileups_df, libsize_s, covariates_df=covariates_df, id_map=id_map)

    # get genotypes
    g = gt.get_genotype(variant_id, vcf)[pileups_rpm_df.columns]

    # average pileups by genotype or category
    cols = np.unique(g[g.notnull()]).astype(int)
    df = pd.concat([pileups_rpm_df[g[g==i].index].mean(axis=1).rename(i) for i in cols], axis=1)
    return df


def plot(pileup_dfs, gene, mappability_bigwig=None, variant_id=None, order='additive',
         title=None, label_pos=None, show_variant_pos=False, max_intron=300, alpha=1, lw=0.5, intron_coords=None, highlight_intron=None,
         ymax=None, rasterized=False, outline=False, labels=None,
         dl=0.75, aw=4.5, dr=0.5, db=0.5, ah=1.5, dt=0.25, ds=0.4):
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

    custom_cycler = cycler('color', [
        sns.color_palette("husl", 8)[5], # blue
        sns.color_palette("Paired")[7],  # orange
        sns.color_palette("Paired")[3],  # green
    ])

    x = np.arange(pileup_dfs[0].shape[0])
    ifct = annotation.get_coord_transform(gene, max_intron=max_intron)  # compress introns
    # ifct = lambda x: x
    xi = ifct(x)

    fig = plt.figure(facecolor=(1,1,1), figsize=(fw,fh))
    ax = fig.add_axes([dl/fw, (db+da+ds)/fh, aw/fw, ah/fh])
    ax.set_prop_cycle(custom_cycler)
    axv = [ax]
    for i in range(1, num_pileups):
        ax = fig.add_axes([dl/fw, (db+da+ds+i*(da2+ah))/fh, aw/fw, ah/fh], sharex=axv[0])
        ax.set_prop_cycle(custom_cycler)
        axv.append(ax)

    if variant_id is not None:
        chrom,pos,ref,alt = variant_id.split('_')[:4]
        pos = int(pos)
        gtlabels = [
            '{0}{0}'.format(ref),
            '{0}{1}'.format(ref, alt),
            '{0}{0}'.format(alt)]
    else:
        pos = None
        # gtlabels = ['Low', 'Medium', 'High']
        # gtlabels = pileup_dfs[0].columns

    s = pileup_dfs[0].sum()
    if isinstance(order, list):
        sorder = order
    elif order=='additive':
        sorder = s.index
        if s[sorder[0]]<s[sorder[-1]]:
            sorder = sorder[::-1]
    elif order=='sorted':
        sorder = np.argsort(s)[::-1]
    elif order=='none':
        sorder = s.index

    if ymax is None:
        ymax = 0
        for k,ax in enumerate(axv):
            for i in sorder:
            # for j,i in enumerate(sorder):
                if i in pileup_dfs[k]:
                    if outline:
                        ax.plot(xi, pileup_dfs[k][i], label=i, lw=lw, alpha=alpha, rasterized=rasterized)
                    else:
                        ax.fill_between(xi, pileup_dfs[k][i], label=i, alpha=alpha, rasterized=rasterized)
            ymax = np.maximum(ymax, ax.get_ylim()[1])


    ce = gene.get_collapsed_coords()
    xl = gene.get_collapsed_coords().reshape(1,-1)[0]
    if label_pos is not None:
        xl = np.unique(np.r_[xl, label_pos])
    x = xl - gene.start_pos

    xinterp = ifct(x)
    if labels is None:
        labels = ['Mean RPM']*num_pileups
    for k,ax in enumerate(axv):
        ax.margins(0.02)
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_ylabel(labels[k], fontsize=12)
        qtl_plot.format_plot(ax, fontsize=10, lw=0.6)
        ax.tick_params(axis='x', length=3, width=0.6, pad=1)
        ax.set_xticks(xinterp)
        ax.set_xticklabels([])
    axv[0].set_xlabel('Exon coordinates on {}'.format(gene.chr), fontsize=12)

    if gene.strand=='+':
        loc = 2
    else:
        loc = 1

    leg = axv[-1].legend(labelspacing=0.15, frameon=False, fontsize=9, borderaxespad=0.5,
                         borderpad=0, loc=loc, handlelength=0.75)
    for line in leg.get_lines():
        line.set_linewidth(1)

    if variant_id is not None and title is None:
        axv[-1].set_title('{} :: {}'.format(variant_id, gene.name), fontsize=10)
    else:
        axv[-1].set_title(title, fontsize=10)

    if label_pos is not None:  # drop this now that introns are highlighted in gene model?
        for i in label_pos:
            j = list(xl).index(i)
            axv[0].get_xticklabels()[j].set_color("red")

    # highlight variant
    if show_variant_pos and pos is not None and pos>=gene.start_pos and pos<=gene.end_pos:
        x = ifct(pos-gene.start_pos)
        for ax in axv:
            xlim = np.diff(ax.get_xlim())
            ylim = np.diff(ax.get_ylim())
            h = 0.02 * ylim
            b = h/np.sqrt(3) * ah/aw * xlim/ylim
            v = np.array([[x-b,-h-0.01*ylim], [x+b,-h-0.01*ylim], [x,-0.01*ylim]])
            polygon = patches.Polygon(v, True, color='r', clip_on=False, zorder=10)
            ax.add_patch(polygon)

    # add gene model
    gax = fig.add_axes([dl/fw, db/fh, aw/fw, da/fh], sharex=axv[0])
    gene.plot(ax=gax, max_intron=max_intron, intron_coords=intron_coords, highlight_intron=highlight_intron, fc='k', ec='none', clip_on=True)
    # gax.set_xticks(ax.get_xticks())
    gax.set_title('')
    gax.set_ylabel('Isoforms', fontsize=10, rotation=0, ha='right', va='center')
    plt.setp(gax.get_xticklabels(), visible=False)
    plt.setp(gax.get_yticklabels(), visible=False)
    for s in ['top', 'right', 'bottom', 'left']:
        gax.spines[s].set_visible(False)
    for line in gax.xaxis.get_ticklines() + gax.yaxis.get_ticklines():
        line.set_markersize(0)
        line.set_markeredgewidth(0)
    axv.append(gax)

    if mappability_bigwig is not None:  # add mappability
        c = gene.get_coverage(mappability_bigwig)
        mpax = fig.add_axes([dl/fw, 0.25/fh, aw/fw, da2/fh], sharex=axv[0])
        mpax.fill_between(xi, c, color=3*[0.6], edgecolor='none', rasterized=rasterized)

        qtl_plot.format_plot(mpax, lw=0.6)
        mpax.set_ylabel('Map.', fontsize=10, rotation=0, ha='right', va='center')
        mpax.tick_params(length=0, labelbottom=False)
        axv.append(mpax)
        plt.sca(axv[0])

    return axv