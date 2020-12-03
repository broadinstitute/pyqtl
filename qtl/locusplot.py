#!/usr/bin/env python3

"""locusplot.py: LocusZoom-style visualization of the p-value landscape for multiple QTL or GWAS"""

__author__ = "Francois Aguet"
__copyright__ = "Copyright 2019, The Broad Institute"
__license__ = "BSD3"

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import seaborn as sns
import argparse
import subprocess
import os
import io
import gzip
import re

from . import annotation
from . import genotype as gt


def get_sample_ids(vcf):
    """Get sample IDs from VCF"""
    if vcf.endswith('.bcf'):
        return subprocess.check_output('bcftools query -l {}'.format(vcf), shell=True).decode().strip().split('\n')
    else:
        with gzip.open(vcf, 'rt') as f:
            for line in f:
                if line[:2]=='##': continue
                break
        return line.strip().split('\t')[9:]


def get_cis_genotypes(chrom, tss, vcf, field='GT', window=1000000):
    """Get dosages from VCF (using tabix)"""
    region_str = chrom+':'+str(np.maximum(tss-window, 1))+'-'+str(tss+window)
    return get_genotypes_region(vcf, region_str, field=field)


def get_genotypes_region(vcf, region, field='GT'):
    """Get dosages from VCF (using tabix)"""
    print('Getting {} for region {}'.format(field, region))
    cmd = 'tabix '+vcf+' '+region
    s = subprocess.check_output(cmd, shell=True, executable='/bin/bash')
    s = s.decode().strip()
    if len(s)==0:
        raise ValueError('No variants in region {}'.format(region))
    s = s .split('\n')
    variant_ids = [si.split('\t', 3)[-2] for si in s]
    field_ix = s[0].split('\t')[8].split(':').index(field)

    if field=='GT':
        gt_map = {'0/0':0, '0/1':1, '1/1':2, './.':np.NaN,
                  '0|0':0, '0|1':1, '1|0':1, '1|1':2, '.|.':np.NaN}
        s = [[gt_map[i.split(':', field_ix+1)[field_ix]] for i in si.split('\t')[9:]] for si in s]
    else:
        s = [[i.split(':', field_ix+1)[field_ix] for i in si.split('\t')[9:]] for si in s]

    return pd.DataFrame(data=s, index=variant_ids, columns=get_sample_ids(vcf), dtype=np.float32)


def load_eqtl(eqtl_file, gene_id, chrom=None):
    """Load full eQTL or ieQTL summary statistics for the specified gene"""
    if eqtl_file.endswith('parquet'):
        p = eqtl_file
        if chrom is not None:
            p = eqtl_file.replace(re.findall('chr\d+', eqtl_file)[0], chrom)
        cols = ['phenotype_id', 'variant_id', 'pval_gi', 'pval_nominal']
        eqtl_df = pd.read_parquet(p, columns=cols)
        eqtl_df = eqtl_df[eqtl_df['phenotype_id']==gene_id].set_index('variant_id').rename(columns={'pval_gi':'pval_nominal'})
    else:
        s = subprocess.check_output('zcat {} | grep {}'.format(eqtl_file, gene_id), shell=True).decode()
        eqtl_cols = ['gene_id', 'variant_id', 'tss_distance', 'ma_samples', 'ma_count', 'maf', 'pval_nominal', 'slope', 'slope_se']
        eqtl_df = pd.read_csv(io.StringIO(s), sep='\t', header=None, names=eqtl_cols, index_col=1)
    eqtl_df['position'] = eqtl_df.index.map(lambda x: int(x.split('_')[1]))
    return eqtl_df


def load_gwas(gwas_file, variant_ids):
    """Load GWAS summary statistics"""
    gwas_df = pd.read_csv(gwas_file, sep='\t', usecols=['panel_variant_id', 'position', 'pvalue', 'frequency', 'sample_size'], index_col=0)
    gwas_df = gwas_df.loc[gwas_df.index.isin(variant_ids)].rename(columns={'pvalue':'pval_nominal', 'frequency':'maf'})
    gwas_df['maf'] = np.where(gwas_df['maf']<=0.5, gwas_df['maf'], 1-gwas_df['maf'])
    return gwas_df


def compute_ld(genotype_df, variant_id):
    """Compute LD (r2)"""
    # return genotype_df.corrwith(genotype_df.loc[variant_id], axis=1, method='pearson')**2
    g0 = genotype_df - genotype_df.values.mean(1, keepdims=True)
    d = (g0**2).sum(1) * (g0.loc[variant_id]**2).sum()
    return (g0 * g0.loc[variant_id]).sum(1)**2 / d


def get_ld(vcf, variant_id, phenotype_bed, window=200000):
    """Load genotypes and compute LD (r2)"""
    phenotype_df = pd.read_csv(phenotype_bed, sep='\t', index_col=3, nrows=0).drop(['#chr', 'start', 'end'], axis=1)
    chrom, pos, _, _, _ = variant_id.split('_')
    pos = int(pos)
    genotype_df = get_cis_genotypes(chrom, pos, vcf, window=window)[phenotype_df.columns]
    gt.impute_mean(genotype_df, verbose=False)
    r2_s = compute_ld(genotype_df, variant_id)
    return r2_s


def get_rsid(id_lookup_table, variant_id):
    s = subprocess.check_output('zcat {} | grep {}'.format(id_lookup_table, variant_id), shell=True).decode()
    rs_id = [i for i in s.strip().split('\t') if i.startswith('rs')]
    assert len(rs_id)==1
    return rs_id[0]


def compare_loci(pval_df1, pval_df2, r2_s, variant_id, rs_id=None,
                 highlight_ids=None, colorbar=True, ah=2, aw=2):
    """plot similar to LocusCompare (Liu et al., Nat Genet, 2019)"""
    assert np.all(pval_df1.index==pval_df2.index)

    dl = 0.75
    dr = 0.75
    db = 0.75
    dt = 0.25
    fw = dl + aw + dr
    fh = db + ah + dt

    fig = plt.figure(facecolor=(1,1,1), figsize=(fw,fh))
    ax = fig.add_axes([dl/fw, db/fh, aw/fw, ah/fh])

    # LocusZoom colors
    lz_colors = ["#7F7F7F", "#282973", "#8CCCF0", "#69BD45", "#F9A41A", "#ED1F24"]
    select_args = {'s':24, 'marker':'D', 'c':"#714A9D", 'edgecolor':'k', 'lw':0.25}
    highlight_args = {'s':24, 'marker':'D', 'c':"#ED1F24", 'edgecolor':'k', 'lw':0.25}
    indep_args = {'s':30, 'marker':'^', 'c':"#E200B2", 'edgecolor':'k', 'lw':0.25}
    cmap = mpl.colors.ListedColormap(lz_colors)
    bounds = np.append(-1, np.arange(0,1.2,0.2))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if colorbar:
        s = 0.66
        cax = fig.add_axes([(dl+aw+0.2)/fw, (db+ah-1.25*s)/fh, s*0.25/fw, s*1.25/fh])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds[1:],  # start at 0
                                        ticks=bounds,
                                        spacing='proportional',
                                        orientation='vertical')
        cax.set_title('r$\mathregular{^2}$', fontsize=12)

    if rs_id is not None:
        t = rs_id
    else:
        t = variant_id

    x = -np.log10(pval_df1['pval_nominal'])
    y = -np.log10(pval_df2['pval_nominal'])

    # sort variants by LD; plot high LD in front
    s = r2_s[x.index].sort_values().index
    ax.scatter(x[s], y[s], c=r2_s[s].replace(np.NaN, -1), s=20, cmap=cmap, norm=norm, edgecolor='k', lw=0.25, clip_on=False)

    if highlight_ids is not None:
        ax.scatter(x[highlight_ids], y[highlight_ids], **highlight_args, clip_on=False)

    x = -np.log10(pval_df1.loc[variant_id, 'pval_nominal'])
    y = -np.log10(pval_df2.loc[variant_id, 'pval_nominal'])
    ax.scatter(x, y, **select_args)
    txt = ax.annotate(t, (x, y), xytext=(-5,5), textcoords='offset points', ha='right')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=4, nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=4, nbins=5))

    ax.set_xlabel('-log$\mathregular{_{10}}$(p-value)', fontsize=12)
    ax.set_ylabel('-log$\mathregular{_{10}}$(p-value)', fontsize=12)

    ax.set_xlim([0, ax.get_xlim()[1]])
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.spines['left'].set_position(('outward', 6))
    ax.spines['bottom'].set_position(('outward', 6))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


def plot_locus(pvals, gene_id=None, variant_ids=None, annot=None, r2_s=None, rs_id=None, show_rsid=True,
               highlight_ids=None, credible_sets=None,
               tracks=None, track_colors=None,
               shared_only=True,
               xlim=None, ymax=None, sharey=None, labels=None, title=None, shade_range=None,
               gene_label_pos='right', chr_label_pos='bottom', window=200000, colorbar=True,
               dl=0.75, aw=4, dr=0.75, db=0.5, ah=1.25, dt=0.25, ds=0.05, gh=0.2, th=1.5,
               single_ylabel=False, ylabel='-log$\mathregular{_{10}}$(p-value)', rasterized=False):
    """
      pvals: pd.DataFrame, or list of pd.DataFrame. Must contain 'pval_nominal' and 'position' columns.
      shared_only: only plot variants that are present in all inputs
      sharey: list of dataset indexes with shared ylim
    """

    if isinstance(pvals, pd.DataFrame):
        pvals = [pvals]
    n = len(pvals)

    if variant_ids is None:
        variant_ids = []
        for p in pvals:
            if 'pval_nominal' in p:
                variant_ids.append(p['pval_nominal'].idxmin())
            elif 'pip' in p:
                variant_ids.append(p['pip'].idxmax())
            else:
                variant_ids.append(None)
    elif isinstance(variant_ids, str):
        variant_ids = [variant_ids]*n

    chrom, pos = variant_ids[0].split('_')[:2]
    pos = int(pos)

    # set up figure
    if chr_label_pos!='bottom':
        db = 0.25
        dt = 0.5
    fw = dl + aw + dr
    fh = db + n*ah + (n-1)*ds + dt
    if gene_id is not None:
        fh += gh
    else:
        gh = 0
    if tracks is not None:
        fh += th + ds
    fig = plt.figure(figsize=(fw,fh))
    axes = [fig.add_axes([dl/fw, (fh-dt-ah)/fh, aw/fw, ah/fh])]
    for i in range(1,n):
        axes.append(fig.add_axes([dl/fw, (fh-dt-ah-i*(ah+ds))/fh, aw/fw, ah/fh], sharex=axes[0]))
    if tracks is not None:
        tax = fig.add_axes([dl/fw, (fh-dt-n*(ah+ds)-th)/fh, aw/fw, th/fh], sharex=axes[0])
    if gene_id is not None:
        gax = fig.add_axes([dl/fw, (db)/fh, aw/fw, gh/fh], sharex=axes[0])

    if xlim is None:
        xlim = np.array([pos-window, pos+window])
    axes[0].set_xlim(xlim)
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=3, nbins=4))

    # LocusZoom colors
    lz_colors = ["#7F7F7F", "#282973", "#8CCCF0", "#69BD45", "#F9A41A", "#ED1F24"]
    select_args = {'s':24, 'marker':'D', 'c':"#714A9D", 'edgecolor':'k', 'lw':0.25}
    highlight_args = {'s':24, 'marker':'D', 'edgecolor':'k', 'lw':0.25}
    cmap = mpl.colors.ListedColormap(lz_colors)
    bounds = np.append(-1, np.arange(0,1.2,0.2))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    if colorbar:
        s = 0.66
        cax = fig.add_axes([(dl+aw+0.1)/fw, (fh-dt-ah+(1-s)/2*ah)/fh, s*ah/5/fw, s*ah/fh])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                        norm=norm,
                                        boundaries=bounds[1:],  # start at 0
                                        ticks=bounds,
                                        spacing='proportional',
                                        orientation='vertical')
        cax.set_title('r$\mathregular{^2}$', fontsize=12)

    # common set of variants
    common_ix = pvals[0].index
    for pval_df in pvals[1:]:
        common_ix = common_ix[common_ix.isin(pval_df.index)]

    # plot p-values
    ylabels = []
    for k,(ax,variant_id,pval_df) in enumerate(zip(axes, variant_ids, pvals)):
        # select variants in window
        m = (pval_df['position']>=xlim[0]) & (pval_df['position']<=xlim[1])
        if shared_only:
            m &= pval_df.index.isin(common_ix)
        window_df = pval_df.loc[m]
        x = window_df['position']
        if 'pval_nominal' in pval_df:
            p = -np.log10(window_df['pval_nominal'])
            ylabels.append(ylabel)
            minp = -np.log10(pval_df.loc[variant_id, 'pval_nominal'])

            # sort variants by LD; plot high LD in front
            if r2_s is not None:
                s = r2_s[window_df.index].sort_values().index
                r2 = r2_s[s].replace(np.NaN, -1)
            else:
                s = pval_df.loc[window_df.index, 'r2'].sort_values().index
                r2 = pval_df.loc[s, 'r2'].replace(np.NaN, -1)
            ax.scatter(x[s], p[s], c=r2, s=20, cmap=cmap, norm=norm, edgecolor='k', lw=0.25, rasterized=rasterized)

        elif 'pip' in pval_df:
            p = window_df['pip']
            ylabels.append('PIP')
            minp = pval_df['pip'].max()
            if 'cs_id' in pval_df:
                pip_df = pval_df[pval_df['cs_id'].notnull()].copy()
                pip_df['cs_id'] = pip_df['cs_id'].astype(int)
                cs_colors = sns.color_palette('Set1', desat=0.66).as_hex()
                cs_cmap = mpl.colors.ListedColormap(cs_colors)
                cs_norm = mpl.colors.BoundaryNorm(np.arange(1,cmap.N+1), cmap.N)
                ax.scatter(pip_df['position'], pip_df['pip'], c=pip_df['cs_id'], s=22, ec='none', cmap=cs_cmap, norm=cs_norm, rasterized=rasterized)
            else:
                raise NotImplementedError

        if credible_sets is not None:
            df = pval_df.loc[credible_sets[k]['variant_id']]
            ax.scatter(df['position'], -np.log10(df['pval_nominal']), c=credible_sets[k]['cs_id']/10, s=50)
            # credible_sets[k]['variant_id']

        if highlight_ids is not None:  # plot relative to lead variant
            if isinstance(highlight_ids, str):
                highlight_ids = [highlight_ids]
            highlight_df = pval_df.loc[highlight_ids].copy()
            highlight_df = highlight_df[~highlight_df.index.isin(variant_ids)]  # drop lead variant
            ix = highlight_df.index
            if 'pip' not in pval_df:
                ax.scatter(x[ix], p[ix], c=r2[ix], cmap=cmap, norm=norm, **highlight_args)
            else:
                if 'cs_id' in pval_df:  # only plot highlight IDs that are in CSs
                    ix = highlight_df.index[highlight_df.index.isin(pip_df.index)]
                    ax.scatter(x[ix], p[ix], c='goldenrod', **highlight_args)
                else:
                    ax.scatter(x[ix], p[ix], c='goldenrod', **highlight_args)

        # plot selected variant, add text label, etc.
        minpos = int(variant_id.split('_')[1])
        if 'pip' not in pval_df:
            ax.scatter(minpos, minp, **select_args)
        else:
            # ax.scatter(minpos, minp, **select_args)
            ax.scatter(minpos, minp, c=pip_df.loc[variant_id, 'cs_id'], cmap=cs_cmap, norm=cs_norm,
                      s=24, marker='D', ec='k', lw=0.25)

        if rs_id is not None:
            if isinstance(rs_id, str):
                t = rs_id
            else:
                t = rs_id[k]
        else:
            t = variant_id

        if show_rsid:  # text label
            if (minpos-xlim[0])/(xlim[1]-xlim[0]) < 0.55:  # right
                txt = ax.annotate(t, (minpos, minp), xytext=(5,5), textcoords='offset points')
            else:
                txt = ax.annotate(t, (minpos, minp), xytext=(-5,5), ha='right', textcoords='offset points')
        # if minp < -np.log10(pval_df['pval_nominal'].min())*0.8:
            txt.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none', boxstyle="round,pad=0.1"))

    for k,ax in enumerate(axes):
        ax.margins(y=0.2)
        if ymax is None:
            ax.set_ylim([0, ax.get_ylim()[1]])
        else:
            ax.set_ylim([0, ymax[k]])
        if shade_range is not None:  # highlight subregion with gray background
            ax.add_patch(patches.Rectangle((shade_range[0], 0), np.diff(shade_range), ax.get_ylim()[1], facecolor=[0.66]*3, zorder=-10))

    if labels is not None:
        for ax,t in zip(axes, labels):
            ax.text(0.02, 0.925, t, transform=ax.transAxes, va='top', ha='left', fontsize=12)

    if single_ylabel:
        # for ax in axes:
        #     x.set_ylabel(None)
        m = db + (n*ah + (n-1)*ds)/2
        fig.text(0.035, m/fh, '-log$\mathregular{_{10}}$(p-value)', va='center', rotation=90, fontsize=14);
    else:
        for k,ax in enumerate(axes):
            ax.set_ylabel(ylabels[k], fontsize=12)#, labelpad=15)
            ax.yaxis.set_label_coords(-0.07*4/aw, 0.5)

    for ax in axes:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=3, nbins=4))
    axes[0].set_title(title, fontsize=12)

    if chr_label_pos=='bottom':
        v = axes
    else:
        v = axes[1:]
    if tracks is not None:
        v += [tax]
    for ax in v:
        plt.setp(ax.get_xticklabels(), visible=False)
        for line in ax.xaxis.get_ticklines():
            line.set_markersize(0)
            line.set_markeredgewidth(0)

    if sharey is not None:  # force equal y limits
        shared_max = 0
        for k in sharey:
            y = axes[k-1].get_ylim()[1]
            if y > shared_max:
                shared_max = y
        for k in sharey:
            axes[k-1].set_ylim([0, shared_max])

    if tracks is not None:  # plot, e.g., ATAC-seq tracks
        ntracks = tracks.shape[1]
        x = tracks.index
        maxv = tracks.max().max()
        for k, label in enumerate(tracks):
            y0 = (ntracks-1-k) * np.ones(len(x))  # vertical offset
            if track_colors is not None and label in track_colors:
                color = track_colors[label]
            else:
                color = 'k'
            c = tracks[label]
            tax.fill_between(x, 0.95*c/maxv + y0, y0,
                             antialiased=False, linewidth=1, facecolor=color,
                             clip_on=True, rasterized=True)
        tax.set_yticks(np.arange(ntracks))
        tax.set_yticklabels(tracks.columns[::-1], fontsize=9, va='bottom')
        for line in tax.yaxis.get_ticklines():
            line.set_markersize(0)
            line.set_markeredgewidth(0)
        for i in ['top', 'bottom', 'right', 'left']:
            tax.spines[i].set_visible(False)
        tax.set_ylim([0, ntracks])

    if gene_id is None or annot.gene_dict[gene_id].chr != chrom:
        axes[-1].xaxis.tick_bottom()
        axes[-1].xaxis.set_label_position('bottom')
        axes[-1].spines['bottom'].set_visible(True)
        axes[-1].tick_params(axis='x', pad=2)
        axes[-1].xaxis.labelpad = 8
        axes[-1].set_xlabel('Position on {} (Mb)'.format(chrom), fontsize=14)
        axes[-1].set_xticklabels(axes[-1].get_xticks()/1e6)
    else:   # add gene model
        gene = annot.gene_dict[gene_id]
        #  plot gene model and annotate
        if gene.end_pos < xlim[0]:
            x = gh/aw/2
            v = np.array([[x,0.2], [x-0.8*gh/aw, 0.5], [x,0.8]])
            polygon = patches.Polygon(v, True, color='k', transform=gax.transAxes, clip_on=False)
            gax.add_patch(polygon)
            txt = '{} (~{:.1f}Mb)'.format(gene.name, (pos-gene.tss)/1e6)
            gax.set_ylim([-1,1])
            gax.text(1.5*x, 0.5, txt, va='center', ha='left', transform=gax.transAxes)
        elif gene.start_pos > xlim[1]:
            x = 1 - gh/aw/2
            v = np.array([[x,0.2], [x+0.8*gh/aw, 0.5], [x,0.8]])
            polygon = patches.Polygon(v, True, color='k', transform=gax.transAxes, clip_on=False)
            gax.add_patch(polygon)
            txt = '{} (~{:.1f}Mb)'.format(gene.name, (gene.tss-pos)/1e6)
            gax.set_ylim([-1,1])
            gax.text(1 - gh/aw/2*1.5, 0.5, txt, va='center', ha='right', transform=gax.transAxes)
        else:
            gene.plot(ax=gax, max_intron=1e9, fc='k', ec='none', reference=1, scale=0.33, show_ylabels=False, clip_on=True)
            if gene_label_pos=='right':
                gax.annotate(gene.name, (np.minimum(gene.end_pos, xlim[1]), 0), xytext=(5,0), textcoords='offset points', va='center', ha='left')
            else:
                gax.annotate(gene.name, (np.maximum(gene.start_pos, xlim[0]), 0), xytext=(-5,0), textcoords='offset points', va='center', ha='right')

        if chr_label_pos=='bottom':
            gax.set_xlabel('Position on {} (Mb)'.format(chrom), fontsize=14)
        else:
            plt.setp(gax.get_xticklabels(), visible=False)
            for line in gax.xaxis.get_ticklines():
                line.set_markersize(0) # tick length
                line.set_markeredgewidth(0) # tick line width
            gax.spines['bottom'].set_visible(False)

        gax.set_yticks([])
        gax.set_yticklabels([])
        gax.spines['top'].set_visible(False)
        gax.spines['left'].set_visible(False)
        gax.spines['right'].set_visible(False)
        gax.set_title('')
        gax.set_xticklabels(gax.get_xticks()/1e6);
        axes.append(gax)

    if chr_label_pos!='bottom':
        axes[0].xaxis.tick_top()
        axes[0].xaxis.set_label_position('top')
        axes[0].set_xlabel('Position on {} (Mb)'.format(chrom), fontsize=14)
        axes[0].spines['top'].set_visible(True)
        axes[0].tick_params(axis='x', pad=2)
        axes[0].xaxis.labelpad = 8

    for ax in axes:
        ax.set_facecolor('none')

    return axes


def plot_ieqtl_locus(eqtl_df, ieqtl_df, gwas_df, r2_s, gene_id, variant_id, annot,
                     independent_df=None, rs_id=None, trait_name=None, pp4=None, window=200000,
                     aw=4, ah=1.25):

    pvals = [
        gwas_df.rename(columns={'pvalue':'pval_nominal'}),
        eqtl_df.loc[eqtl_df.index.isin(r2_s.index)],
        ieqtl_df.loc[ieqtl_df.index.isin(r2_s.index)]
    ]

    if trait_name is None:
        trait_name = 'GWAS'

    labels = [trait_name]
    if pp4 is None:
        labels.extend(['eQTL', 'ieQTL'])
    else:
        labels.extend(['eQTL (PP4 = {:.2f})'.format(pp4[0]), 'ieQTL (PP4 = {:.2f})'.format(pp4[1])])

    plot_locus(pvals, r2_s, gene_id, variant_id, annot, rs_id=rs_id,
                    highlight_ids=None, aw=aw, ah=ah,
                    labels=labels, shade_range=None, gene_label_pos='right', chr_label_pos='bottom', window=window)



if __name__=='__main__':
    mpl.use('Agg')

    parser = argparse.ArgumentParser(description='locus plot')
    parser.add_argument('--eqtl', required=True, help='QTL summary statistics file containing all pairwise associations')
    parser.add_argument('--ieqtl', required=True, help='iQTL summary statistics file containing all pairwise associations')
    parser.add_argument('--gwas', required=True, help='GWAS summary statistics file')
    parser.add_argument('--vcf', required=True, help='VCF file')
    parser.add_argument('--phenotype_bed', required=True, help='Phenotype BED file used for QTL mapping (required for parsing sample IDs)')
    parser.add_argument('--gene_id', required=True, help='Gene ID')
    parser.add_argument('--gtf', required=True, help='Gene annotation in GTF format')
    parser.add_argument('--variant_id', help='Variant ID')
    parser.add_argument('--phenotype_id', default=None, help='Select p-values for a specific phenotype, e.g., for sQTLs')
    parser.add_argument('--rs_id', help='')
    parser.add_argument('--id_lookup_table', help='Lookup table mapping variant IDs to rs IDs (rs ID must be in last column)')
    parser.add_argument('--window', default=200000, type=int, help='')
    parser.add_argument('--labels', nargs='+', default=None)
    parser.add_argument('--ymax', nargs='+', type=np.float64, default=None)
    parser.add_argument('--sharey', nargs='+', type=int, help='Use same y-axis for the specified plots (1-indexed, with top plot starting at 1.)', default=None)
    parser.add_argument('--top_variant', default='ieQTL', choices=['GWAS', 'eQTL', 'ieQTL'])
    parser.add_argument('--output_dir', default='.', type=str, help='')
    args = parser.parse_args()

    print('Loading gene annotation')
    annot = annotation.Annotation(args.gtf)
    gene = annot.gene_dict[args.gene_id]
    chrom = gene.chr

    if args.phenotype_id is not None:
        load_id = args.phenotype_id
    else:
        load_id = args.gene_id

    print('Loading eQTL summary statistics')
    eqtl_df = load_eqtl(args.eqtl, load_id, chrom)

    print('Loading ieQTL summary statistics')
    ieqtl_df = load_eqtl(args.ieqtl, load_id, chrom)
    if not np.all(ieqtl_df.index.isin(eqtl_df.index)):
        print('WARNING: ieQTL results contain variants not present in eQTL results')

    print('Loading GWAS summary statistics')
    gwas_df = load_gwas(args.gwas, eqtl_df.index)

    if args.variant_id is None:
        common_ix = ieqtl_df.index[ieqtl_df.index.isin(eqtl_df.index) & ieqtl_df.index.isin(gwas_df.index)]
        if args.top_variant == 'ieQTL':
            variant_id = ieqtl_df.loc[common_ix, 'pval_nominal'].idxmin()
        elif args.top_variant == 'eQTL':
            variant_id = eqtl_df.loc[common_ix, 'pval_nominal'].idxmin()
        else:
            variant_id = gwas_df.loc[common_ix, 'pval_nominal'].idxmin()
    else:
        variant_id = args.variant_id
    chrom, pos, ref, alt, _ = variant_id.split('_')
    pos = int(pos)

    print('Loading genotypes and computing LD')
    r2_s = get_ld(args.vcf, variant_id, args.phenotype_bed)

    rs_id = args.rs_id
    if rs_id is None and args.id_lookup_table is not None:
        print('Parsing rsID lookup table')
        rs_id = get_rsid(args.id_lookup_table, variant_id)

    print('Generating plot')
    plot_locus([gwas_df, eqtl_df, ieqtl_df], args.gene_id, variant_id, annot, r2_s=r2_s,
               rs_id=rs_id, labels=[i.encode('utf-8').decode('unicode_escape') for i in args.labels],
               ymax=args.ymax, sharey=args.sharey,
               window=args.window, shared_only=True)

    pdf_file = os.path.join(args.output_dir, '{}.{}.locus_plot.pdf'.format(gene.name, variant_id))
    plt.savefig(pdf_file)

    print('Done.')
