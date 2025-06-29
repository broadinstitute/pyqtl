import numpy as np
import pandas as pd
import scipy.stats
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from matplotlib.colors import hsv_to_rgb, ListedColormap, LogNorm
import seaborn as sns
import scipy.cluster.hierarchy as hierarchy
from cycler import cycler
from collections.abc import Iterable
import copy
import itertools

from . import stats
from . import map as qtl_map


def setup_figure(aw=4.5, ah=3, xspace=[0.75,0.25], yspace=[0.75,0.25],
                 colorbar=None, ds=0.15, cw=0.12, ct=0, ch=None,
                 margin_axes=None, mx=0.5, dx=0.15, my=0.5, dy=0.15, polar=False):
    """
    """
    dl, dr = xspace
    db, dt = yspace
    fw = dl + aw + dr
    fh = db + ah + dt
    if margin_axes in ['x', 'both']:
        fw += dx + mx
    if margin_axes in ['y', 'both']:
        fh += dy + my
    fig = plt.figure(facecolor='none', figsize=(fw,fh))
    axes = [fig.add_axes([dl/fw, db/fh, aw/fw, ah/fh], facecolor='none', zorder=1, polar=polar)]
    if margin_axes in ['y', 'both']:
        axes.append(fig.add_axes([dl/fw, (db+ah+dy)/fh, aw/fw, my/fh], sharex=axes[0], facecolor='none', zorder=0))
    if margin_axes in ['x', 'both']:
        axes.append(fig.add_axes([(dl+aw+dx)/fw, db/fh, mx/fw, ah/fh], sharey=axes[0], facecolor='none', zorder=0))
        if colorbar != 'horizontal':
            dl += dx + mx
    if colorbar == 'horizontal':
        if ch is None:
            ch = 0.4 * aw
        axes.append(fig.add_axes([(dl+aw-ch)/fw, (db+ah+ds)/fh, ch/fw, cw/fh], facecolor='none'))
    elif colorbar is not None and colorbar != False:  # vertical
        if ch is None:
            ch = 0.4 * ah
        axes.append(fig.add_axes([(dl+aw+ds)/fw, (db+ah-ch-ct)/fh, cw/fw, ch/fh], facecolor='none'))

    if len(axes) == 1:
        axes = axes[0]
    return axes


#     if not box:
#         ax.spines['left'].set_position(('outward', 6))
#         ax.spines['bottom'].set_position(('outward', 6))
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#     ax.tick_params(axis='both', which='both', direction='out', labelsize=fontsize)

def get_axgrid(nr, nc, ntot=None, sharex=False, sharey=False,
               x_offset=6, y_offset=6, margins=None, polar=False,
               background_axes=False,
               dl=0.5, aw=2, dx=0.75, dr=0.25,
               db=0.5, ah=2, dy=0.75, dt=0.25,
               colorbar=None, ds=0.15, cw=0.15, ct=0, ch=None,
               tri=None, fontsize=10, hide=['top', 'right']):
    """
    """
    if ntot is None:
        ntot = nr * nc

    if not isinstance(aw, Iterable):
        aw = nc * [aw]

    if not isinstance(polar, Iterable):
        polar = ntot * [polar]

    fw = dl + sum(aw) + (nc-1)*dx + dr
    fh = db + nr*ah + (nr-1)*dy + dt
    fig = plt.figure(figsize=(fw,fh), facecolor='none')
    axes = []
    n = 0

    if tri is None:
        si = lambda x: 0
    elif tri == 'upper':
        si = lambda x: x

    for j in range(nr):
        for i in range(si(j), nc):
            if n < ntot:
                ax = fig.add_axes([(dl+sum(aw[:i])+i*dx)/fw, (db+(nr-j-1)*(ah+dy))/fh, aw[i]/fw, ah/fh], facecolor='none', zorder=0, polar=polar[n],
                                  sharex=axes[0] if sharex and n>0 else None,
                                  sharey=axes[0] if sharey and n>0 else None)
                if not polar[n]:
                    format_plot(ax, fontsize=fontsize, hide=hide, x_offset=x_offset, y_offset=y_offset)
                ax.margins(margins)
                axes.append(ax)
                n += 1

    if ch is None:
        ch = ah/2

    # add axes in background for plotting overlays
    if background_axes:
        bax = fig.add_axes([dl/fw, db/fh, (sum(aw) + (nc-1)*dx)/fw, (nr*ah + (nr-1)*dy)/fh],
                           facecolor='none', zorder=-1, label='background',
                           sharex=axes[0] if sharex and nc == 1 else None,
                           sharey=axes[0] if sharey and nr == 1 else None)
        format_plot(bax, hide=['top', 'right', 'bottom', 'left'])
        hide_ticks(bax)
        bax.margins(0)
    else:
        bax = None

    # add colorbars
    if isinstance(colorbar, Iterable):
        cax = []
        for k in colorbar:
            i = k // nc  # row
            j = k - i*nc  # col
            cax.append(fig.add_axes([(dl+sum(aw[:j+1])+j*dx+ds)/fw, (db+(nr-i)*ah+(nr-i-1)*dy-ch-ct)/fh, cw/fw, ch/fh]))
    elif colorbar == True:
        cax = fig.add_axes([(dl+sum(aw)+(nc-1)*dx+ds)/fw, (db+nr*ah+(nr-1)*dy-ch-ct)/fh, cw/fw, ch/fh])
    else:
        cax = None
    r = [axes]
    if cax is not None:
        r.append(cax)
    if bax is not None:
        r.append(bax)
    if len(r) == 1:
        r = r[0]
    else:
        r = tuple(r)
    return r


def hide_ticks(ax, axis='both'):
    if axis in ['x', 'both']:
        plt.setp(ax.get_xticklabels(), visible=False)
        for line in ax.xaxis.get_ticklines():
            line.set_markersize(0)
            line.set_markeredgewidth(0)
    if axis in ['y', 'both']:
        plt.setp(ax.get_yticklabels(), visible=False)
        for line in ax.yaxis.get_ticklines():
            line.set_markersize(0)
            line.set_markeredgewidth(0)


def format_plot(ax, tick_direction='out', tick_length=4, hide=['top', 'right'],
                hide_spines=True, lw=1, fontsize=10,
                equal_limits=False, x_offset=0, y_offset=0, vmin=None):

    # ax.autoscale(False)
    for i in ['left', 'bottom', 'right', 'top']:
        ax.spines[i].set_linewidth(lw)

    ax.tick_params(axis='both', which='both', direction=tick_direction, labelsize=fontsize)

    # set tick positions
    if 'top' in hide and 'bottom' in hide:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    elif 'top' in hide:
        ax.get_xaxis().set_ticks_position('bottom')
    elif 'bottom' in hide:
        ax.get_xaxis().set_ticks_position('top')
    else:
        ax.get_xaxis().set_ticks_position('both')

    if 'left' in hide and 'right' in hide:
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    elif 'left' in hide:
        ax.get_yaxis().set_ticks_position('right')
    elif 'right' in hide:
        ax.get_yaxis().set_ticks_position('left')
    elif len(hide) == 0:
        ax.get_xaxis().set_ticks_position('bottom')
        ax.get_yaxis().set_ticks_position('left')
    else:
        ax.get_yaxis().set_ticks_position('both')

    if hide_spines:
        for i in hide:
            ax.spines[i].set_visible(False)

    # adjust tick size
    for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
        line.set_markersize(tick_length)
        line.set_markeredgewidth(lw)

    for line in (ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True)):
        line.set_markersize(tick_length/2)
        line.set_markeredgewidth(lw/2)

    ax.spines['left'].set_position(('outward', y_offset))
    ax.spines['bottom'].set_position(('outward', x_offset))

    if equal_limits:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lims = [np.minimum(xlim[0], ylim[0]), np.maximum(xlim[1], ylim[1])]
        if vmin is not None:
            lims[0] = vmin
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    # ax.autoscale(True)  # temporary fix?


def plot_vlines(ax, pos, color, linestyle='-', alpha=1, lw=1, ylim=None, ymax=None, zorder=None):
    """Plot vertical lines spanning ylim"""
    if ylim is None:
        if ymax is not None:
            ylim = [0, ymax]
        else:
            ylim = ax.get_ylim()

    kwargs = {'c':color, 'lw':lw, 'alpha':alpha, 'zorder':0, 'linestyle':linestyle, 'zorder':zorder}
    npos = len(pos)
    if npos > 1:
        x = np.tile(np.array(pos), [2, 1])
        y = np.tile(ylim, [npos, 1]).T
        ax.plot(x, y, **kwargs)
    else:
        ax.plot([pos[0]]*2, ylim, **kwargs)


def plot_qtl(g, p, label_s=None, label_colors=None, split=False, split_colors=None, covariates_df=None,
            legend_text=None, show_pval=False, show_slope=False, normalized=False, loc=None, ax=None, color=[0.5]*3,
            variant_id=None, jitter=0, bvec=None, boxplot=False, xlabel=None, clip_on=True,
            aw=2, ah=2, dr=0.25, dt=0.25, db=0.5, dl=0.75, width=0.75, lw=1,
            ylabel='Normalized expression', title=None, show_counts=True):
    """"""

    assert p.index.equals(g.index)

    if show_pval or show_slope:
        stats_s = qtl_map.calculate_association(g, p, covariates_df=covariates_df).iloc[0]

    if covariates_df is not None:
        # only residualize the phenotype for plotting
        p = stats.residualize(p.copy(), covariates_df.loc[p.index])

    qtl_df = pd.concat([g.round().astype(int), p], axis=1)
    qtl_df.columns = ['genotype', 'phenotype']
    if label_s is not None:
        qtl_df = pd.concat([qtl_df, label_s], axis=1, sort=False)

    if ax is None:
        ax = setup_figure(aw, ah, xspace=[dl, dr], yspace=[db, dt])

    if not normalized:
        if split and label_s is not None:
            if split_colors is None:
                split_colors = [
                    hsv_to_rgb([0.025, 1, 0.8]),
                    hsv_to_rgb([0.575, 1, 0.8])
                ]
            pal = sns.color_palette(split_colors)
            i = qtl_df.columns[2]
            sns.violinplot(x="genotype", y="phenotype", hue=i, hue_order=sorted(qtl_df[i].unique()),
                           data=qtl_df, palette=pal, ax=ax, order=[0,1,2], density_norm='width', cut=0, linewidth=lw, width=width)
            l = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=8, handlelength=0.6, ncol=2, handletextpad=0.5, labelspacing=0.33)
            l.set_title(None)
        else:
            sns.violinplot(x="genotype", y="phenotype", data=qtl_df, width=width, density_norm='width',
                           cut=0, ax=ax, order=[0,1,2], color=color, linewidth=lw, clip_on=clip_on)
    else:
        pass
        # if labels is not None:
        #     ax.scatter(g, p, c=labels, cmap=colors.LinearSegmentedColormap.from_list('m', label_colors), alpha=0.8, s=25, edgecolors='none')
        # else:
        #     # ax.scatter(g, p, c=hsv_to_rgb([0.55,0.8,0.8]), alpha=0.8, s=25, edgecolors='none')
        #     ax.scatter(g, p, c='k', alpha=0.66, s=25, edgecolors='none')

    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12)
    format_plot(ax, lw=1, fontsize=9, x_offset=6, y_offset=6)
    ax.set_xlim([-0.5,2.5])
    ax.spines['bottom'].set_bounds([0, 2])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=5, nbins=5))

    if show_slope:
        x = np.array([-0.5, 2.5])
        r = scipy.stats.linregress(g, p)
        ax.plot(x, r.intercept + x*r.slope, linestyle=(0, (3, 2)), lw=2, color='tab:blue')

    if title is not None:
        ax.set_title(title, fontsize=12)

    if variant_id is not None:
        ref, alt = variant_id.split('_')[2:4]
        labels = [
            f'{ref}/{ref}',
            f'{ref}/{alt}',
            f'{alt}/{alt}',
        ]
    else:
        labels = [0, 1, 2]

    if show_counts:
        if not split:
            gcounts = g.astype(int).value_counts()
            labels = [f"{v}\n{gcounts.get(k, 0)}" for k,v in enumerate(labels)]
        else:
            var_s = qtl_df[qtl_df.columns[2]]
            c = np.unique(var_s)
            assert len(c) == 2

            gcounts1 = g[var_s == c[0]].value_counts().reindex(np.arange(3), fill_value=0)
            gcounts2 = g[var_s == c[1]].value_counts().reindex(np.arange(3), fill_value=0)
            labels = [f"{v}\n({gcounts1[k]},{gcounts2[k]})" for k,v in enumerate(labels)]
    ax.set_xticks(range(len(labels)), labels)

    if show_pval:
        if stats_s['slope'] > 0:
            ax.text(0.05, 1, f"P = {stats_s['pval_nominal']:.2g}", va='top', ha='left', transform=ax.transAxes, fontsize=11)
        else:
            ax.text(0.95, 1, f"P = {stats_s['pval_nominal']:.2g}", va='top', ha='right', transform=ax.transAxes, fontsize=11)

    return ax


def plot_interaction(p, g, i, variant_id=None, annot=None, covariates_df=None, lowess=None,
                     xlabel=None, ylabel=None, title=None, alpha=0.8, s=20, fontsize=14,
                     ah=3, aw=3):
    """
    Plot interaction QTL

    Model:
      p = b0 + b1*g + b2*i + b3*gi

    Args:
      lowess: fraction of data to use [0,1]
    """

    assert p.index.equals(g.index) and p.index.equals(i.index)

    if covariates_df is not None:
        assert p.index.equals(covariates_df.index)
        X = np.c_[len(g)*[1],g,i,g*i,covariates_df]
    else:
        X = np.c_[len(g)*[1],g,i,g*i]
    b,_,_,_ = np.linalg.lstsq(X, p, rcond=None)

    if variant_id is not None:
        ref, alt = variant_id.split('_')[2:4]
    else:
        ref, alt = 'ref', 'alt'
    labels = {
        0: f'{ref}/{ref}',
        1: f'{ref}/{alt}',
        2: f'{alt}/{alt}',
    }

    ax = setup_figure(ah, aw)
    ax.margins(0.02)

    custom_cycler = cycler('color', [
        # hsv_to_rgb([0.55,1,0.8]),
        # sns.color_palette("Paired")[7],  # orange
        # hsv_to_rgb([0,1,0.8]),
        sns.color_palette("husl", 8)[5], # blue
        sns.color_palette("Paired")[7],  # orange
        sns.color_palette("Paired")[3],  # green
    ])
    ax.set_prop_cycle(custom_cycler)

    gorder = [0,1,2]
    # gorder = [2,1,0]
    # mu = [p[g==g0].mean() for g0 in np.unique(g)]
    # if mu[0]<mu[2]:
    #     gorder = gorder[::-1]
    for d in gorder:
        ix = g[g == d].index
        ax.scatter(i[ix], p[ix], s=s, alpha=alpha, edgecolor='none', label=labels[d], clip_on=False)
        if lowess is not None:
            lw = sm.nonparametric.lowess(p[ix], i[ix], lowess)
            ax.plot(lw[:, 0], lw[:, 1], '--', lw=2)
    format_plot(ax, fontsize=12)
    xlim = np.array(ax.get_xlim())
    for d in gorder:  # regression lines
        y = lambda x: b[0] + b[1]*d + b[2]*x + b[3]*d*x
        ax.plot(xlim, y(xlim), '-', lw=1.5)

    leg = ax.legend(fontsize=12, labelspacing=0.25, handletextpad=0, borderaxespad=0, handlelength=1.5)
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=3, integer=True, nbins=4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=3, integer=True, nbins=4))

    if xlabel is None:
        xlabel = i.name
    if ylabel is None:
        try:
            ylabel = annot.get_gene(p.name).name
        except:
            pass
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is None:
        title = variant_id
    ax.set_title(title, fontsize=fontsize)
    ax.spines['bottom'].set_position(('outward', 6))
    ax.spines['left'].set_position(('outward', 6))
    return ax


def plot_ld(ld_df, ld_threshold=0.1, s=0.25, alpha=1, yscale=3, xunit=1e6, ld_bounds=None,
            cmap=plt.cm.Greys, start_pos=None, end_pos=None, ax=None, cax=None,
            clip_on=False, rasterized=True):
    """"""

    assert ld_df.index.equals(ld_df.columns)
    ld_df = ld_df.copy()
    pos = ld_df.index.map(lambda x: int(x.split('_')[1]))

    # drop duplicates (multi-allelic sites)
    m = ~pos.duplicated()
    ld_df = ld_df.loc[m, ld_df.columns[m]]

    variant_df = pd.DataFrame(index=ld_df.index)
    variant_df['chr'] = variant_df.index.map(lambda x: x.split('_')[0])
    variant_df['pos'] = pos[m]

    if start_pos is None:
        start_pos = variant_df['pos'].iloc[0]
    if end_pos is None:
        end_pos = variant_df['pos'].iloc[-1]

    ld_df.rename(index=variant_df['pos'], columns=variant_df['pos'], inplace=True)
    ld_df.columns.name = 'col'
    ld_df.index.name = 'row'
    ld_df.values[np.triu_indices(ld_df.shape[0])] = np.nan

    v = ld_df.stack().reset_index()
    v = v[v[0] >= ld_threshold]
    X = v[['row', 'col']].copy().values.T
    X[1,:] -= start_pos
    x0 = np.array([[start_pos, 0]]).T
    R = np.array([[1, 1], [-1, 1]])/np.sqrt(2)

    # set up figure
    if ax is None:
        pad = 0.1
        dl = pad
        aw = 8
        dr = 0.5
        db = 0.5
        ah = aw/yscale  # must also scale ylim below
        dt = pad
        fw = dl+aw+dr
        fh = db+ah+dt
        ds = 0.1
        fig = plt.figure(facecolor=(1,1,1), figsize=(fw,fh))
        ax = fig.add_axes([dl/fw, db/fh, aw/fw, ah/fh])
        cax = fig.add_axes([(dl+aw+ds)/fw, db/fh, 0.1/fw, 0.8/fh])

    # plot
    X = np.dot(R, X-x0)/np.sqrt(2) + x0
    order = np.argsort(v[0])
    h = ax.scatter(X[0,order]/xunit, X[1,order]/xunit, s=s, c=v[0].iloc[order], marker='D', clip_on=clip_on,
                   alpha=alpha, edgecolor='none', cmap=cmap, vmin=0, vmax=1, rasterized=rasterized)

    if cax is not None:
        hc = plt.colorbar(h, cax=cax)
        hc.set_label('$\mathregular{R^2}$', fontsize=12, rotation=0, ha='left', va='center')
        hc.locator = ticker.MaxNLocator(min_n_ticks=3, nbins=2)
    xlim = np.array([start_pos, end_pos]) / xunit
    ax.set_xlim(xlim)
    ax.set_ylim([-np.diff(xlim)[0]/yscale, 0])

    for s in ['left', 'top', 'right']:
        ax.spines[s].set_visible(False)
    ax.set_yticks([])

    ax.set_xlabel(f"Position on {variant_df['chr'].iloc[0]} (Mb)", fontsize=14)

    if ld_bounds is not None:
        ci = (ld_bounds[:-1] + ld_bounds[1:]) / 2  # center position for each block
        y = -np.diff(ld_bounds) / 2 / xunit
        yi = np.array([i for i in itertools.chain(*itertools.zip_longest(np.zeros(len(ld_bounds)), y)) if i is not None])
        xi = np.array([i for i in itertools.chain(*itertools.zip_longest(ld_bounds, ci)) if i is not None])
        ax.plot(xi/xunit, yi, linestyle=(0, (4, 3)), lw=1, color='tab:red')

    return ax


def plot_locus_summary(region_str, tracks_dict=None, ld_df=None, ld_bounds=None, coverage_cat=None,
                       track_colors=None, labels=None, order=None,
                       pip_df=None, pip_order=None, pip_colors=None, pip_legend=False,
                       gene=None, ld_marker_size=1, aw=6, ah=4, dl=2, dr=1, ph=0.1, gh=0.15):
    """
    Visualization of genetic locus, combining coverage tracks (e.g., ATAC-seq),
    variants (e.g., fine-mapped QTLs), genes/transcripts, and LD.

    Inputs:
      pip_df: variants with PIPs from fine-mapping
      tracks_dict: tracks
      ld_df:  LD matrix
    """

    if isinstance(tracks_dict, dict):
        ntracks = len(tracks_dict)
        ah = ntracks*0.2
    elif isinstance(tracks_dict, pd.DataFrame):
        ntracks = tracks_dict.shape[1]
    else:
        ntracks = 0
        ah = 0

    if gene is not None:
        if not isinstance(gene, Iterable):
            gene = [gene]
        gh = gh * len(gene)  # gene model
    else:
        gh = 0

    db = 0.5
    ldh = 1  # LD plot
    ds0 = 0
    ds = 0.1
    if pip_df is None:
        ph = 0
    elif isinstance(pip_df, pd.DataFrame):
        pip_df = [pip_df]
    pah = [ph*df['trait_id'].nunique() for df in pip_df]

    dt = 0.25
    fw = dl + aw + dr
    fh = db + ldh + ds0 + gh + ds + sum(pah) + ds*(len(pah)-1) + ds + ah + dt
    fig = plt.figure(figsize=(fw,fh))
    axes = []
    if tracks_dict is not None:
        ax =  fig.add_axes([dl/fw, (db+ldh+ds0+gh+ds+sum(pah)+len(pah)*ds)/fh, aw/fw, ah/fh], facecolor='none')
        axes.append(ax)
    if pip_df is not None:
        faxes = []
        for k,h in enumerate(pah):
            fax = fig.add_axes([dl/fw, (db+ldh+ds0+gh+ds+(len(pah)-1-k)*ds+sum(pah[k+1:]))/fh, aw/fw, h/fh],
                               facecolor='none', sharex=axes[0] if len(axes)>0 else None)
            faxes.append(fax)
            axes.append(fax)
    if gene is not None:
        gax = fig.add_axes([dl/fw, (db+ldh+ds0)/fh, aw/fw, gh/fh], facecolor='none')
        axes.append(gax)
    if ld_df is not None:
        lax = fig.add_axes([dl/fw, db/fh, aw/fw, ldh/fh], facecolor='none', sharex=axes[0] if len(axes)>0 else None)
        lcax = fig.add_axes([(dl+aw+ds)/fw, (db+ldh/2)/fh, 0.1/fw, ldh/2/fh], facecolor='none')
        axes.append(lax)

    chrom, start_pos, end_pos = re.split(':|-', region_str)
    start_pos = int(start_pos)
    end_pos = int(end_pos)
    x = np.arange(start_pos, end_pos+1) / 1e6

    if order is not None:
        labels = order
    elif tracks_dict is not None:
        labels = list(tracks_dict.keys())

    if isinstance(tracks_dict, dict):
        #maxv = np.max([np.max(tracks_dict[k]) for k in labels])
        for k, label in enumerate(labels):
            c = tracks_dict[label]
            y0 = (ntracks-1-k) * np.ones(len(x))  # vertical offset
            # if label in track_colors:
            #     color = track_colors[label]
            # else:
            #     color = 'k'
            # ax.fill_between(x, 0.8*c/np.nanmax(c)+y0, y0,
            # ax.fill_between(x, 0.8*c/15+y0, y0,
            maxv = np.max(c)
            ax.fill_between(x, 0.9*c/maxv + y0, y0, antialiased=False, linewidth=1,
                            facecolor=track_colors.get(label, 'k') if track_colors is not None else 'k',
                            clip_on=True, rasterized=True)

        ax.set_yticks(np.arange(ntracks))
        ax.set_yticklabels([i.replace('_', ' ') for i in labels[::-1]], fontsize=8, va='bottom')
        ax.spines['left'].set_bounds((0, ntracks-1))
        ax.spines['left'].set_position(('outward', 6))
        format_plot(ax, fontsize=8, hide=['top', 'right', 'bottom'], y_offset=6)
        ax.set_ylim([-0.5, ntracks-0.5])

    elif isinstance(tracks_dict, pd.DataFrame):  # plot as heatmap
        ax.invert_yaxis()
        ax.imshow(np.log10(tracks_dict.values.T),
                  #extent=(),
                  aspect='auto', interpolation='none')
        ax2 = fig.add_axes([(dl-0.2)/fw, (db+ldh+ds0+gh+ds+ph+ds)/fh, 0.1/fw, ah/fh], sharey=ax)
        # format_plot(ax, fontsize=8, hide=['top', 'right', 'bottom', 'left'])
        format_plot(ax2, fontsize=8, hide=['top', 'right', 'bottom', 'left'])
        ax2.set_xticks([])

        cohort_index_dict = {i:k for k,i in enumerate(np.unique(coverage_cat))}
        # if cohort_colors is None:
        n = len(cohort_index_dict)
        cmap = ListedColormap(plt.cm.get_cmap('Spectral', n)(np.arange(n)), 'indexed')
        # else:
        #     cmap = ListedColormap(np.stack(pd.Series(cohort_index_dict).sort_values().index.map(cohort_colors)))

        # if orientation == 'vertical':
        ax2.imshow(coverage_cat.map(cohort_index_dict).values.reshape(-1,1), aspect='auto', cmap=cmap)
        # else:
        #     ax.imshow(cohort_s.map(cohort_index_dict).values.reshape(1,-1), aspect='auto', cmap=cmap)
        #
        # if lax is None:
        #     lax = ax
        for k,i in cohort_index_dict.items():
            ax.scatter(np.nan, np.nan, marker='s', c=[cmap(i)], label=f'{k}')
        # if legend:
        ax.legend(loc='upper right', borderaxespad=None, bbox_to_anchor=(-0.05,1), handlelength=1, title='Taxonomy', fontsize=8)
        ax.set_ylim([ntracks-0.5, -0.5])
        ax.set_yticks([])

    if tracks_dict is not None:
        plt.setp(ax.get_xticklabels(), visible=False)
    if len(axes) > 0:
        axes[0].set_xlim([x[0], x[-1]])

    if pip_df is not None:
        for k, df in enumerate(pip_df):
            fax = faxes[k]
            pip_order = df['trait_category'].unique()  # TODO: add back as option?
            i = 0
            traits = []
            for category_id in pip_order:
                cdf = df[df['trait_category'] == category_id]
                for i,(trait_id,gdf) in enumerate(cdf.groupby('trait_id', observed=True), i+1):
                    traits.append(trait_id)
                    fax.scatter(gdf['position']/1e6, [i-1]*gdf.shape[0], s=30*gdf['pip'],
                                color=pip_colors.get(category_id, 'k') if pip_colors is not None else 'k', edgecolor='none', clip_on=False)
                if cdf.shape[0] > 0:
                    fax.scatter(np.nan, np.nan, s=20, color=pip_colors.get(category_id, 'k') if pip_colors is not None else 'k', label=category_id, edgecolor='none')
            fax.invert_yaxis()
            fax.set_yticks(np.arange(len(traits)))

            fax.spines['bottom'].set_visible(False)
            fax.spines['top'].set_visible(False)
            fax.spines['left'].set_position(('outward', 6))
            fax.spines['left'].set_bounds((0, i-1))
            fax.spines['right'].set_visible(False)
            fax.set_xlim([x[0], x[-1]])
            fax.set_yticklabels(traits, fontsize=7)
            plt.setp(fax.get_xticklabels(), visible=False)
            for line in fax.xaxis.get_ticklines():
                line.set_markersize(0)
                line.set_markeredgewidth(0)

            if pip_legend == True:
                fax.legend(loc='upper left', borderaxespad=0, borderpad=0.25, bbox_to_anchor=(1.01,1), fontsize=8, handlelength=0.75, handletextpad=0.5, labelspacing=0)


    if gene is not None:
        for k,g in enumerate(gene[::-1]):
            g.plot(ax=gax, max_intron=1e9, pc_color='k', nc_color='k', ec='none', yoffset=k, scale=0.33, clip_on=True)
            gax.annotate(g.name, (g.end_pos, k),
                         xytext=(5,0), textcoords='offset points',
                         va='center', ha='left', fontsize=10)
        xlim = np.array([x[0], x[-1]])*1e6
        gax.set_xlim(xlim)
        gax.set_xticks([])
        gax.set_yticks([])
        gax.spines['bottom'].set_visible(False)
        gax.spines['top'].set_visible(False)
        gax.spines['left'].set_visible(False)
        gax.spines['right'].set_visible(False)

    if ld_df is not None:
        format_plot(lax, fontsize=10)
        plot_ld(ld_df, ld_bounds=ld_bounds, start_pos=start_pos, end_pos=end_pos, cmap=plt.cm.Greys,
                s=ld_marker_size, clip_on=True, yscale=aw/ldh, ax=lax, cax=lcax)

    if len(axes) > 0:
        axes[-1].set_xlabel(f"Position on {chrom} (Mb)", fontsize=12)

    return axes


def plot_effects(dfs, args, ax=None,
                 xspace=[2.25,2,0.5], yspace=[0.5,3,0.5], xlim=None,
                 xlabel='log$\mathregular{_{2}}$(Fold enrichment)', ylabel=None):
    """"""

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        args = [args]
    ix = dfs[0].index.tolist()
    for df in dfs[1:]:
        assert np.all(df.index == ix)

    if ax is None:
        dl, aw, dr = xspace
        db, ah, dt = yspace

        fw = dl + aw + dr
        fh = db + ah + dt
        fig = plt.figure(facecolor=(1,1,1), figsize=(fw,fh))
        ax = fig.add_axes([dl/fw, db/fh, aw/fw, ah/fh])

    if xlim is not None:
        ax.set_xlim(xlim)
    y = np.arange(len(ix))
    ax.set_ylim([y[0]-0.5, y[-1]+0.5])

    ax.plot([0,0], [-0.5,len(ix)-0.5], '--', color=[0.33]*3, lw=1, zorder=-8)

    n = len(dfs)
    d = 0
    if n == 2:
        # d = [-0.25, 0.25]
        # d = [-0.2, 0.2]
        d = [-0.15,0.15]
    elif n == 3:
        d = [-0.25, 0, 0.25]
    elif n == 4:
        d = [-0.25, -0.15, 0.15, 0.25]

    for k,df in enumerate(dfs):
        mean_col = df.columns[0]
        ci_cols = df.columns[1:]
        delta = (df[ci_cols].T - df[mean_col]).abs()
        ax.errorbar(df[mean_col], y+d[k], xerr=delta.values, **args[k])

    if xlim is None:
        xlim = ax.get_xlim()
    for i in y:
        if i % 2 == 0:
            c = [0.95]*3
            c = [1]*3
        else:
            c = [0.75]*3
            c = [0.9]*3
        patch = patches.Rectangle((xlim[0], i-0.5), np.diff(xlim), 1, fc=c, zorder=-10)
        ax.add_patch(patch)

    ax.set_xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(ix)

    ax.invert_yaxis()
    return ax


def _qq_scatter(ax, pval, ntests=None, label=None, c=None, zorder=None,
                max_values=100000, step=1000, is_sorted=False, args=None):
    """"""
    if ntests is None:
        ntests = len(pval)
    n = len(pval)
    if n > max_values:
        xi = np.array(list(range(1, max_values+1)) + list(range(max_values+step, n+step, step)))
    else:
        xi = np.arange(1, n+1)
    x = -np.log10(xi/(ntests+1))

    if not is_sorted:
        log_pval_sorted = -np.log10(np.sort(pval))
    else:
        log_pval_sorted = -np.log10(pval)

    ax.scatter(x, list(log_pval_sorted[:max_values]) + list(log_pval_sorted[max_values::step]),
               c=c, zorder=zorder, label=label, **args)


def qqplot(pval, pval_null=None, ntests=None, ntests_null=None, max_values=100000, step=1000, is_sorted=False,
           title='', labels=None, fontsize=12, ax=None, equal_axes=False):
    """QQ-plot

      ntests: total number of tests if not equal to len(pval),
              e.g., if only tail of p-value distribution is provided
    """
    if labels is None:
        labels = ['', '']
    if ntests is None:
        ntests = len(pval)

    if ax is None:
        ax = setup_figure(2,2)
    ax.margins(x=0.02, y=0.05)
    args = {'s':16, 'edgecolor':'none', 'clip_on':False, 'alpha':1, 'rasterized':True}

    # Q-Q plot for pval
    _qq_scatter(ax, pval, ntests=ntests, label=labels[0], c=None, zorder=30,
                max_values=max_values, step=step, is_sorted=is_sorted, args=args)

    # Q-Q plot for null
    if pval_null is not None:
        _qq_scatter(ax, pval_null, ntests=ntests_null, label=labels[1], c=[[0.5]*3], zorder=20,
                    max_values=max_values, step=step, is_sorted=is_sorted, args=args)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=5, nbins=4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=5, nbins=4))

    ax.set_xlabel('Expected -log$\mathregular{_{10}}$(p-value)', fontsize=fontsize)
    ax.set_ylabel('Observed -log$\mathregular{_{10}}$(p-value)', fontsize=fontsize)
    format_plot(ax, fontsize=fontsize-2)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if equal_axes:
        m = np.maximum(xlim[1], ylim[1])
        ax.set_xlim([0, m])
        ax.set_ylim([0, m])
    else:
        ax.set_xlim([0, xlim[1]])
        ax.set_ylim([0, ylim[1]])

    # plot confidence interval
    ci = 0.95
    xi = np.linspace(1, ntests, 100000)
    x = -np.log10(xi/(ntests+1))
    clower = -np.log10(scipy.stats.beta.ppf((1-ci)/2, xi, xi[::-1]))
    cupper = -np.log10(scipy.stats.beta.ppf((1+ci)/2, xi, xi[::-1]))
    ax.fill_between(x, cupper, clower, color=[[0.8]*3], clip_on=True, rasterized=True)
    b = -np.log10([ntests/(ntests+1), 1/(ntests+1)])
    ax.autoscale(False)
    ax.plot(b, b, '--', lw=1, color=[0.2]*3, zorder=50, clip_on=True)

    ax.spines['left'].set_position(('outward', 6))
    ax.spines['bottom'].set_position(('outward', 6))
    ax.set_title(f'{title}', fontsize=12)
    if labels[0] != '':
        ax.legend(loc='upper left', fontsize=10, handlelength=0.5, handletextpad=0.33)
    return ax


class CohortLabel(object):
    def __init__(self, cohort_s, cmap=None, colors=None, label_pos='left', vmin=None, vmax=None, bad_color=None):
        assert cmap is not None or colors is not None
        assert not cohort_s.index.duplicated().any()
        if cohort_s.dtype == 'O':
            cohort_s = cohort_s.astype('category')
        self.cohort_s = cohort_s
        if cmap is not None and bad_color is not None:
            cmap = copy.copy(cmap)
            cmap.set_bad(bad_color, 1)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.name = cohort_s.name
        self.label_pos = label_pos
        self.colors = colors

        if cohort_s.dtype.name == 'category':
            # get numerical index
            self.values_s = cohort_s.astype(str).map({j:i for i,j in enumerate(cohort_s.cat.categories)})
            if colors is None:
                n = len(cohort_s.cat.categories)
                colors = cmap(np.linspace(0, 1, np.maximum(n, 5)))
                self.colors = {k:v for k,v in zip(cohort_s.cat.categories, colors)}
            self.cmap = ListedColormap(cohort_s.cat.categories.map(self.colors))
        else:
            self.values_s = cohort_s

    def plot(self, ix=None, ax=None, show_frame=False):
        if ax is None:
            ax, cax = setup_figure(2, 0.5, colorbar=True, ch=0.5)
            # ax, cax = setup_figure(0.5, 2, colorbar=True, ch=0.5)

        if ix is None:
            x = self.values_s.values
        else:
            x = self.values_s[ix].values

        # detect orientation
        bbox = ax.get_window_extent().transformed(ax.get_figure().dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        if width > height:
            x = x.reshape(1, -1)
        else:
            x = x.reshape(-1, 1)
        h = ax.imshow(x, aspect='auto', cmap=self.cmap, interpolation='none', origin='lower')
        if width > height:
            if self.label_pos == 'left':
                ax.set_ylabel(self.name, fontsize=10, rotation=0, va='center', ha='right')
            elif self.label_pos == 'right':
                ax.yaxis.set_label_position('right')
                ax.set_ylabel(self.name, fontsize=10, rotation=0, va='center', ha='left')
            plt.setp(ax.get_yticklabels(), visible=False)

        if not show_frame:
            for i in ax.spines:
                ax.spines[i].set_visible(False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines():
            line.set_markersize(0)
            line.set_markeredgewidth(0)

        # prepare legend
        if self.cohort_s.dtype.name == 'category':
            for c in self.cohort_s.cat.categories:
                ax.scatter(np.nan, np.nan, c=[self.colors[c]], label=c, s=30, marker='s')


def check_labels(labels):
    if labels is not None:
        if isinstance(labels, CohortLabel):
            labels = [labels]
        else:
            assert all([isinstance(i, CohortLabel) for i in labels])
        n = len(labels)
    else:
        n = 0
    return labels, n


def clustermap(df, Zx=None, Zy=None, cluster=True, aw=3, ah=3, lw=1, vmin=None, vmax=None, cmap=plt.cm.Blues,
               norm=None, origin='lower', dendrogram_pos='top', col_labels=None, row_labels=None,
               fontsize=10, clabel='', cfontsize=10, label_colors=None, colorbar=True, colorbar_orientation='vertical',
               method='average', metric='euclidean', optimal_ordering=False, value_labels=False,
               show_xlabels=False, show_ylabels=False, tick_length=0, rotation=-45, ha='left', va='top',
               tri=False, rasterized=False, count_sort=False,
               show_frame=False, dl=1, dr=1, dt=0.2, lh=0.1, ls=0.01,
               db=1.5, dd=0.4, ds=0.03, ch=1, cw=0.175, dc=0.1, dtc=0):
    """"""
    col_labels, nc = check_labels(col_labels)
    row_labels, nr = check_labels(row_labels)

    if Zx is None and cluster:
        Zx = hierarchy.linkage(df.T, method=method, metric=metric, optimal_ordering=optimal_ordering)
        Zy = hierarchy.linkage(df,   method=method, metric=metric, optimal_ordering=optimal_ordering)
    elif Zy is None and cluster:
        Zy = Zx

    fw = dl + aw + dr + nr*(lh+ls)
    fh = db + ah + ds + dd + dt + nc*(lh+ls)
    fig = plt.figure(figsize=(fw,fh), facecolor='none')
    dl2 = dl + nr*(lh+ls)
    if dendrogram_pos == 'top':
        ax = fig.add_axes([dl2/fw, db/fh, aw/fw, ah/fh])
        # column labels
        tax = []
        for k in range(nc):
            tax.append(
                fig.add_axes([dl2/fw, (db+ah+(k+1)*ls+k*lh)/fh, aw/fw, lh/fh], sharex=ax)
            )
        # row labels
        lax = []
        for k in range(nr):
            lax.append(
                fig.add_axes([(dl+k*(lh+ls))/fw, db/fh, lh/fw, ah/fh], sharey=ax)
            )
        # dendrogram
        dax = fig.add_axes([dl2/fw,         (db+ah+nc*(ls+lh)+ds)/fh, aw/fw, dd/fh])
        axes = [ax, *lax, *tax, dax]
        # colorbar
        if colorbar:
            if colorbar_orientation == 'vertical':
                cax = fig.add_axes([(dl2+aw+dc)/fw, (db+ah-ch-dtc)/fh, cw/fw, ch/fh], label='Colorbar')
            else:
                cax = fig.add_axes([(dl2+aw-ch-dtc)/fw, (db-cw-dc)/fh, ch/fw, cw/fh], label='Colorbar')
            axes.append(cax)
    else:
        dax = fig.add_axes([dl/fw, db/fh, aw/fw, dd/fh])
        ax =  fig.add_axes([dl/fw, (db+dd+ds)/fh, aw/fw, ah/fh])
        axes = [ax, dax]
        if colorbar:
            cax = fig.add_axes([(dl+aw+dc)/fw, (db+dd+ds)/fh, cw/fw, ch/fh])
            axes.append(cax)

    if Zx is not None:
        with plt.rc_context({'lines.linewidth': lw}):
            z = hierarchy.dendrogram(Zx, ax=dax, count_sort=count_sort, orientation='top', link_color_func=lambda k: 'k')
        ix = df.columns[z['leaves']]  # equivalent to hierarchy.leaves_list(Zx) if count_sort=False
    else:
        ix = df.columns
    dax.axis('off')

    if Zy is not None:
        iy = df.index[hierarchy.leaves_list(Zy)]
    elif df.index.equals(df.columns):
        iy = ix
    else:
        iy = df.index

    if dendrogram_pos == 'bottom':
        dax.invert_yaxis()

    df = df.loc[iy, ix].copy()
    if tri:
        if dendrogram_pos == 'top':
            if origin == 'upper':
                df.values[np.tril_indices(df.shape[0], -1)] = np.nan
            else:
                df.values[np.triu_indices(df.shape[0])] = np.nan
        elif dendrogram_pos == 'bottom':
            df.values[np.tril_indices(df.shape[0])] = np.nan

    if value_labels:
        irange = np.arange(df.shape[0])
        jrange = np.arange(df.shape[1])
        for i in irange:
            for j in jrange:
                if not np.isnan(df.values[j,i]):
                    ax.text(i, j, f'{df.values[j,i]:.2f}', ha='center', va='center')

    h = ax.imshow(df.values, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                  interpolation='none', rasterized=rasterized, aspect='auto')
    if show_xlabels:
        ax.set_xticks(np.arange(df.shape[1]))
        if rotation not in [90, -90]:
            ax.set_xticklabels(ix, rotation=rotation, rotation_mode='anchor', ha=ha, va=va, fontsize=fontsize)
        else:
            ax.set_xticklabels(ix, rotation=rotation, ha=ha, va=va, fontsize=fontsize)
        ax.tick_params(axis='x', length=tick_length)
        if show_xlabels == 'top':
            ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    else:
        ax.set_xticks([])

    if show_ylabels:
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_yticklabels(iy, fontsize=fontsize)
        ax.tick_params(axis='y', length=tick_length)
    else:
        ax.set_yticks([])

    # plot labels
    for k in range(nr):
        row_labels[k].plot(ax=lax[k], ix=iy, show_frame=True)
    for k in range(nc):
        col_labels[k].plot(ax=tax[k], ix=ix, show_frame=True)

    if lax:
        plt.setp(ax.get_yticklabels(), visible=False)
        for line in ax.yaxis.get_ticklines():
            line.set_markersize(0)
            line.set_markeredgewidth(0)

    if dendrogram_pos == 'bottom':
        ax.yaxis.tick_right()
    # else:
    #     ax.xaxis.tick_top()

    if label_colors is not None:  # plot color legend
        s = 1.015
        xlim = ax.get_xlim()
        b = xlim[1] - s*np.diff(xlim)
        ax.set_xlim(xlim)
        ax.scatter([b]*df.shape[1], np.arange(df.shape[1]), s=40, c=label_colors.iloc[hierarchy.leaves_list(Zx)], clip_on=False)
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_yticklabels(iy, fontsize=fontsize)
        ax.tick_params(axis='y', pad=12, length=0)

        # s = 1.02
        # ylim = ax.get_ylim()
        # b = ylim[1] - s*np.diff(ylim)
        # ax.set_ylim(ylim)
        # ax.scatter(np.arange(df.shape[1]), [b]*df.shape[1], s=36, c=label_colors[hierarchy.leaves_list(Zx)], clip_on=False)
        # ax.tick_params(axis='x', pad=12)

    # plot colorbar
    if colorbar:
        cbar = plt.colorbar(h, cax=cax, orientation=colorbar_orientation)
        if norm is None:
            cax.locator_params(nbins=4)
        cbar.set_label(clabel, fontsize=cfontsize+1)
        cax.tick_params(labelsize=cfontsize)

    if not show_frame:
        for i in ['left', 'top', 'right', 'bottom']:
            ax.spines[i].set_visible(False)

    plt.sca(ax)
    return axes


def hexdensity(x, y, bounds=None, bins='log', scale='log',
               cmap=None, vmin=None, vmax=None, ax=None, cax=None,
               unit='TPM', entity='genes',
               gridsize=175, fontsize=12, show_corr=True, clip_on=True, rasterized=False):
    """Wrapper for hexbin"""

    if ax is None: # setup new axes
        ax, cax = setup_figure(2, 2, xspace=[0.75, 1], yspace=[0.75, 0.5], colorbar=True, ch=1, cw=0.12)
        ax.margins(0.01)
        format_plot(ax, fontsize=fontsize-2, x_offset=6, y_offset=6)

    if cmap is None:
        cmap = copy.copy(plt.cm.RdYlBu_r)
        cmap.set_bad('w', 1.)

    rho = scipy.stats.spearmanr(x, y)[0]
    x = x.copy()
    y = y.copy()
    nanidx = (x == 0) | (y == 0)
    if any(nanidx):
        x[nanidx] = np.nan
        y[nanidx] = np.nan

    h = ax.hexbin(x, y, bins=bins, xscale=scale, yscale=scale, linewidths=0.1,
                  gridsize=gridsize, cmap=cmap, vmin=vmin, vmax=vmax, mincnt=1, zorder=1,
                  clip_on=clip_on, rasterized=rasterized)

    if bounds is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bounds = [np.minimum(xlim[0], ylim[0]), np.maximum(xlim[1], ylim[1])]
    elif len(bounds) == 2:
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
    else:
        ax.set_xlim(bounds[:2])
        ax.set_ylim(bounds[2:])

    if show_corr:
        t = ax.text(1, 0, r'$\rho$ = {:.2f}'.format(rho), transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=fontsize, zorder=2)
        t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none', boxstyle="round,pad=0.1"))

    if cax is not None:
        hc = plt.colorbar(h, cax=cax, orientation='vertical', ticks=ticker.LogLocator(numticks=4))
        hc.set_label('log$\mathregular{_{10}}$('+entity+')', fontsize=fontsize)

    if isinstance(x, pd.Series):
        ax.set_xlabel(f'{x.name} ({unit})' if unit is not None else f'{x.name}', fontsize=fontsize)
    if isinstance(y, pd.Series):
        ax.set_ylabel(f'{y.name} ({unit})' if unit is not None else f'{y.name}', fontsize=fontsize)

    return ax, cax
