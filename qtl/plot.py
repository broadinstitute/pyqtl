import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.colors import hsv_to_rgb
import seaborn as sns
import scipy.cluster.hierarchy as hierarchy

from . import stats, plot


def setup_figure(aw=4.5, ah=3, xspace=[0.75,0.25], yspace=[0.75,0.25]):
    """
    """
    fw = aw + np.sum(xspace)
    fh = ah + np.sum(yspace)
    fig = plt.figure(facecolor=(1,1,1), figsize=(fw,fh))
    ax = fig.add_axes([xspace[0]/fw, yspace[0]/fh, aw/fw, ah/fh])
    return ax


def format_plot(ax, tick_direction='out', tick_length=4, hide=['top', 'right'],
                hide_spines=True, lw=1, fontsize=8):

    ax.autoscale(False)

    for i in ['left', 'bottom', 'right', 'top']:
        ax.spines[i].set_linewidth(lw)

    ax.tick_params(axis='both', which='both', direction=tick_direction, labelsize=fontsize)

    # set tick positions
    if 'top' in hide and 'bottom' in hide:
        ax.get_xaxis().set_ticks_position('none')
    elif 'top' in hide:
        ax.get_xaxis().set_ticks_position('bottom')
    elif 'bottom' in hide:
        ax.get_xaxis().set_ticks_position('top')
    else:
        ax.get_xaxis().set_ticks_position('both')

    if 'left' in hide and 'right' in hide:
        ax.get_yaxis().set_ticks_position('none')
    elif 'left' in hide:
        ax.get_yaxis().set_ticks_position('right')
    elif 'right' in hide:
        ax.get_yaxis().set_ticks_position('left')
    elif len(hide)==0:
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

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()


def plot_qtl(g, p, label_s=None, label_colors=None, split=False, split_colors=None, covariates_df=None,
            legend_text=None, normalized=False, loc=None, ax=None, color=[0.5]*3,
            variant_id=None, jitter=0, bvec=None, boxplot=False, xlabel=None,
            ylabel='Normalized expression', title=None, genotype_counts=None):

    if covariates_df is not None:
        p = stats.residualize(p.copy(), covariates_df.loc[p.index])

    eqtl_df = pd.concat([g, p], axis=1)
    eqtl_df.columns = ['genotype', 'phenotype']
    if label_s is not None:
        eqtl_df = pd.concat([eqtl_df, label_s], axis=1, sort=False)

    if ax is None:
        ax = plot.setup_figure(2, 2, yspace=[0.75, 0.25])
    ax.spines['bottom'].set_position(('outward', 4))
    ax.spines['left'].set_position(('outward', 4))

    if not normalized:
        if split:
            if split_colors is None:
                split_colors = [
                    hsv_to_rgb([0.025, 1, 0.8]),
                    hsv_to_rgb([0.575, 1, 0.8])
                ]
            pal = sns.color_palette(split_colors)

            i = eqtl_df.columns[2]
            sns.violinplot(x="genotype", y="phenotype", hue=i, hue_order=sorted(eqtl_df[i].unique()),
                           data=eqtl_df, palette=pal, ax=ax, order=[0,1,2], scale='width', dogde=False, linewidth=1, width=0.75)
            l = ax.legend(loc=loc, fontsize=8, handletextpad=0.5, labelspacing=0.33)
            l.set_title(None)
        else:
            colors = [
                color,
            ]
            pal = sns.color_palette(colors)
            sns.violinplot(x="genotype", y="phenotype",
                           data=eqtl_df, palette=pal, ax=ax, order=[0,1,2])
    else:
        pass
        # if labels is not None:
        #     ax.scatter(g, p, c=labels, cmap=colors.LinearSegmentedColormap.from_list('m', label_colors), alpha=0.8, s=25, edgecolors='none')
        # else:
        #     # ax.scatter(g, p, c=hsv_to_rgb([0.55,0.8,0.8]), alpha=0.8, s=25, edgecolors='none')
        #     ax.scatter(g, p, c='k', alpha=0.66, s=25, edgecolors='none')

    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12)
    format_plot(ax, lw=1, fontsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=5, nbins=5))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=3, nbins=3))

    if title is not None:
        ax.set_title(title, fontsize=12)#, pad=8)

    if variant_id is not None:
        ref,alt = variant_id.split('_')[2:4]
        if not split:
            gcounts = g.astype(int).value_counts()
            ax.set_xticklabels([
                '{0}/{0}\n({1})'.format(ref, gcounts[0]),
                '{0}/{1}\n({2})'.format(ref, alt, gcounts[1]),
                '{0}/{0}\n({1})'.format(alt, gcounts[2]),
            ])
        else:
            var_s = eqtl_df[eqtl_df.columns[2]]
            c = sorted(var_s.unique())
            assert len(c)==2

            gcounts1 = g[var_s.loc[g.index]==c[0]].value_counts()
            gcounts2 = g[var_s.loc[g.index]==c[1]].value_counts()
            ax.set_xticklabels([
                '{0}/{0}\n({1},{2})'.format(ref, gcounts1[0], gcounts2[0]),
                '{0}/{1}\n({2},{3})'.format(ref, alt, gcounts1[1], gcounts2[1]),
                '{0}/{0}\n({1},{2})'.format(alt, gcounts1[2], gcounts2[2]),
            ])

    return ax


def plot_effects(dfs, args, ax=None,
                 xspace=[2.25,2,0.5], yspace=[0.5,3,0.5], xlim=None,
                 xlabel='log$\mathregular{_{2}}$(Fold enrichment)', ylabel=None):

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
        args = [args]
    ix = dfs[0].index.tolist()
    for df in dfs[1:]:
        assert np.all(df.index==ix)

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
    if n==2:
        # d = [-0.25, 0.25]
        # d = [-0.2, 0.2]
        d = [-0.15,0.15]
    elif n==3:
        d = [-0.25, 0, 0.25]
    elif n==4:
        d = [-0.25, -0.15, 0.15, 0.25]

    for k,df in enumerate(dfs):
        mean_col = df.columns[0]
        ci_cols = df.columns[1:]
        delta = (df[ci_cols].T - df[mean_col]).abs()
        ax.errorbar(df[mean_col], y+d[k], xerr=delta.values, **args[k])

    if xlim is None:
        xlim = ax.get_xlim()
    for i in y:
        if np.mod(i,2)==0:
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


def clustermap(df, Z=None, aw=3, ah=3, lw=1, vmin=None, vmax=None, cmap=plt.cm.Blues,
               origin='lower', dendrogram_pos='top',
               fontsize=10, clabel='', cfontsize=10, label_colors=None, colorbar_orientation='vertical',
               method='average', metric='euclidean', optimal_ordering=False,
               rotation=-45, ha='left', va='top', tri=False,
               dl=1, dr=1, dt=0.2,
               db=1.5, dd=0.4, ds=0.03, ch=1, cw=0.2, dc=0.15):

    if Z is None:
        Z = hierarchy.linkage(df, method=method, metric=metric, optimal_ordering=optimal_ordering)

    fw = dl+aw+dr
    fh = db+ah+dd+ds+dt

    fig = plt.figure(facecolor=(1,1,1), figsize=(fw,fh))
    if dendrogram_pos=='top':
        ax = fig.add_axes([dl/fw, db/fh, aw/fw, ah/fh])
        dax = fig.add_axes([dl/fw, (db+ah+ds)/fh, aw/fw, dd/fh])
        cax = fig.add_axes([(dl+aw+dc)/fw, (db+ah-ch)/fh, cw/fw, ch/fh])
    else:
        dax = fig.add_axes([dl/fw, db/fh, aw/fw, dd/fh])
        ax =  fig.add_axes([dl/fw, (db+dd+ds)/fh, aw/fw, ah/fh])
        cax = fig.add_axes([(dl+aw+dc)/fw, (db+dd+ds)/fh, cw/fw, ch/fh])

    if Z is not None:
        with plt.rc_context({'lines.linewidth': lw}):
            z = hierarchy.dendrogram(Z, ax=dax,  orientation='top', link_color_func=lambda k: 'k')
        ix = df.columns[hierarchy.leaves_list(Z)]
    else:
        ix = df.columns
    dax.axis('off')

    if dendrogram_pos=='bottom':
        dax.invert_yaxis()

    df = df.loc[ix, ix].copy()
    if tri:
        if dendrogram_pos=='top':
            df.values[np.triu_indices(df.shape[0])] = np.NaN
        elif dendrogram_pos=='bottom':
            df.values[np.tril_indices(df.shape[0])] = np.NaN

    h = ax.imshow(df, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[1]))
    ax.set_xticklabels(ix, rotation=rotation, fontsize=fontsize, ha=ha, va=va)
    ax.set_yticklabels(ix, fontsize=fontsize)

    if dendrogram_pos=='bottom':
        ax.yaxis.tick_right()
    else:
        ax.xaxis.tick_top()

    if label_colors is not None:
        s = 1.015
        xlim = ax.get_xlim()
        b = xlim[1] - s*np.diff(xlim)
        ax.set_xlim(xlim)
        ax.scatter([b]*df.shape[0], np.arange(df.shape[0]), s=48, c=label_colors[hierarchy.leaves_list(Z)], clip_on=False)
        ax.tick_params(axis='y', pad=12)

    cbar = plt.colorbar(h, cax=cax, orientation=colorbar_orientation)
    cax.locator_params(nbins=4)

    cbar.set_label(clabel, fontsize=cfontsize+2)
    cax.tick_params(labelsize=cfontsize)

    for i in ['left', 'top', 'right', 'bottom']:
        ax.spines[i].set_visible(False)
    ax.tick_params(length=0)

    plt.sca(ax)
