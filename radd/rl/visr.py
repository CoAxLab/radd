#!/usr/local/bin/env python
from __future__ import division
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
import matplotlib as mpl
from IPython.display import display, Latex


def plot_traces_rts(p, all_traces, rts, names=['A', 'B', 'C', 'D'], tb=1000):

    rtkeys = np.sort(rts.keys())
    rt_dists = [np.asarray(rts[k])*1e3-(np.mean(p['tr'])*1e3) for k in rtkeys]
    tb = np.ceil(np.max([np.max(rti) if len(rti)>0 else 0 for rti in rt_dists]))+100

    sns.set(style='white', font_scale=1.2)
    f, axes = build_multi_axis(p, tb=tb)
    clrs = sns.color_palette('muted', 5)

    for i in xrange(len(all_traces)):
        for ii, ax in enumerate(axes.flatten()):
            x=np.arange(len(all_traces[i][ii]))
            ax.plot(x, all_traces[i][ii], color=clrs[ii], alpha=.51, lw=.75)

    for i, ax in enumerate(axes.flatten()):
        divider = make_axes_locatable(ax)
        axx = divider.append_axes("top", size=.7, pad=0.01, sharex=ax)
        for spine in ['top', 'left', 'bottom', 'right']:
            axx.spines[spine].set_visible(False)
        axx.set_xticklabels([])
        axx.set_yticklabels([])
        if len(rt_dists[i])<=1:
            continue
        sns.distplot(rt_dists[i], ax=axx, label=k, color=clrs[i])
        text_str='$\mu_{%s}=%.fms$'%(names[i], np.mean(rt_dists[i]))
        ax.text(x[0]-50, np.mean(p['a'])-.06, text_str, fontsize=15)


def plot_summary(outcomes, titles=['Order of Choices','Number of Choices per Card', 'Change in Q(card)', 'Change in P(card)', '$v^G_t$', '$v^N_t$'], plot_traces=False, p=None, tb=1000):

    sns.set_palette('muted')
    f, axes = plt.subplots(3, 2, figsize=(14,16))
    a1, a2, a3, a4, a5, a6 = axes.flatten()
    choices, rts, all_traces, qdict, choicep, vdhist, vihist = outcomes

    names = np.sort(qdict.keys())
    name_labels = [name.upper() for name in names]

    a1.plot(choices, lw=0, marker='o')
    a1.set_ylim(-.5, 3.5); a1.set_yticks(np.arange(4))
    a1.set_yticklabels(name_labels)

    a2.hist(np.asarray(choices))
    a2.set_xticks(np.arange(4))
    a2.set_xticklabels(name_labels)

    for i, n in enumerate(names):
        a3.plot(np.array(qdict[n])*100, label=name_labels[i])
        a4.plot(choicep[n], label=name_labels[i])
        a5.plot(vdhist[n], label=name_labels[i])
        a6.plot(vihist[n], label=name_labels[i])

    a3.legend(loc=0)
    a4.legend(loc=0)
    a5.legend(loc=0)
    a6.legend(loc=0)

    f.subplots_adjust(hspace=.35, wspace=.4)

    for i, ax in enumerate(axes.flatten()):
        ax.set_title(titles[i])
        sns.despine(ax=ax)
    if plot_traces and p is not None:
        plot_traces_rts(p, all_traces, rts, tb=tb)


def get_avg_slope_trace(traces, nalt=4):

    slope_func = lambda x0,x1,y0,y1: (y1-y0)/(x1-x0)
    rise, run, slopes=[], [], []
    for alt_n in range(nalt):
        y0 = 0
        y1 = np.mean([traces_i[alt_n][-1] for traces_i in traces])
        x0 = 0
        x1 = np.ceil(np.mean([traces_i[alt_n].shape[0] for traces_i in traces]))
        rise.append(y1)
        run.append(x1)
        slopes.append(slope_func(x0,x1,y0,y1))

    return rise, run, slopes


def gen_mappable(vals_to_map, cm='rainbow'):

    ncolors = len(vals_to_map)
    clrs = sns.color_palette(cm, ncolors)
    mycmap = mpl.colors.ListedColormap(clrs)

    vmin = np.min(vals_to_map)
    vmax = np.max(vals_to_map)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    scalar_mappable = mpl.cm.ScalarMappable(cmap=mycmap, norm=norm)
    scalar_mappable.set_array([])

    return scalar_mappable, vmin, vmax


def plot_reactivity_strategy(trialsdf, igtdf, cm='rainbow', save=False, pq='P', plot_scatter=False):

    n = trialsdf.a_go.unique().size
    a_go = np.sort(trialsdf.a_go.unique())
    sm, vmin, vmax = gen_mappable(vals_to_map=a_go, cm=cm)
    if plot_scatter:
        f, axes = plt.subplots(2,2, figsize=(12, 9))
        ax1, ax2, ax3, ax4 = axes.flatten()
    else:
        f, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1, ax2 = axes.flatten()

    for grp, grpdf in trialsdf.groupby('a_go'):
        colr = sm.to_rgba(grp)
        ax1.plot(grpdf.vdiff.values, color=colr)
        ax2.plot(grpdf.v_opt_diff.values, color=colr)
        sns.despine()

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(sm, cax)
    sm.colorbar.set_ticks([vmin, vmax])
    cax.set_yticklabels(['{:.3f}'.format(vmin), '{:.3f}'.format(vmax)])

    ax1.set_title('$\Delta_{card} = V_{D}(t)-V_{I}(t)$', fontsize=15)
    ax2.set_title('$\Delta_{OS} = (\Delta_C + \Delta_D) - (\Delta_A + \Delta_B)$', fontsize=14)
    ax1.set_ylabel('Selected Channel Strength ($\Delta_{card}$)', fontsize=12)
    ax2.set_ylabel('Optimal-Subopt Ch. Strength ($\Delta_{OS}$)', fontsize=12)

    for ax in [ax1, ax2]:
        ax.set_xlabel('Trials', fontsize=15)
    if plot_scatter:
        if pq=='P':
            measure = 'Payoff'
        else:
            measure = 'Sensitivity'
        pvals_by_group = igtdf.groupby('a_go').mean().loc[:, pq].values
        reactivity = np.array([grpdf.vdiff.mean() for grp, grpdf in trialsdf.groupby('a_go')])
        strategy = np.array([grpdf.v_opt_diff.mean() for grp, grpdf in trialsdf.groupby('a_go')])
        for i in range(n):
            colr = sm.to_rgba(a_go[i])
            ax3.scatter(pvals_by_group[i], reactivity[i], color=colr, s=15)
            ax4.scatter(pvals_by_group[i], strategy[i], color=colr, s=15)
        ax3.set_ylabel('$\mu \Delta_{card}$', fontsize=15)
        ax4.set_ylabel('$\mu \Delta_{OS}$', fontsize=15)
        ax3.set_xlabel(measure, fontsize=15)
        ax4.set_xlabel(measure, fontsize=15)

    f.subplots_adjust(wspace=.3, hspace=.2)
    plt.tight_layout()
    if save:
        savestr = "_aN" + str(trialsdf.a_no.unique()[0])
        f.savefig(''.join(['reactivity_strategy_', measure, savestr, '.png']), dpi=400)


def build_multi_axis(p, nresp=4, tb=1000):
    bound = p['a']
    onset = p['tr']
    if hasattr(bound, '__iter__'):
        bound = bound[0]
        onset = onset[0]
    # init figure, axes, properties
    f, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
    f.subplots_adjust(hspace=.1, top=.99, bottom=.05)
    w = tb + 40
    h = bound
    start = onset - 80
    axes=axes.flatten()
    # c=["#e74c3c", '#27ae60', '#4168B7', '#8E44AD']
    for i, ax in enumerate(axes):
        plt.setp(ax, xlim=(start - 1, w + 1), ylim=(0 - (.01 * h), h + (.01 * h)))
        ax.hlines(y=h, xmin=start, xmax=w, color='k')
        ax.hlines(y=0, xmin=start, xmax=w, color='k')
        ax.vlines(x=tb, ymin=0, ymax=h, color='#2043B0', lw=1.5, linestyle='-', alpha=.5)
        ax.vlines(x=start + 2, ymin=0, ymax=h, color='k')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    sns.despine(top=True, right=True, bottom=True, left=True)
    return f, axes
