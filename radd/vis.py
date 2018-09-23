#!/usr/local/bin/env python
from __future__ import division
import sys
import warnings
from future.utils import listvalues
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from radd.tools import colors, analyze
from IPython.display import display, Latex
from scipy.stats.mstats import mquantiles
from itertools import product

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore", module="matplotlib")

cdict = colors.get_cpals('all')
rpal = cdict['rpal']
bpal = cdict['bpal']
gpal = cdict['gpal']
ppal = cdict['ppal']
heat = cdict['heat']
cool = cdict['cool']
slate = cdict['slate']


def plot_model_fits(y, yhat, ssd=None, ssderr=None, quantiles=np.arange(.1, 1.,.1), err=None, clrs=None, save=False, bw='scott', savestr=None, same_axis=True, lbls=None, cumulative=True, suppressLegend=False, hist=False, kde=True,  shade=True, norm_hist=True, data=None, simData=None, plot_error_rts=True, figure=None):
    """ main plotting function for displaying model fit predictions over data
    """
    sns.set(style='white', font_scale=1.3)
    if np.ndim(y)==1:
        y = y.reshape(1, -1)
    nlevels = y.shape[0]
    yhat = yhat.reshape(nlevels, -1)
    nquant = quantiles.size
    nssd = y[0, 1:-nquant*2].size
    ncols = 2
    figwidth = 10
    if plot_error_rts:
        ncols = 3
        figwidth = 13

    if figure is not None:
        f = figure
        axes = np.asarray(f.axes).reshape(1, ncols)
    elif same_axis:
        f, axes = plt.subplots(1, ncols, figsize=(figwidth, 4.3))
        axes = axes.reshape(1, ncols)
    else:
        f, axes = plt.subplots(nlevels, ncols, figsize=(figwidth, 4*nlevels))
        axes = axes.reshape(nlevels, ncols)

    if ssd is None and nssd>1:
        ssd = np.vstack([np.linspace(.2, .5, nssd)]*nlevels)
        ssd = ssd.reshape(nlevels, -1)

    if clrs is None:
        clrs = colors.assorted_list()
    if lbls is None:
        lbls = [None]*nlevels

    plot_acc = plot_stop_fit
    if nssd==1:
        plot_acc = plot_stop_fit_single
        if ssderr is None:
            ssderr = [np.array([0])]*nlevels

    saccErr, quantErr = [None]*2

    for i in range(nlevels):
        ax_ix = i
        if same_axis:
            ax_ix = 0
        if plot_error_rts:
            ax1, ax2, ax3 = axes[ax_ix]
            qAxes = np.array([ax2,ax3])
        else:
            ax1, ax2 = axes[ax_ix]
            qAxes = np.array([ax2])
        sacc, quant = unpack_vector(y[i], nlevels=nlevels, nquant=nquant, nssd=nssd)
        saccHat, quantHat = unpack_vector(yhat[i], nlevels=nlevels, nquant=nquant, nssd=nssd)

        if err is not None:
            err = err.reshape(nlevels, -1)
            saccErr, quantErr = unpack_vector(err[i], nlevels=nlevels, nquant=nquant, nssd=nssd)

        if nssd==1:
            plot_acc(y=sacc, yhat=saccHat, yerr=saccErr, x=ssd[i], xerr=ssderr[i], color=clrs[i], label=lbls[i], ax=ax1)
        else:
            plot_acc(sacc, saccHat, x=ssd[i], err=saccErr, color=clrs[i], label=lbls[i], ax=ax1, suppressLegend=suppressLegend)

        if not cumulative:
            plot_quantiles(data=data[i], clr=clrs[i], axes=qAxes, bw=bw, cumulative=cumulative, hist=True, kde=False, shade=True, norm_hist=norm_hist, same_axis=same_axis, alpha=.4)

            plot_quantiles(data=simData[i], clr=clrs[i], axes=qAxes, bw=bw, cumulative=cumulative, hist=False, kde=True, shade=False, norm_hist=norm_hist, same_axis=same_axis, alpha=1.)

        else:
            qpcHat, qpeHat = quantHat
            qpcHatX, qpcHatCurve = analyze.fit_logistic(qpcHat, quantiles)
            ax2.errorbar(quant[0], quantiles, xerr=quantErr[0], marker='o', color=clrs[i], ms=6.5, linewidth=0, elinewidth=2.)
            ax2.plot(qpcHatX, qpcHatCurve, linewidth=2, color=clrs[i])
            ax2.plot(qpcHat, quantiles, lw=0., marker='o', ms=10, mew=0, alpha=.1, mfc=clrs[i], mec=clrs[i], color=clrs[i])
            ax2.plot(qpcHat, quantiles, lw=0., marker='o', ms=10, mew=1.5, alpha=.8, mfc='none', mec=clrs[i])
            if plot_error_rts:
                qpeHatX, qpeHatCurve = analyze.fit_logistic(qpeHat, quantiles)
                ax3.errorbar(quant[1], quantiles, xerr=quantErr[1], marker='o', color=clrs[i], ms=6.5, linewidth=0, elinewidth=2.)
                ax3.plot(qpeHatX, qpeHatCurve, linewidth=2, color=clrs[i])
                ax3.plot(qpeHat, quantiles, lw=0., marker='o', ms=10, mew=0, alpha=.1, mfc=clrs[i], mec=clrs[i], color=clrs[i])
                ax3.plot(qpeHat, quantiles, lw=0., marker='o', ms=10, mew=1.5, alpha=.8, mfc='none', mec=clrs[i])

        format_rt_axes(qAxes, cdf=cumulative, yhat=yhat, quantiles=quantiles)
        if nssd==1:
            ax1.set_xlim(np.min(ssd)*.99, np.max(ssd)*1.01)
            xt = np.linspace(np.min(ssd)*.99, np.max(ssd)*1.01,4)
            xtl = [np.int(t*1000) for t in xt]
            ax1.set_xticks(xt)
            ax1.set_xticklabels(xtl)
    if save:
        plt.savefig('.'.join([savestr, 'png']), dpi=600)
    # plt.legend()



def plot_stop_fit(y, yhat, x=None, err=None, color=None, label=None, ax=None, alpha=1., m='o', suppressLegend=False):
    """ plotting function for displaying model-predicted
    stop curve (across SSDs) over empirical estimates
    """
    if ax is None:
        f, ax = plt.subplots(1, figsize=(4, 3))
    if x is None:
        x = np.linspace(.100, .500, len(y), dtype='float')
    if label is None:
        label = ['Data', 'Model']
    else:
        label = [label.capitalize(), None]
    if suppressLegend:
        label = [None, None]
    plot_stop_data(y, x=x, err=err, color=color, label=label[0], ax=ax, lw=0)
    plot_stop_curve_predicted(yhat, x=x, color=color, label=label[1], ax=ax, alpha=alpha)
    format_stop_axes(ax, x)
    # plt.legend()
    sns.despine()



def plot_stop_data(y, x=None, err=None, label=None, lw=2, alpha=1, color='k', ax=None, **kwargs):
    """
    PLOT empirical stop curve
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(4,3))
    if x is None:
        x = np.linspace(.250, .50, len(y))
    if err is not None:
        ax.errorbar(x, y, yerr=err, lw=lw, elinewidth=2, color=color, marker='o', ms=6.5, alpha=alpha, label=label)
    else:
        ax.plot(x, y, lw=lw, color=color, marker='o', ms=6.5, alpha=alpha, label=label)
    sns.despine()
    if label is not None:
        ax.legend()
    return ax



def plot_stop_curve_predicted(y, x=None, label=None, alpha=1., color='k', ax=None, get_pse=False, **kwargs):
    """
    PLOT model-estimated stop curve
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(4,3))
    if x is None:
        x = np.linspace(.250, .50, len(y))
    # xsim, ysim = analyze.fit_sigmoid(x, y)
    xsim, ysim = analyze.fit_logistic(x[::-1], y)
    ax.plot(x, y, lw=0., marker='o', ms=10, mew=1.5, alpha=.1, mfc=color, mec=color)
    ax.plot(x, y, lw=0., marker='o', ms=10, mew=1.5, alpha=.8, mfc='none', mec=color)
    ax.plot(xsim[::-1], ysim, lw=2., color=color, alpha=1, linestyle='-')
    ax.plot(x[0], y[0], lw=2., marker='o', ms=10, mew=1.5, alpha=.8, mfc='none', mec=color, label=label, color=color)
    if get_pse:
        PSEix = (np.abs(ysim - .5)).argmin()
        return xsim[PSE]
    sns.despine()
    return ax



def plot_quantiles(data, axes=None, clr=None, lbl=None, bw='scott', alpha=1., cumulative=True, hist=False, kde=True, shade=True, norm_hist=True, same_axis=False, ms=6.5, mcolor=None, quantiles=np.arange(.1, 1.,.1)):
    goData=data[data.response==1]
    cor = goData[goData['acc']==1]
    err = goData[goData['acc']==0]
    clrCor, clrErr = clr, clr
    if clr is None:
        clrCor, clrErr = "#27ae60",'#de143d'
    if axes is None:
        if same_axis:
            f, axCor = plt.subplots(1, figsize=(5, 4))
            axErr = axCor
        else:
            f, (axCor, axErr) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        axCor, axErr = axes
    if cumulative:
        plot_rt_cdf(cor, ax=axCor, clr=clrCor, lbl=lbl, alpha=alpha, ms=ms, mcolor=mcolor, quantiles=quantiles, bw=bw)
        plot_rt_cdf(err, ax=axErr, clr=clrErr, lbl=lbl, alpha=alpha, ms=ms, mcolor=mcolor, quantiles=quantiles, bw=bw)
    else:
        plot_rt_pdf(cor, ax=axCor, clr=clrCor, lbl=lbl, hist=hist, kde=kde, shade=shade, norm_hist=norm_hist, bw=bw, alpha=alpha)
        plot_rt_pdf(err, ax=axErr, clr=clrErr, hist=hist, kde=kde, shade=shade, norm_hist=norm_hist, bw=bw, alpha=alpha)
        format_rtPDF_axes([axCor, axErr], cor, err)



def plot_rt_cdf(data, ax=None, clr='blue', mcolor=None, lbl=None, alpha=1., ms=6.5, quantiles=np.arange(.1, 1.,.1), lw=2., mew=1., bw='scott'):
    data = data.copy()
    if 'idx' not in data.columns:
        data['idx'] = 1
    if ax is None:
        f, ax = plt.subplots(1, figsize=(4, 3))
    if mcolor is None:
        mcolor = clr
        lw = 0.
        mew = 0
    idxQuant = pd.DataFrame([mquantiles(idxdf.rt.values, quantiles) for idx, idxdf in data.groupby('idx')])
    quant, quantErr = idxQuant.mean(axis=0), idxQuant.sem(axis=0)*1.96
    ax.errorbar(quant, quantiles, xerr=quantErr, color=clr, linewidth=0, elinewidth=1.5, marker='o', ms=ms, mfc=clr, mew=0., label=lbl, mec=clr, alpha=alpha)
    ax.errorbar(quant, quantiles, xerr=quantErr, color=clr, linewidth=0, elinewidth=1.5, marker='o', ms=ms, mfc='none', mew=mew, label=lbl, mec=clr, alpha=.7)
    sns.kdeplot(data.rt.values, cumulative=True, color=clr, linewidth=lw, ax=ax, alpha=1., bw=bw)
    ax.set_xlim(quant.min()*.93, quant.max()*1.05)
    ax.set_ylim(-0.02, 1.02)



def plot_rt_pdf(data, ax=None, clr='blue', lbl=None, alpha=.5, hist=False, kde=True, shade=True, norm_hist=True, bw=.01, bins=18):
    if ax is None:
        f, ax = plt.subplots(1, figsize=(4, 3))
    sns.distplot(data.rt.values, color=clr, label=lbl, kde=True, hist=False, kde_kws={'shade':shade, 'alpha':alpha, 'bw':bw}, norm_hist=norm_hist, hist_kws={'alpha':alpha}, ax=ax, bins=bins)


def plot_stop_fit_single(y, yhat, yerr=None, x=0, xerr=None, ax=None, linestyles=None, color=None, label=None, **kwargs):

    if label is None:
        label = ['Data', 'Model']
    else:
        label = [label, None]
    y = y[1]; yerr=yerr[1]; yhat=yhat[1]
    yline = np.linspace(y-yerr, y+yerr, 4)
    xline = np.ones(yline.size)*x

    ax.plot(x, y, marker='o', ms=6.5, color=color, label=label[0], linewidth=0)
    ax.fill_betweenx(yline, xline-xerr, xline+xerr, alpha=.25, color=color)
    ax.plot(x, yhat, marker='o', ms=10, mew=1.5, alpha=.8, mfc='none', mec=color, label=label[1], lw=0)
    ax.plot(x, yhat, marker='o', ms=10, mew=0, alpha=.1, mfc=color, mec=color)

    # ax.legend(loc=0)
    plt.tight_layout()
    ax.set_ylim(0, 1)



def unpack_vector(vector, nlevels=1, nquant=9, nssd=5):
    if nssd>1:
        # get accuracy at each SSD
        sacc = vector[1:1+nssd]
    else:
        # get go, stop accuracy
        sacc = vector[:2]
    quant = vector[-nquant*2:]
    qc = quant[:nquant]
    qe = quant[-nquant:]
    return sacc, np.vstack([qc, qe])



def format_stop_axes(ax, xticks, xlim=None):
    if xlim is None:
        xlim=(xticks[0]*.935, xticks[-1]*1.05)
    xtls = [np.int(np.round(xx,3)*1000) for xx in xticks]
    yticks = np.array([0., .2, .4, .6, .8,  1.])
    plt.setp(ax, xticks=xticks, xlim=xlim, yticks=yticks, ylim=(-.01, 1.05))
    ax.set_ylabel('Stop Accuracy')
    ax.set_xticklabels(xtls)
    ax.set_yticklabels(yticks)
    # ax.legend(loc=0)
    if ax.is_last_row():
        ax.set_xlabel('SSD (ms)')
    plt.tight_layout()
    sns.despine()
    return ax



def format_rtPDF_axes(axes, cor, err):
    df = pd.concat([cor, err])
    df = df.reset_index(drop=True)
    axCor, axErr = axes
    axCor.set_ylabel('P(RT)')
    axErr.set_ylabel('')
    axCor.set_yticklabels([])
    axErr.set_yticklabels([])
    x = [df.rt.min()*.95, df.rt.median(), df.rt.max()*1.05]
    x = [np.asscalar(np.round(xi,2)) for xi in x]
    axCor.set_xticks(x)
    axErr.set_xticks(x)
    axCor.set_xticklabels([str(xi*1000)[:3] for xi in x])
    axErr.set_xticklabels([str(xi*1000)[:3] for xi in x])
    axCor.set_xlim(x[0], x[-1])
    axErr.set_xlim(x[0], x[-1])
    axCor.set_xlabel('Correct RT (ms)')
    axErr.set_xlabel('Error RT (ms)')



def format_rt_axes(axes, cdf=True, yhat=None, quantiles=np.arange(.1, 1.,.1)):
    if yhat is not None:
        rtqList = [unpack_vector(yh, nquant=quantiles.size)[1:] for yh in yhat]
        corquant = np.hstack(np.vstack(rtqList)[:, 0, :])
        errquant = np.hstack(np.vstack(rtqList)[:, 1, :])
        xxticks = [np.linspace(np.nanmin(rtq)*.93, np.nanmax(rtq)*1.06, 5) for rtq in [corquant, errquant]]
        xxticks = [np.array([np.round(xt, 2) for xt in xxt]) for xxt in xxticks]
        xxtls = [np.array([int(xtl*1000) for xtl in xxt]) for xxt in xxticks]
        xxlim = [(xxt[0], xxt[-1]) for xxt in xxticks]
    q_axes = []
    for ax in axes.flatten():
        if not ax.is_first_col():
            q_axes.append(ax)
    if cdf:
        yylim=(-.02, 1.05); yyticks=np.arange(0, 1.2, .2)
        ylabel = 'Cumulative P(Go)'
    else:
        ydata = np.hstack([np.hstack([l.get_ydata() for l in ax.lines]) for ax in axes])
        yyticks = np.array([0, np.nanmax(ydata)*1.05])
        yyticks = np.array([int(np.round(yt)) for yt in yyticks])
        yylim=(-.02, yyticks[-1]);
        ylabel = 'Probability Mass'
    xlabels = ['Correct RT (ms)', 'Error RT (ms)']
    for i, ax in enumerate(axes):
        plt.setp(ax, yticks=yyticks, ylim=yylim, xlabel=xlabels[i], xlim=xxlim[i], xticks=xxticks[i], xticklabels=xxtls[i])
        if ax.is_last_row():
            plt.setp(ax, xlabel=xlabels[i])
        else:
            plt.setp(ax, xlabel='')
    if not cdf:
        axes[0].set_yticklabels(yyticks)
        if len(axes)>1:
            axes[1].set_yticklabels([], fontsize=.05)
    else:
        axes[0].set_yticklabels([], fontsize=.05)
        if len(axes)>1:
            axes[1].set_yticklabels([], fontsize=.05)
    axes[0].set_ylabel(ylabel, fontsize=17)
    plt.tight_layout()
    sns.despine()
    return axes



def compare_nested_models(fitdf, model_ids, yerr=None, plot_stats=True, verbose=False):
    gof = {}
    fitdf = fitdf[fitdf.pvary.isin(model_ids)]
    # print GOF stats for both models
    for i, p in enumerate(model_ids):
        name = parameter_name(p)
        gof[p] = fitdf.loc[i, ['AIC', 'BIC']]
    # Which model provides a better fit to the data?
    aicwinner = model_ids[np.argmin([gof[mid][0] for mid in model_ids])]
    bicwinner = model_ids[np.argmin([gof[mid][1] for mid in model_ids])]
    if verbose:
        print('AIC likes {} model'.format(aicwinner))
        print('BIC likes {} model'.format(bicwinner))
    plot_model_gof(gof, aicwinner, model_ids, yerr=yerr)
    return gof



def plot_model_gof(gof_dict, aicwinner, pvary=None, yerr=None):
    if pvary is None:
        pvary = np.sort(list(gof_dict))
    nmodels = len(pvary)
    if yerr is None:
        yerr = [np.zeros(2) for i in range(nmodels)]
    f, ax = plt.subplots(1, figsize=(10,7))
    # ax.invert_yaxis()
    x = np.arange(1, nmodels*2, 2)
    clrs = [colors.param_color_map(p) for p in pvary]
    lbls = {p: parameter_name(p,True) for p in pvary}
    for i, p in enumerate(pvary):
        yaic, ybic = gof_dict[p][['AIC', 'BIC']]
        aic_err, bic_err = yerr[i]
        lbl = lbls[p]
        if p==aicwinner:
            lbl+='*'
        ax.bar(x[i]-.32, yaic, yerr=aic_err, color=clrs[i], ecolor="#34495e", error_kw={'linewidth':3, 'alpha':1}, alpha=.8, width=.64, align='center', edgecolor=clrs[i], label=lbl)
        ax.bar(x[i]+.32, ybic, yerr=bic_err, color=clrs[i], ecolor="#34495e", error_kw={'linewidth':3, 'alpha':1}, alpha=.65, width=.64, align='center', edgecolor=clrs[i])
    vals = np.hstack(gof_dict.values()).astype(float)
    yylim = (vals.max()*.97, vals.min()*1.07)
    plt.setp(ax, xticks=x, ylim=yylim, xlim=(0, x[-1]+1), ylabel='IC')
    ax.set_xticklabels(['AIC|BIC']*nmodels, fontsize=14)
    sns.despine(bottom=True, top=False)
    ax.legend(loc=0, fontsize=14)
    ax.invert_yaxis()



def plot_param_distributions(p=['a', 'sso', 'ssv', 'tr', 'v', 'xb', 'z'], n=2000, method='random'):
    from radd import theta
    pkeys = np.sort(list(p))
    nparams = pkeys.size
    p_dists = theta.random_inits(pkeys=pkeys, ninits=n, method=method)
    clrs = colors.param_color_map('all')
    lbls = {pk: parameter_name(pk,True) for pk in pkeys}
    ncols = np.ceil(nparams/2.).astype(int)
    fig, axes = plt.subplots(2, ncols, figsize=(10, 4.5))
    axes = axes.flatten()
    for i, pk in enumerate(pkeys):
        sns.distplot(p_dists[pk], kde=False, ax=axes[i], norm_hist=True, color=clrs[pk], label=lbls[pk], hist_kws={'alpha':.8})
        sns.kdeplot(p_dists[pk], ax=axes[i], color=clrs[pk], shade=True, linewidth=0, alpha=.5, bw='scott')
    for ax in axes:
        ax.legend(loc=1, fontsize=15)
    plt.tight_layout()
    sns.despine()



def compare_param_estimates(p1, p2, depends_on=None):
    popt1 = deepcopy(p1)
    popt2 = deepcopy(p2)
    if depends_on is not None:
        _ = [popt1.pop(param) for param in list(depends_on)]
    pnames = np.sort(list(popt1))
    x = np.arange(pnames.size)
    clrs = sns.color_palette(palette='muted', n_colors=x.size)
    i = 0
    f, ax = plt.subplots(1, figsize=(6,5))
    for param in pnames:
        plt.plot(x[i], abs(popt1[param]), marker='o', markersize=10, color=clrs[i])
        plt.plot(x[i], abs(popt2[param]), marker='s', markersize=8, color=clrs[i])
        i+=1
    _ = plt.setp(plt.gca(), xlim=(x[0]-.5, x[-1]+.5), xticks=x, xticklabels=pnames)



def get_plot_labels(clmap=None):
    nconds = len(list(clmap))
    clevels = [list(levels) for levels in listvalues(clmap)]
    if nconds>1:
        level_data = list(product(*clevels))
        clevels = [' '.join([str(lvl).capitalize() for lvl in lvls]) for lvls in level_data]
    else:
        clevels = np.hstack(clevels).tolist()
        clevels = [lvl.capitalize() for lvl in clevels]
    return clevels


def parameter_name(param, tex=False):
    ix = 0
    if tex:
        ix = 1
    param_name = {'v':['Drift', '$v_{E}$'],
        'ssv': ['Brake Drift', '$v_{B}$'],
        'a': ['Threshold', '$a$'],
        'tr': ['Onset', '$tr$'],
        'xb': ['Dynamic Gain', '$\gamma$'],
        'sso': ['Brake Onset', '$so_{B}$'],
        'z': ['Execution Baseline', '$z_{E}$'],
        #'v_ssv': ['Drift Ratio', '$v_{E},v_{B}$'],
        'aG': ['Alpha+', '$\\alpha^+$'],
        'aErr': ['Alpha-', '$\\alpha^-$'],
        'A': ['Alpha', '$\\beta$'],
        'B': ['Beta', '$\\alpha$'],
        'R': ['Rho', '$\\rho$'],
        'flat': ['Flat', 'Flat'],
        'all': ['Flat', 'Flat']}
    if '_' in param:# and param!='v_ssv':
        param = param.split('_')
    if isinstance(param, list):
        return ','.join([param_name[p][ix] for p in param])
    return param_name[param][ix]
