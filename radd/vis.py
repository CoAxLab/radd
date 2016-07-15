#!/usr/local/bin/env python
from __future__ import division
import sys
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from radd.tools import colors, messages, analyze
from IPython.display import HTML, Javascript, display
import warnings
warnings.simplefilter('ignore', np.RankWarning)

cdict = colors.get_cpals('all')
rpal = cdict['rpal']
bpal = cdict['bpal']
gpal = cdict['gpal']
ppal = cdict['ppal']
heat = cdict['heat']
cool = cdict['cool']
slate = cdict['slate']

def plot_model_fits(y, yhat, fitparams, err=None, palettes=[gpal, bpal], save=False, cdf=True, bw=.01, sameaxis=False):
    """ main plotting function for displaying model fit predictions over data
    """
    sns.set(style='darkgrid', rc={'figure.facecolor':'white'}, font_scale=1.5)
    nlevels = fitparams['nlevels']
    ssd, nssd, nss, nss_per, ssd_ix = fitparams.ssd_info
    if len(palettes) != nlevels:
        palettes = colors.get_cpals(aslist=True)[:nlevels]
    if sameaxis or nlevels==1:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
        axes = np.asarray([ax1, ax2, ax3])
    else:
        f, axes = plt.subplots(nlevels, 3, figsize=(14, 4.5*nlevels), sharey=True)
    if nssd==1:
        plot_acc = plot_accuracy
    else:
        plot_acc = scurves
    clrs = [pal(2) for pal in palettes]
    lbls = get_plot_labels(fitparams)
    y_dat = unpack_vector(y, fitparams, bw=bw)
    y_kde = unpack_vector(y, fitparams, bw=bw, kde=True)
    yhat_dat = unpack_vector(yhat, fitparams, bw=bw)
    if err is not None:
        y_err = unpack_vector(err, fitparams, bw=bw)
        sc_err = [e[0] for e in y_err]
        qp_err = [[e[1], e[2]] for e in y_err]
    else:
        sc_err, qp_err = [[None]*nlevels]*2
    for i in range(nlevels):
        if not sameaxis:
            ax1, ax2, ax3 = axes[i]
        accdata = [y_dat[i][0], yhat_dat[i][0]]
        qpdata = [y_dat[i], yhat_dat[i]]
        plot_acc(accdata, err=sc_err[i], ssd=ssd[i], colors=clrs[i], labels=lbls[i], ax=ax1, ssdlabels=sameaxis)
        plot_quantiles(qpdata, err=qp_err[i], colors=clrs[i], axes=[ax2,ax3], kde=y_kde[i], bw=bw)
    axes = format_axes(axes)
    if save:
        plt.savefig(fitparams['model_id']+'.png', dpi=600)
    if fitparams['fit_on']=='subjects' and save:
        plt.close('all')

def scurves(data, err=None, colors=None, labels=None, ssd=None, ax=None, get_pse=False, pretune_fx='interpolate', **kwargs):
    """ plotting function for displaying model-predicted
    stop curve (across SSDs) over empirical estimates
    """
    if err is None:
        err = [np.zeros(len(dat)) for dat in data]
    if ax is None:
        f, ax = plt.subplots(1, figsize=(5.5, 6))
    if colors is None:
        colors = slate(len(data))
    if labels is None:
        labels = [''] * len(data)
    if ssd is not None:
        ssds = ssd.squeeze()*1e3
        x = np.array([np.float(d) for d in ssds])
        xtls = ["{:.0f}".format(d) for d in ssds]
    else:
        x = np.arange(len(data[0]), dtype='float')*50
        xtls = [str(xx) for xx in x]
    pse=[]
    for i, yi in enumerate(data):
        if pretune_fx=='interpolate':
            xsim, ysim = analyze.scurve_interpolate(x, yi)
        elif pretune_fx=='polynomial':
            xsim, ysim = analyze.scurve_poly_fit(x, yi)
        xsim, ysim = analyze.fit_sigmoid(xsim, ysim)
        idx = (np.abs(ysim - .5)).argmin()
        pse.append(xsim[idx])
        if not i%2:
            ax.errorbar(x, yi, yerr=err[i], lw=0., elinewidth=1.5, color=colors[i],  marker='o', ms=5)
            ax.plot(xsim, ysim, lw=2, color=colors[i], label=labels[i])
            continue
        ax.plot(x, yi, lw=0., color=colors[i], marker='o', ms=10, mfc="none", mec=colors[i], mew=1.7, label=labels[i])
    xxlim = (xsim[0], xsim[-1])
    ytls = np.arange(0, 1.2, .2).astype(np.int)
    plt.setp(ax, xticks=x, xlim=xxlim, xticklabels=xtls, ylim=(-.01, 1.05), yticks=ytls, yticklabels=ytls)
    ax.set_ylabel('Percent Correct'); ax.legend(loc=0)
    ax.legend(loc=0)
    plt.tight_layout()
    sns.despine()
    if get_pse:
        return pse

def plot_quantiles(data, err=None, axes=None, colors=None, labels=None, kde=None, quantiles=np.array([.1, .3, .5, .7, .9]), bw=.008):
    """ plotting function for displaying model-predicted
    quantile-probabilities over empirical estimates
    """
    y_data, yhat_data = data
    c, c_hat = colors
    if axes is not None:
        axc, axe = axes
    else:
        f, (axc, axe) = plt.subplots(1, 2, figsize=(10, 4))
    qc, qc_hat = y_data[1], yhat_data[1]
    qe, qe_hat = y_data[2], yhat_data[2]
    if err is not None:
        qc_err, qe_err = err
    else:
        qc_err, qe_err = [np.zeros(len(qc))]*2
    if kde is not None:
        qc_kde, qe_kde = kde[1], kde[2]
        sns.kdeplot(qc_kde, cumulative=1, color=c, ax=axc, linewidth=2, linestyle='-', bw=bw)
        sns.kdeplot(qe_kde, cumulative=1, color=c, ax=axe, linewidth=2, linestyle='-', bw=bw)
    axc.errorbar(qc, quantiles, xerr=qc_err, color=c, linewidth=0, elinewidth=1.5, marker='o', ms=5, label=labels)
    axe.errorbar(qe, quantiles, xerr=qe_err, color=c, linewidth=0, elinewidth=1.5, marker='o', ms=5, label=labels)
    axc.plot(qc_hat, quantiles, mec=c_hat, linewidth=0, marker='o', ms=10, mfc='none', mew=1.7, label=labels)
    axe.plot(qe_hat, quantiles, mec=c_hat, linewidth=0, marker='o', ms=10, mfc='none', mew=1.7, label=labels)

def plot_accuracy(data=[], err=None, ssd=None, ax=None, linestyles=None, colors=None, labels=None, xtls=['Go', 'Stop'], ssdlabels=False, **kwargs):
    """ plotting function for displaying model-predicted
    stop probability (at mean SSD) on top of empirical estimates
    (used when data is collected using tracking procedure to est. SSRT)
    """
    if err is None:
        err = [np.zeros(len(dat)) for dat in data]
    if ax is None:
        f, ax = plt.subplots(1, figsize=(5.5, 6))
    if colors is None:
        colors = slate(len(data))
    if labels is None:
        labels = [''] * len(data)
    if linestyles is None:
        linestyles = ['-', '--'] * len(data)
    x = np.arange(len(data[0]), dtype='float')
    if ssd is not None:
        ssdstr = "SSD={:.2f}".format(ssd.mean(axis=0)*1e3)
        if ssdlabels:
            labels[0] = ' '.join([labels[0], ssdstr])
            xtls = ['Go', 'Stop']
        else:
            xtls = ['Go', ssdstr]
    for i, yi in enumerate(data):
        if not i%2:
            ax.errorbar(x, yi, yerr=err[i], linestyle=linestyles[i], lw=1.5, elinewidth=1.5, color=colors[i], label=labels[i], marker='o', ms=5)
            continue
        xjit = x + (i*.02)
        ax.plot(xjit, yi, lw=0, marker='o', ms=10, mfc='none', mec=colors[i], mew=1.7, label=labels[i])

    plt.setp(ax, xticks=x, xlim=(-0.25, 1.25), xticklabels=xtls, ylim=(-.01, 1.05), yticks=np.arange(0, 1.2, .2))
    ax.set_ylabel('Percent Correct'); ax.legend(loc=0)
    sns.despine()
    plt.tight_layout()

def unpack_vector(vector, fitparams, kde=False, bw=.01):
    nlevels = fitparams['nlevels']
    nquant = fitparams['quantiles'].size
    nssd = fitparams['ssd_info'][1]
    vector = vector.reshape(nlevels, int(vector.size/nlevels))
    unpacked = []
    for v in vector:
        if nssd>1:
            # get accuracy at each SSD
            presp = v[1:1+nssd]
        else:
            # get go, stop accuracy
            presp = v[:2]
        quant = v[-nquant*2:]
        quant_cor = quant[:nquant]
        quant_err = quant[-nquant:]
        if kde:
            quant_cor = analyze.kde_fit_quantiles([quant_cor], bw=bw)
            quant_err = analyze.kde_fit_quantiles([quant_err], bw=bw)
        unpacked.append([presp, quant_cor, quant_err])
    return unpacked

def format_axes(axes):
    q_axes = []
    for ax in axes.flatten():
        if not ax.is_first_col():
            q_axes.append(ax)
    yylim=(-.01, 1.05); yyticks=np.arange(0, 1.2, .2)
    axesqp = [np.hstack([l.get_xdata() for l in ax.lines]) for ax in q_axes]
    xxticks = [[qdata.min(), qdata.mean(), qdata.max()] for qdata in axesqp]
    xxlim = [(xxt[0]-.008, xxt[-1]+.008) for xxt in xxticks]
    xxtls = [["{:.2f}".format(xtl) for xtl in xxt] for xxt in xxticks]
    for ax in axes.flatten():
        plt.setp(ax, ylim=yylim, yticks=yyticks)
        if not ax.is_first_col():
            plt.setp(ax, yticklabels=yyticks)
        if not ax.is_first_col():
            xl, xt, xtl = xxlim[0], xxticks[0], xxtls[0]
            if ax.is_last_col():
                xl, xt, xtl = xxlim[-1], xxticks[-1], xxtls[-1]
            plt.setp(ax, xlim=xl, xticks=xt, xticklabels=xtl)
            if ax.is_last_row():
                plt.setp(ax, xlabel='RT (s)')
            else:
                plt.setp(ax, xlabel='')
        if ax.is_first_col() and ax.is_last_row():
            ax.set_xlabel('SSD (ms)')
    axes = axes.flatten()
    axes[0].set_title('Accuracy', position=(.5, 1.02))
    axes[1].set_title('Correct Quant-Prob', position=(.5, 1.02))
    axes[2].set_title('Error Quant-Prob', position=(.5, 1.02))
    sns.despine()
    plt.tight_layout()
    return axes

def get_plot_labels(fitparams):
    lbls = np.hstack(fitparams['clmap'].values())
    labels = [[lbl + dtype for dtype in [' data', ' model']] for lbl in lbls]
    return labels
