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
sns.set(style='white', context='notebook', rc={'text.color': 'black', 'axes.labelcolor': 'black', 'figure.facecolor': 'white'}, font_scale=1.5)

cdict = colors.get_cpals('all')
rpal = cdict['rpal']
bpal = cdict['bpal']
gpal = cdict['gpal']
ppal = cdict['ppal']
heat = cdict['heat']
cool = cdict['cool']
slate = cdict['slate']

def unpack_vector(vector, nlevels=1, nquant=5, nssd=1, kde_quant=False):
    unpacked = []
    vector = vector.reshape(nlevels, int(vector.size/nlevels))
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
        if kde_quant:
            quant_cor = analyze.kde_fit_quantiles([quant_cor], bw=.005)
            quant_err = analyze.kde_fit_quantiles([quant_err], bw=.005)
        unpacked.append([presp, quant_cor, quant_err])
    return unpacked

def plot_model_fits(y, yhat, fitparams, kde_quant=True, palette=bpal, save=False, cdf=True):
    nlevels = y.ndim
    fitparams['nlevels'] = nlevels
    quantiles = fitparams['quantiles']
    ssd, nssd, nss, nss_per, ssd_ix = fitparams['ssd_info']
    f, axes = plt.subplots(nlevels, 3, figsize=(12, 3.5*nlevels), sharey=True)
    axes = np.asarray(axes)
    clrs_lvl = palette(nlevels)
    y_data = unpack_vector(y, nlevels=nlevels, nquant=quantiles.size, nssd=nssd, kde_quant=False)
    yhat_data = unpack_vector(yhat, nlevels=nlevels, nquant=quantiles.size, nssd=nssd, kde_quant=True)
    if nssd>1:
        ssds = ssd.mean(axis=0).squeeze()*1e3
        scx = np.array([np.float(d) for d in ssds])
        plot_func = scurves
    else:
        scx = np.array([0, 1])
        plot_func = plot_accuracy
    labels = get_plot_labels(fitparams)

    for i in range(nlevels):
        if nlevels>1:
            ax1, ax2, ax3 = axes[i]
        else:
            ax1, ax2, ax3 = axes
        stop_acc = [y_data[i][0], yhat_data[i][0]]
        qc, qc_hat = [y_data[i][1], yhat_data[i][1]]
        qe, qe_hat = [y_data[i][2], yhat_data[i][2]]
        c = clrs_lvl[i]
        lemp, lhat = labels[i]
        plot_func(stop_acc, x=scx, colors=[c, c], markers=False, labels=labels[i], linestyles=['-', '--'], ax=ax1)
        sns.kdeplot(qc_hat, cumulative=cdf, color=c, ax=ax2, linewidth=2, linestyle='--', bw=.01)
        sns.kdeplot(qe_hat, cumulative=cdf, color=c, ax=ax3, linewidth=2, linestyle='--', bw=.01)
        ax2.plot(qc, quantiles, color=c, linewidth=0, marker='o')
        ax3.plot(qe, quantiles, color=c, linewidth=0, marker='o')

        if i==0:
            ax1.set_title('Stop Accuracy', position=(.5, 1.02))
            ax2.set_title('Correct RT Quantiles', position=(.5, 1.02))
            ax3.set_title('Error RT Quantiles', position=(.5, 1.02))
            if nlevels>1:
                for ax in axes[0, :]:
                    ax.set_xticklabels([])
    for ax in axes[:, -2:].flatten():
        qdata = np.hstack([arr[1:] for arr in yhat_data]).flatten()
        xxlim = (qdata.min()-.03, qdata.max()+.03)
        ax.set_xlim(xxlim[0], xxlim[1])
    for ax in axes.flatten()[-2:]:
        ax.set_xlabel('RT (s)')
        xxticks = [xxlim[0], np.mean(xxlim), xxlim[-1]]
        xxtls = ["{:.2f}".format(xtl) for xtl in xxticks]
        plt.setp(ax, xlim=xxlim, xticks=xxticks, xticklabels=xxtls)
    axes[-1, 0].set_xlabel('SSD (ms)')
    if save:
        savestr = '_'.join([fitparams['kind'], str(fitparams['idx'])+'.png'])
        plt.savefig(savestr, dpi=600)
        plt.close('all')

def scurves(lines=[],  yerr=[], pretune_fx='interpolate', ax=None, linestyles=None, colors=None, labels=None, mc=None, x=None, get_pse=False, **kwargs):

    if ax is None:
        f, ax = plt.subplots(1, figsize=(5.5, 6))
    if colors is None:
        colors = slate(len(lines))
    if labels is None:
        labels = [''] * len(lines)
    if linestyles is None:
        linestyles = ['-'] * len(lines)
    lines = [(line) if type(line) == list else line for line in lines]
    if x is None:
        x = np.arange(len(lines[0]), dtype='float')*50
    pse=[]
    for i, yi in enumerate(lines):
        if pretune_fx=='interpolate':
            xsim, ysim = analyze.scurve_interpolate(x, yi)
        elif pretune_fx=='polynomial':
            xsim, ysim = analyze.scurve_poly_fit(x, yi)
        xsim, ysim = analyze.fit_sigmoid(xsim, ysim)
        idx = (np.abs(ysim - .5)).argmin()
        pse.append(xsim[idx])
        # Plot the results
        if yerr != []:
            ax.errorbar(x, yi, yerr=yerr[i], color=colors[i], lw=0, marker='o', elinewidth=2, ecolor=colors[i])
            ax.plot(xsim, ysim, linestyle=linestyles[i], lw=2., color=colors[i], label=labels[i])
        else:
            if not i%2:
                ax.plot(x, yi, lw=0., color=colors[i], label=labels[i], marker='o')
            else:
                ax.plot(xsim, ysim, linestyle=linestyles[i], lw=2.1, color=colors[i], label=labels[i])
        pse.append(xsim[idx])
    xxlim = (xsim[0], xsim[-1])
    plt.setp(ax, xticks=x, xlim=xxlim, ylim=(-.01, 1.05), yticks=np.arange(0, 1.2, .2), ylabel='P(Stop)')
    ax.legend(loc=0)
    ax.set_xlim(xxlim[0], xxlim[1])
    plt.tight_layout()
    sns.despine()
    if get_pse:
        return pse

def plot_accuracy(lines=[], yerr=None, x=np.array([0, 1]), ax=None, linestyles=None, colors=None, labels=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, figsize=(5.5, 6))
    if colors is None:
        colors = slate(len(lines))
    if labels is None:
        labels = [''] * len(lines)
    if linestyles is None:
        linestyles = ['-'] * len(lines)
    lines = [(line) if type(line) == list else line for line in lines]
    if x is None:
        x = np.arange(len(lines[0]), dtype='float')
    for i, yi in enumerate(lines):
        xjitter = x + (i*.02)
        color = colors[i]
        if yerr is not None:
            ax.errorbar(xjitter, yi, yerr=yerr[i], color=color, lw=2, elinewidth=2, ecolor=color)
        else:
            ax.plot(xjitter, yi, linestyle=linestyles[i], lw=2, color=color, label=labels[i])

    plt.setp(ax, xticks=x, xlim=(-0.25, 1.25), ylim=(.25, 1.05), yticks=(.3, 1))
    ax.set_ylabel('Percent Correct'); ax.legend(loc=0)

    ax.set_xticklabels(['Go', 'Stop'])
    plt.tight_layout()
    sns.despine()

def get_plot_labels(fitparams):
    lbls = np.hstack(fitparams['clmap'].values())
    labels = [[lbl + dtype for dtype in ['_data', '_model']] for lbl in lbls]
    return labels

def plot_kde_cdf(quant, bw=.1, ax=None, color=None):
    if ax is None:
        f, ax = plt.subplots(1, figsize=(5, 5))
    if color is None:
        color = 'k'
    kdefits = analyze.kde_fit_quantiles(quant, bw=bw)
    sns.kdeplot(kdefits, cumulative=True,  color=color, ax=ax, linewidth=2.5)
    ax.set_xlim(kdefits.min() * .94, kdefits.max() * 1.05)
    ax.set_ylabel('P(RT<t)')
    ax.set_xlabel('RT (s)')
    ax.set_ylim(-.05, 1.05)
    ax.set_xticklabels(ax.get_xticks() * .1)
    plt.tight_layout()
    sns.despine()
