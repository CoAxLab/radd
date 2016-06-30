#!/usr/local/bin/env python
from __future__ import division
import os
import sys
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy import array
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from radd.tools import colors, messages, analyze, theta
from scipy.stats.mstats import mquantiles as mq
from numpy import cumsum as cs
from numpy import append as app
from radd.tools.analyze import kde_fit_quantiles
from IPython.display import HTML, Javascript, display


sns.set(style='white', context='notebook', rc={'text.color': 'black', 'axes.labelcolor': 'black', 'figure.facecolor': 'white'}, font_scale=1.5)

cdict = colors.get_cpals('all')
rpal = cdict['rpal']
bpal = cdict['bpal']
gpal = cdict['gpal']
ppal = cdict['ppal']
heat = cdict['heat']
cool = cdict['cool']
slate = cdict['slate']

def unpack_vector(vector, fitparams, kde_quant=False):
    ssd, nssd, nss, nss_per, ssd_idx = fitparams['ssd_info']
    nquant = fitparams['quantiles'].size
    nlevels = fitparams['nlevels']
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
            quant_cor = kde_fit_quantiles([quant_cor], bw=.01)
            quant_err = kde_fit_quantiles([quant_err], bw=.01)
        unpacked.append([presp, quant_cor, quant_err])
    return unpacked

def plot_model_fits(y, yhat, fitparams, kde_quant=True, palette=bpal, save=False, cdf=True):
    nlevels = y.ndim
    fitparams['nlevels'] = nlevels
    f, axes = plt.subplots(nlevels, 3, figsize=(12, 4*nlevels))
    axes = np.asarray(axes)
    clrs_lvl = palette(nlevels)
    y_data = unpack_vector(y, fitparams, kde_quant=kde_quant)
    yhat_data = unpack_vector(yhat, fitparams, kde_quant=kde_quant)
    ssd, nssd, nss, nss_per, ssd_ix = fitparams['ssd_info']
    if nssd>1:
        ssds = ssd.mean(axis=0).squeeze()*1e3
        scx = np.array([np.float(d) for d in ssds])[::-1]
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
        if kde_quant:
            sns.kdeplot(qc, cumulative=cdf, color=c, ax=ax2, linewidth=2, bw=.01)
            sns.kdeplot(qc_hat, cumulative=cdf, color=c, ax=ax2, linewidth=2, linestyle='--', bw=.01)
            sns.kdeplot(qe, cumulative=cdf, color=c, ax=ax3, linewidth=2, bw=.01)
            sns.kdeplot(qe_hat, cumulative=cdf, color=c, ax=ax3, linewidth=2, linestyle='--', bw=.01)
        else:
            ax2.plot(qc, color=c, linewidth=2)
            ax2.plot(qc_hat, color=c, linewidth=2, linestyle='--',)
            ax3.plot(qe, color=c, linewidth=2)
            ax3.plot(qe_hat, color=c, linewidth=2, linestyle='--',)
        if i==0:
            ax1.set_title('Stop Accuracy', position=(.5, 1.02))
            ax2.set_title('Correct RT Quantiles', position=(.5, 1.02))
            ax3.set_title('Error RT Quantiles', position=(.5, 1.02))
            if nlevels>1:
                for ax in axes[0, :]:
                    ax.set_xticklabels([])
    for ax in axes.flatten()[-3:]:
        ax.set_xlabel('RT (s)')
    if save:
        savestr = '_'.join([fitparams['kind'], str(fitparams['idx'])+'.png'])
        plt.savefig(savestr, dpi=600)

def scurves(lines=[],  yerr=[], pstop=.5, ax=None, linestyles=None, colors=None, labels=None, mc=None, x=None, pse=[], **kwargs):
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
        x = np.arange(len(lines[0]), dtype='float')[::-1]*50
    xtls = x.copy()
    xsim = np.linspace(x.min()-20, x.max()+20, 10000)
    x = analyze.res(-x, lower=x[0], upper=x[-1])
    xxlim = (x[0]-20, x[-1]+20)

    for i, yi in enumerate(lines):
        color = colors[i]
        y = analyze.res(yi, lower=yi[-1], upper=yi[0])
        p_guess = (np.mean(x), np.mean(y), .48, .5)
        p, cov, infodict, mesg, ier = optimize.leastsq(analyze.residuals, p_guess, args=(x, y), full_output=1, maxfev=10000, ftol=1.e-20)
        x0, y0, c, k = p
        pxp = analyze.sigmoid(p, xsim)
        idx = (np.abs(pxp - pstop)).argmin()
        pse.append(xsim[idx])
        # Plot the results
        if yerr != []:
            ax.errorbar(x, yi, yerr=yerr[i], color=colors[i], lw=0, marker='o', elinewidth=2, ecolor=colors[i])
            ax.plot(xsim, pxp, linestyle=linestyles[i], lw=2., color=colors[i], label=labels[i])
        else:
            ax.plot(xsim, pxp, linestyle=linestyles[i], lw=2.1, color=colors[i], label=labels[i])
        pse.append(xsim[idx])
    plt.setp(ax, xticks=x, ylim=(-.01, 1.05), yticks=np.arange(0,1.2,.2))#[0,  1])
    ax.set_xlabel('SSD (ms)')
    ax.set_ylabel('P(Stop)')
    ax.legend(loc=0)
    plt.tight_layout()
    sns.despine()
    return (pse)

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

def render_animation(anim):
    from IPython.display import HTML
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))

def animate_dpm(model):
    """ to render animation within a notebook :
        vis.render_animation(vis.animated_dpm_example(MODEL))
    """
    from matplotlib import animation
    from copy import deepcopy
    params = deepcopy(model.inits)
    bound=theta.scalarize_params(params)['a']
    x, gtraces, straces, xi, yi, nframes = gen_re_traces(model, params)
    f, axes = build_decision_axis(onset=x[0][0], bound=bound)
    glines = [axes[i].plot([], [], linewidth=1.5)[0] for i, n in enumerate(gtraces)]
    slines = [axes[i].plot([xi[i]], [yi[i]], linewidth=1.5)[0] for i, n in enumerate(straces)]
    f_args = (x, gtraces, glines, straces, slines, params, xi, yi)
    anim=animation.FuncAnimation(f, re_animate_multiax, fargs=f_args, frames=nframes, interval=1, blit=True)
    return anim

def gen_re_traces(model, params):
    sim = deepcopy(model.opt.simulator)
    sim.__update_steps__(dt=.001)
    ssd, nssd, nss, nss_per, ssd_ix = sim.ssd_info
    nsstot = nss * nssd
    params = sim.vectorize_params(params)
    bound = params['a'][0]
    Pg, Tg, xtb = sim.__update_go_process__(params)
    Ps, Ts = sim.__update_stop_process__(params)
    Ts=[Ts[0, i] for i in [2, -1]]
    dvg, dvs = sim.sim_fx(params, analyze=False)
    dvg = dvg[0, :nss, :].reshape(nssd, nss_per, dvg.shape[-1])
    gtraces = [dvg[i, 0] for i in [2, -1]]
    straces = [dvs[0, i, 0] for i in [2, -1]]
    gtraces = [gt[gt <= bound] for gt in gtraces]
    ssi, xinit_ss, integrated, dvgs, dvss = [], [], [], [], []
    for i, (g, s) in enumerate(zip(gtraces, straces)):
        xinit_ss.append(Tg[0] - Ts[i])
        ssi.append(g[:xinit_ss[i]])
        ss = s[:Ts[i]]
        s = np.append(g[:xinit_ss[i]], ss[ss >= 0])
        ixmin = np.min([len(g), len(s)])
        dvgs.append(g[:ixmin])
        dvss.append(s[:ixmin])
    nframes = [len(gt) for gt in dvgs]
    x = params['tr'][0] * 1000 + [np.arange(nf) for nf in nframes]
    sim.__update_steps__(dt=.005)
    return [x, dvgs, dvss, xinit_ss, ssi, np.max(nframes)]

def build_decision_axis(onset, bound, ssd=np.array([300, 400]), tb=650):
    # init figure, axes, properties
    f, axes = plt.subplots(len(ssd), 1, figsize=(7, 7), dpi=300)
    #f.subplots_adjust(wspace=.05, top=.1, bottom=.1)
    f.subplots_adjust(hspace=.05, top=.99, bottom=.09)
    w = tb + 40
    h = bound
    start = onset - 80
    for i, ax in enumerate(axes):
        plt.setp(ax, xlim=(start - 1, w + 1), ylim=(0 - (.01 * h), h + (.01 * h)))
        ax.vlines(x=ssd[i], ymin=0, ymax=h, color="#e74c3c", lw=2.5, alpha=.5)
        ax.hlines(y=h, xmin=start, xmax=w, color='k')
        ax.hlines(y=0, xmin=start, xmax=w, color='k')
        ax.vlines(x=tb, ymin=0, ymax=h, color='#2043B0', lw=2.5, linestyle='-', alpha=.5)
        ax.vlines(x=start + 2, ymin=0, ymax=h, color='k')
        ax.text(ssd[i] + 10, h * .88, str(ssd[i]) + 'ms', fontsize=19)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    sns.despine(top=True, right=True, bottom=True, left=True)
    return f, axes

def re_animate_multiax(i, x, gtraces, glines, straces, slines, params, xi, yi):
    gcolor = '#27ae60'
    scolor = '#e74c3c'
    for n in range(len(x)):
        ex, gt, gl, st, sl, ix, iy = [xx[n] for xx in [x, gtraces, glines, straces, slines, xi, yi]]
        try:
            gl.set_data(ex[:i + 1], gt[:i + 1])
            gl.set_color(gcolor)
            sl.set_data(ex[ix:i + 1], st[ix:i + 1])
            sl.set_color(scolor)
        except Exception:
            return sl, gl
        #f.savefig('animation_frames/movie/img' + str(i) +'.png', dpi=300)
    return sl, gl

def anim_to_html(anim):
    from tempfile import NamedTemporaryFile
    VIDEO_TAG = """<video controls>
             <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
             Your browser does not support the video tag.
            </video>"""
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return VIDEO_TAG.format(anim._encoded_video)


class PBinJ(object):
    """ animated ProgressBar (PB) to be used (inJ)upyter notebooks
    """
    def __init__(self, n, color_n=0, disp=False, progbar=True, infobar=False, ptitle='globalfmin', infotitle='fmin'):

        colors = ['seagreen', '#4f8bef', "#0b559f", "#de2b25"]
        self.color = colors[color_n]
        self.n=n
        self.displayed=False
        self.bars=[]
        self.infobar=infobar
        self.progbar=progbar
        try:
            from IPython.display import clear_output
            self.animate = self.animate_ipython
            self.init_ipython_bars(infotitle=infotitle, ptitle=ptitle)
        except Exception:
            import sys
            self.init_bars(infotitle=infotitle, ptitle=ptitle)

    def update(self, i=None, new_info=None, new_progress=None): 
        self.animate(i=i, new_info=new_info, new_progress=new_progress)

    def init_ipython_bars(self, infotitle=None, ptitle=None):
        import uuid
        if self.infobar:
            self.infoid=str(uuid.uuid4())
            self.ibar="""<div<p>{0}</p> <div style="border: none; width:500px"> <div id="{1}" style="background-color:#FFF; color:#000; text:{2}; width:100.">&nbsp;</div> </div> </div>""".format(infotitle, self.infoid, 'fmin')
            self.bars.append(self.ibar)
        if self.progbar:
            self.pbid=str(uuid.uuid4())
            self.pbar="""<div<p>{0}</p> <div style="border: 1px solid #111; width:500px">
                <div id="{1}" style="background-color:{2}; width:0%%; text:'fmin={3}'; color:#fff;">
                &nbsp;</div> </div> </div>""".format(ptitle, self.pbid, self.color, 'fmin')
            self.bars.append(self.pbar)

    def init_bars(self, infotitle=None, ptitle=None):
        import progressbar
        if self.progbar:
            self.new_prog_string = ''.join([ptitle, ': {0}'])
            self.pbar = progressbar.ProgressBar(maxval=self.n)
            self.bars.append(self.pbar)
        if self.infobar:
            self.new_info_string = ''.join([infotitle, ': {0}'])

    def display_bars(self):
        for bar in self.bars:
            display(HTML(bar))
        self.displayed=True

    def animate_ipython(self, i=None, new_info=None, new_progress=None):
        if self.displayed==False:
            self.display_bars()
        if new_info is not None and self.infobar:
            display(Javascript("$('div#%s').text(%s)" % (self.infoid, str(new_info))))
        if new_progress is not None and self.progbar:
            display(Javascript("$('div#%s').text(%s)" % (self.pbid, str(new_progress))))
        if i is not None and self.progbar:
            display(Javascript("$('div#%s').width('%.2f%%')" % (self.pbid, (i*1./self.n)*100)))

    def animate(self, i=None, new_info=None, new_progress=None):
        if self.displayed==False and self.progbar:
            self.pbar.start()
        if new_info is not None and self.infobar:
            sys.stdout.write('\r'+self.new_info_string.format(str(new_info)))
            sys.stdout.flush()
        if new_progress is not None and self.progbar:
            sys.stdout.write('\r'+self.new_prog_string.format(str(new_progress)))
            sys.stdout.flush()
        if i is not None and self.progbar:
            self.pbar.update(i+1)
