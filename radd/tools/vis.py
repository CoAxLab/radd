#!/usr/local/bin/env python
from __future__ import division
import os
import pandas as pd
import numpy as np
from numpy import array
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from radd.tools import colors, messages, analyze
from scipy.stats.mstats import mquantiles as mq
from numpy import cumsum as cs
from numpy import append as app

sns.set(style='white', rc={'text.color': 'black', 'axes.labelcolor': 'black', 'figure.facecolor': 'white'})

cdict = colors.get_cpals('all')
rpal = cdict['rpal']
bpal = cdict['bpal']
gpal = cdict['gpal']
ppal = cdict['ppal']
heat = cdict['heat']
cool = cdict['cool']
slate = cdict['slate']


def scurves(lines=[],  yerr=[], pstop=.5, ax=None, linestyles=None, colors=None, markers=False, labels=None, mc=None, x=None):
    dont_label = False
    sns.set_context('notebook', font_scale=1.7)
    if ax is None:
        f, ax = plt.subplots(1, figsize=(5.5, 6))
    if colors is None:
        colors = slate(len(lines))
    if labels is None:
        labels = [''] * len(lines)
        dont_label = True
    if linestyles is None:
        linestyles = ['-'] * len(lines)

    lines = [(line) if type(line) == list else line for line in lines]
    pse = []
    scl = 100

    if x is None:
        #x = np.arange(len(lines[0]), dtype='float')[::-1]*50
        x = array([400, 350, 300, 250, 200], dtype='float')

    xtls = x.copy()
    xsim = np.linspace(x.min()-20, x.max()+20, 10000)

    yylabel = 'P(Stop)'
    xxlabel = 'SSD (ms)'

    markers = True
    mclinealpha = [.7, 1] * len(lines)

    x = analyze.res(-x, lower=x[0], upper=x[-1])
    xxlim = (x[0]-20, x[-1]+20)
    print(x)
    for i, yi in enumerate(lines):
        color = colors[i]
        y = analyze.res(yi, lower=yi[-1], upper=yi[0])
        p_guess = (np.mean(x), np.mean(y), .5, .2)
        p, cov, infodict, mesg, ier = optimize.leastsq(analyze.residuals, p_guess, args=(x, y), full_output=1, maxfev=10000, ftol=1.e-20)
        x0, y0, c, k = p
        pxp = analyze.sigmoid(p, xsim)
        idx = (np.abs(pxp - pstop)).argmin()
        pse.append(xsim[idx])
        # Plot the results
        if yerr != []:
            ax.errorbar(x, yi, yerr=yerr[i], color=colors[i], lw=0, marker='o', elinewidth=2, ecolor=colors[i])
            ax.plot(xsim, pxp, linestyle=linestyles[i], lw=1.5, color=colors[i], label=labels[i])
            #ax.errorbar(xp, pxp, yerr=yerr, color=color, ecolor=color, capsize=0, lw=0, elinewidth=3)
        if markers:
            a = mclinealpha[i]
            ax.plot(xsim, pxp, linestyle=linestyles[i], lw=2.5, color=color, label=labels[i], alpha=a)
            for ii in range(len(y)):
                ax.plot(x[ii], y[ii], lw=0, marker='o', ms=9, color=color, markerfacecolor='none', mec=color, mew=1.5, alpha=.8)
        else:
            ax.plot(xsim, pxp, linestyle=linestyles[i], lw=3.5, color=colors[i], label=labels[i])
        pse.append(xsim[idx])

    plt.setp(ax, xticks=x, ylim=(-.01, 1.05), yticks=np.arange(0,1.2,.2))#[0,  1])
    #ax.set_xticklabels([int(xt) for xt in xtls])
    #ax.set_yticklabels(np.arange(0,1.2,.2))#[0.0, 1.0])
    #ax.set_xlabel(xxlabel)
    #ax.set_ylabel(yylabel)

    #ax.legend(loc=0)
    plt.tight_layout()
    sns.despine()
    return (pse)


def plot_fits(y, yhat, cdf=True, plot_params={}, save=False, axes=None, kind='', savestr='fit_plot', split='HL', xlim=(.45, .65), label=None, colors=None, data=None, mc=None):

    sns.set_context('notebook', font_scale=1.6)
    pp = plot_params
    if axes is None:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)
    else:
        ax1, ax2, ax3 = axes
    if colors is None:
        colors = ["#4168B7"] * 2
    if kind == 'pro':
        xlim = (.48, .58)
    # pull out data vectors
    sc, gq, eq = unpack_yvector(y, kind=kind)
    fitsc, fitgq, fiteq = unpack_yvector(yhat, kind=kind)

    if data is not None:
        axes, pp = plot_data_dists(data, kind=kind, cdf=cdf, axes=[ax1, ax2, ax3], data_type='real')
        fit_cq, fit_eq = [analyze.kde_fit_quantiles(q, bw=.01) for q in [fitgq, fiteq]]
    else:
        kdefits = [analyze.kde_fit_quantiles(q, bw=.01) for q in [gq, fitgq, eq, fiteq]]
        dat_cq, fit_cq, dat_eq, fit_eq = kdefits
        axes, pp = plot_data_dists(data=[dat_cq, dat_eq], kind=kind, cdf=cdf, axes=[ax1, ax2, ax3], data_type='interpolated')

    shade = pp['shade']
    lw = pp['lw']
    ls = pp['ls']
    alpha = pp['alpha']
    bw = pp['bw']
    #sns.distplot(fit_cq, kde=False, color=colors[0], norm_hist=True, ax=ax1, bins=45)
    #sns.distplot(fit_eq, kde=False, color=colors[1], norm_hist=True, ax=ax2, bins=45)
    sns.kdeplot(fit_cq, color='Blue', cumulative=cdf, linestyle='--',
                bw=.01, ax=ax1, linewidth=3, alpha=.70, shade=shade, label=label)
    sns.kdeplot(fit_eq, color='Red', cumulative=cdf, linestyle='--',
                bw=.01, ax=ax2, linewidth=3, alpha=.60, shade=shade)

    for ax in axes:
        if ax.is_last_col():
            continue
        ax.set_xlim(xlim[0], xlim[1])
        if ax.is_first_col():
            ax.set_ylabel('P(RT)')
        if ax.is_last_row():
            ax.set_xlabel('RT (ms)')
        ax.set_xticklabels([int(xx) for xx in ax.get_xticks() * 1000])

    # Plot observed and predicted stop curves
    scurves([sc, fitsc], kind=kind, linestyles=['-', '--'], ax=ax3, colors=colors, markers=True, mc=mc)
    plt.tight_layout()
    sns.despine()
    if save:
        plt.savefig(savestr + '.png', format='png', dpi=300)


def react_fit_plots(m, color="#4168B7", is_flat=False, save=False):

    #redata = m.data
    sns.set_context('notebook', font_scale=1.6)
    if is_flat:
        f, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes = array([axes])
        fits = [m.fits[0]]
        y = [m.flat_y]
        ncond = 1
    else:
        f, axes = plt.subplots(2, 3, figsize=(12, 7))
        fits = m.fits.reshape(2, 16)
        y = m.avg_y
        ncond = m.ncond

    labels = ['Baseline', 'Caution']

    for i in range(ncond):
        plot_fits(y[i], fits[i], kind='', colors=[color] * 2, axes=axes[i])

    for ax in axes.flatten():
        if ax.is_last_col():
            continue
        ax.set_xlim(.4, .65)
        if ax.is_first_col():
            ax.set_ylabel('P(RT)')
        if ax.is_last_row():
            ax.set_xlabel('RT (ms)')
        ax.set_xticklabels([int(xx) for xx in ax.get_xticks() * 1000])

    if save:
        kind = m.kind
        mdescr = messages.describe_model(m.depends_on)
        if 'and' in mdescr:
            mdeps = mdescr.split(' and ')
            mdescr = '_'.join([dep for dep in mdeps])

        savestr = '_'.join([kind, mdescr, 'fits'])
        plt.savefig(savestr + '.png', dpi=500)
        plt.savefig(savestr + '.svg', rasterized=True)


def profits(y, yhat, cdf=False, plot_params={}, save=False, axes=None, kind='', savestr='fit_plot', split='HL', xlim=(.4, .65), label=None, colors=None, data=None, mc=None):
    sns.set_context('notebook', font_scale=1.6)

    pp = plot_params
    if axes is None:
        f, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(14, 5.5), sharey=False)
    else:
        ax1, ax2, ax3 = axes
    if colors is None:
        colors = ["#4168B7"] * 2
    if kind == 'pro':
        xlim = (.48, .58)
    # pull out data vectors
    sc, gq, eq = unpack_yvector(y, kind=kind)
    fitsc, fitgq, fiteq = unpack_yvector(yhat, kind=kind)
    kdefits = [analyze.kde_fit_quantiles(q, bw=.01) for q in [gq, fitgq, eq, fiteq]]
    dat_cq, fit_cq, dat_eq, fit_eq = kdefits
    sns.distplot(fit_cq, kde=False, color=colors[0], norm_hist=True, ax=ax1, bins=25)
    sns.distplot(fit_eq, kde=False, color=colors[1], norm_hist=True, ax=ax2, bins=25)

    for ax in axes:
        if ax.is_last_col():
            continue
        ax.set_xlim(xlim[0], xlim[1])
        if ax.is_first_col():
            ax.set_ylabel('P(RT)')
        if ax.is_last_row():
            ax.set_xlabel('RT (ms)')
        ax.set_xticklabels([int(xx) for xx in ax.get_xticks() * 1000])

    # Plot observed and predicted stop curves
    scurves([sc, fitsc], kind=kind, linestyles=['-', '--'], ax=ax3, colors=colors, markers=True, mc=mc)
    plt.tight_layout()
    sns.despine()
    if save:
        plt.savefig(savestr + '.png', format='png', dpi=300)


def plot_data_dists(data, kind='', data_type='real', cdf=False, axes=[], get_rts=False):

    emp_kq = lambda rts: analyze.kde_fit_quantiles(mq(rts, prob=np.arange(0, 1, .02)), bw=.008)

    ax1, ax2, ax3 = axes
    if data_type == 'real':
        if kind == 'pro':
            hi_rts = data.query('response==1 & pGo>.5').rt.values
            lo_rts = data.query('response==1 & pGo<.5').rt.values
        elif kind == '':
            hi_rts = data.query('response==1 & acc==1').rt.values
            lo_rts = data.query('response==1 & acc==0').rt.values
        dat_cq = emp_kq(hi_rts)
        dat_eq = emp_kq(lo_rts)
        if get_rts:
            return axes, plot_params, rts
    elif data_type == 'interpolated':
        dat_cq, dat_eq = data

    if cdf:
        shade = False
        alpha = 1
        bw = .01
        lw = 3.5
        ls = '--'
        sns.kdeplot(dat_cq, color='k', cumulative=cdf, ax=ax1, linewidth=lw, linestyle='-')
        sns.kdeplot(dat_eq, color='k', cumulative=cdf, ax=ax2, linewidth=lw, linestyle='-')
    else:
        # set parameters for simulated plots
        shade = True
        alpha = .5
        bw = .01
        lw = 2.5
        ls = '-'
        sns.distplot(dat_cq, kde=False, color='k', norm_hist=True, ax=ax1, bins=50)
        sns.distplot(dat_eq, kde=False, color='k', norm_hist=True, ax=ax2, bins=50)

    plot_params = {'shade': shade, 'alpha': alpha, 'bw': bw, 'lw': lw, 'ls': ls}
    if get_rts:
        return axes, plot_params, rts
    return axes, plot_params


def plot_reactive_fits(model, cumulative=True, save=False, col=None):

    sns.set_context('notebook', font_scale=1.6)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    y = model.avg_y
    r, c = y.shape
    if col is None:
        col = [bpal(2), rpal(2)]
    yhat = model.fits.reshape(r, c)
    yh_id = 'fits.png'

    if save:
        savestr = '_'.join([get_model_name(model), yh_id])

    linestyles = ['-', '--']
    for i in range(model.ncond):
        sc, cq, eq = unpack_yvector(y[i])
        sc_hat, cqhat, eqhat = unpack_yvector(yhat[i])

        for ii, qc in enumerate([cq, cqhat]):
            kdeqc = analyze.kde_fit_quantiles(qc, bw=.01)
            sns.kdeplot(kdeqc, linestyle=linestyles[ii], cumulative=cumulative, ax=ax1, linewidth=3.5, color=col[i][ii])
        for ii, qe in enumerate([eq, eqhat]):
            kdeqe = analyze.kde_fit_quantiles(qe, bw=.01)
            sns.kdeplot(kdeqe, cumulative=cumulative, linestyle=linestyles[ii], ax=ax2, linewidth=3.5, color=col[i][ii])

        labels = [' '.join([model.labels[i], x]) for x in ['data', 'model']]
        # Plot observed and predicted stop curves
        scurves([sc, sc_hat], labels=labels, kind='', colors=col[i], linestyles=['-', '--'], ax=ax3, markers=True)
        plt.tight_layout()
        sns.despine()

    ax1.set_title('Correct RTs', fontsize=17)
    ax2.set_title('SS Trial RTs (Errors)', fontsize=17)
    ax3.set_title('P(Stop) Across SSD', fontsize=17)
    #[axx.set_xlim(.470,.650) for]

    for axx in [ax1, ax2]:
        axx.set_xlim(.470, .650)
        xxticks = axx.get_xticks()
        axx.set_ylabel('P(RT<t)')
        axx.set_xlabel('RT (s)')
        #axx.set_ylim(-.05, 1.05)
        axx.set_xticks(xxticks)
        axx.set_xticklabels([int(xx) for xx in xxticks * 1000])
        axx.set_xlim(.470, .650)
    if save:
        plt.savefig(savestr, dpi=300)


def unpack_yvector(y, kind=''):

    if 'pro' in kind:
        sc, gq, eq = y[:6], y[6:11], y[11:]
    else:
        sc, gq, eq = y[1:6], y[6:11], y[11:]

    return sc, gq, eq


def get_model_name(model):
    mname = model.kind
    mdep = messages.describe_model(model.depends_on)
    if 'x' in model.kind:
        mname = '_'.join([mname, model.dynamic])
    mname = '_'.join([mname, mdep])
    return mname


def plot_idx_fits(obs, sim, kind='', save=False):

    if kind == '':
        df = df.where(df > 0).dropna()
        for idx, idx_c in obs.iterrows():
            try:
                save_str = '_'.join([str(idx), idx_c['Cond'], 'pred'])
                y = idx_c.loc['Go':'e90'].values.astype(np.float)
                yhat = df.iloc[idx, :].values.astype(np.float)
                plot_fits(y, yhat, kind='', save=save, savestr=save_str)
            except Exception:
                continue
    elif kind == 'pro':
        """
        TODO
        """

        df = None


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


def gen_pro_traces(ptheta, bias_vals=[], bias='v', integrate_exec_ss=False, return_exec_ss=False, pgo=np.arange(0, 1.2, .2)):

    dvglist = []
    dvslist = []

    if bias_vals == []:
        deplist = np.ones_like(pgo)

    for val, pg in zip(bias_vals, pgo):
        ptheta[bias] = val
        ptheta['pGo'] = pg
        dvg, dvs = RADD.run(ptheta, ntrials=10, tb=.565)
        dvglist.append(dvg[0])

    if pg < .9:
        dvslist.append(dvs[0])
    else:
        dvslist.append([0])

    if integrate_exec_ss:
        ssn = len(dvslist[0])
        traces = [np.append(dvglist[i][:-ssn], (dvglist[i][-ssn:] + ss) - dvglist[i][-ssn:]) for i, ss in enumerate(dvslist)]
        traces.append(dvglist[-1])
        return traces

    elif return_exec_ss:
        return [dvglist, dvslist]
    else:
        return dvglist


def gen_re_traces(model, params, integrate_exec_ss=False, ssdlist=np.arange(.2, .45, .05), integrate=False):

    nssd = model.simulator.nssd
    nss = model.simulator.nss
    nsstot = nss * nssd
    nc = model.simulator.ncond

    dvg, dvs = model.simulate(params, analyze=False, return_traces=True)
    gtraces = dvg[0, :nssd]
    straces = [dvs[0, i, i] for i in range(nssd)]

    Ps, Ts = model.simulator.__update_stop_process__(params)
    Pg, Tg = model.simulator.__update_go_process__(params)
    bound = params['a'][0]
    gtraces = [gt[gt < bound] for gt in gtraces]
    ssi, xinit_ss, integrated, dvgs, dvss = [], [], [], [], []
    for i, (g, s) in enumerate(zip(gtraces, straces)):
        xinit_ss.append(Tg[0] - Ts[i])
        ssi.append(g[:xinit_ss[i]])
        ss = s[:Ts[i]]
        s = np.append(g[:xinit_ss[i]], ss[ss >= 0])
        ixmin = np.min([len(g), len(s)])
        dvgs.append(g[:ixmin])
        dvss.append(s[:ixmin])
        if integrate:
            tx = xinit_ss[i]
            integrated.append(app(g[:tx], cs(app(g[tx], (np.diff(g[tx:]) + np.diff(s[tx:]))))))

    nframes = [len(gt) for gt in dvgs]
    x = params['tr'][0] * 1000 + [np.arange(nf) for nf in nframes]
    return [x, dvgs, dvss, xinit_ss, ssi, np.max(nframes)]


def build_decision_axis(onset, bound, ssd=np.arange(200, 450, 50), tb=650):

    # init figure, axes, properties
    f, axes = plt.subplots(len(ssd), 1, figsize=(5, 5), dpi=300)
    #f.subplots_adjust(wspace=.05, top=.1, bottom=.1)
    f.subplots_adjust(hspace=.05, top=.99, bottom=.05)
    w = tb + 40
    h = bound
    start = onset - 80
    # c=["#e74c3c", '#27ae60', '#4168B7', '#8E44AD']
    for i, ax in enumerate(axes):
        plt.setp(ax, xlim=(start - 1, w + 1), ylim=(0 - (.01 * h), h + (.01 * h)))
        ax.vlines(x=ssd[i], ymin=0, ymax=h, color="#e74c3c", lw=1.5, alpha=.5)
        ax.hlines(y=h, xmin=start, xmax=w, color='k')
        ax.hlines(y=0, xmin=start, xmax=w, color='k')
        ax.vlines(x=tb, ymin=0, ymax=h, color='#2043B0', lw=1.5, linestyle='-', alpha=.5)
        ax.vlines(x=start + 2, ymin=0, ymax=h, color='k')
        ax.text(ssd[i] + 10, h * .87, str(ssd[i]) + 'ms', fontsize=15)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    sns.despine(top=True, right=True, bottom=True, left=True)

    return f, axes

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


def pro_animate(i, x, protraces, prolines):

    clist = sns.color_palette('autumn', n_colors=6)[::-1]

    for nline, (pline, ptrace) in enumerate(zip(prolines, protraces)):
        pline.set_data(x[:i + 1], ptrace[:i + 1])
        pline.set_color(clist[nline])

    return prolines,


def anim_to_html(anim):

    from tempfile import NamedTemporaryFile

    VIDEO_TAG = """<video controls>
             <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
             Your browser does not support the video tag.
            </video>"""

    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, dpi=300, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")
    return VIDEO_TAG.format(anim._encoded_video)


def display_animation(anim):
    from IPython.display import HTML
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))


def plot_all_traces(DVg, DVs, theta, ssd=np.arange(.2, .45, .05), kind='dpm'):

    ncond = DVg.shape[0]
    nssd = DVs.shape[1]
    f, axes = plt.subplots(nssd, ncond, figsize=(12, 14))
    for i in range(ncond):
        params = {k: v[i] if hasattr(v, '__iter__') else v for k, v in theta.items()}
        for ii in range(nssd):
            plot_traces(DVg=DVg[i], DVs=DVs[i, ii], ssd=ssd[ii], sim_theta=params, ax=axes[ii, i], kind=kind)
    return f


def plot_traces(DVg=[], DVs=[], sim_theta={}, kind='dpm', ssd=.450, ax=None, tau=.005, tb=.650, cg='#16a085', cr=['#f1c40f', '#e74c3c', '#c0392b']):

    if ax is None:
        f, ax = plt.subplots(1, figsize=(8, 5))
    tr = sim_theta['tr']
    a = sim_theta['a']
    z = 0
    for i, igo in enumerate(DVg):
        ind = np.argmax(igo >= a)
        xx = [np.arange(tr, tr + (len(igo[:ind - 1]) * tau), tau), np.arange(tr, tb, tau)]
        x = xx[0 if len(xx[0]) < len(xx[1]) else 1]
        plt.plot(x, igo[:len(x)], color=cg, alpha=1, linewidth=2)

        ax.vlines(ssd, 0, a, linewidth=1, alpha=.5, color='#6C7A89')
        if kind in ['irace', 'dpm', 'iact'] and i < len(DVs[0]):
            if np.any(DVs[0] <= 0):
                ind = np.argmax(DVs[0] <= 0)
            else:
                ind = np.argmax(DVs[0] >= a)
            xx = [np.arange(ssd, ssd + (len(DVs[0][:ind - 1]) * tau), tau), np.arange(ssd, tb, tau)]
            x = xx[0 if len(xx[0]) < len(xx[1]) else 1]
            crx = sns.blend_palette(cr, n_colors=len(x))

            for ii in range(len(x)):
                if ii == len(x) - 1:
                    break
                plt.plot([x[ii], x[ii + 1]], [np.asscalar(DVs[0][ii]), np.asscalar(DVs[0][ii + 1])], color=crx[ii], alpha=.9, linewidth=2)

    xlow = np.min([tr, ssd])
    xlim = (xlow * .75, 1.05 * tb)

    print xlim
    if kind == 'pro' or np.any(DVs <= 0):
        ylow = 0
        ylim = (-.03, a * 1.03)
    else:
        ylow = z
        ylim = (z - .03, a * 1.03)

    plt.setp(ax, xlim=xlim, ylim=ylim)
    ax.hlines(y=z, xmin=xlow, xmax=tb, linewidth=2, linestyle='--', color="k", alpha=.5)
    ax.hlines(y=a, xmin=xlow, xmax=tb, linewidth=2, linestyle='-', color="k")
    ax.hlines(y=ylow, xmin=xlow, xmax=tb, linewidth=2, linestyle='-', color="k")
    ax.vlines(x=xlow, ymin=ylow * .998, ymax=a * 1.002, linewidth=2, linestyle='-', color="k")
    sns.despine(top=True, bottom=True, right=True, left=True)
    ax.set_xlim(xlim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    return ax
