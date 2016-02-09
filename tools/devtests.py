#!/usr/local/bin/env python
from __future__ import division
from radd import utils, build
from scipy.stats.mstats import mquantiles as mq
from lmfit import report_fit, fit_report
import numpy as np
import pandas as pd
from lmfit import Parameters, Minimizer
from radd.utils import *
from scipy.optimize import minimize as mina


def compare_ols_wls_predictions(data, inits, wts=None, save=False, track='predictions', depends=['xx'], model='', ntrials=5000, maxfev=50, ftol=1.e-3, xtol=1.e-3, all_params=1, disp=True):

    m = build.Model(data=data, inits=inits, depends_on={'v': 'Cond'}, fit='subjects')
    m.prepare_fit()
    y = m.observed.query('Cond=="bsl"').mean().iloc[1:].values

    wls_update = track_optimization(y, inits=inits, wts=m.wts['bsl'], collector=[], track=track, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, all_params=all_params)

    ols_update = track_optimization(y, inits=inits, wts=np.ones_like(m.wts['bsl']), collector=[], track=track, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, all_params=all_params)

    sns.set_context('notebook', font_scale=1.4)
    htmin, htmax = [], []
    nsims = np.min([len(ols_update), len(wls_update)])
    alpha = np.linspace(.04, .08, nsims)

    for i in range(nsims):

        if track == 'predictions':
            wres = y - wls_update[i]
            ores = y - ols_update[i]
            if i > 0:
                wres_pre = y - wls_update[i - 1]
                ores_pre = y - ols_update[i - 1]
        else:
            wres = wls_update[i]
            ores = ols_update[i]
            if i > 0:
                wres_pre = wls_update[i - 1]
                ores_pre = ols_update[i - 1]

        plt.plot(wres, color='#3498db', alpha=alpha[i], linestyle='-', lw=.42)
        plt.plot(ores, color='#e84b3a', alpha=alpha[i], linestyle='-', lw=.42)

        if i != 0:
            wls_fill = plt.fill_between(np.arange(16), wres_pre, wres, facecolor='#3498db', alpha=alpha[i])
            ols_fill = plt.fill_between(np.arange(16), ores_pre, ores, facecolor='#e84b3a', alpha=alpha[i])
        htmin.append(np.hstack([ores, wres]).min())
        htmax.append(np.hstack([ores, wres]).max())

    ax = plt.gca()
    ax.plot(wres, label='wls', color='#1270b9', alpha=1,
            linestyle='-', marker='o', ms=3, lw=2)
    ax.plot(ores, label='ols', color='#c0392b', alpha=1,
            linestyle='-', marker='o', ms=3, lw=2)

    ht = ([np.min(htmin), np.max(htmax)])
    ax.fill_betweenx(ht, x1=0, x2=5, color='#2c3e50', alpha=.1)
    ax.fill_betweenx(ht, x1=5, x2=10, color='#48647c', alpha=.1)
    ax.fill_betweenx(ht, x1=10, x2=15, color='#2c3e50', alpha=.1)
    ax.hlines(y=0, xmin=0, xmax=15, color='k',
              linestyle='-', lw=2, label='Perfect Fit')
    plt.setp(ax, xlim=(0, 15), ylim=(ht[0], ht[1]))
    ax.set_xlabel('Contrast Vector', fontsize=17)
    ax.set_ylabel("Residuals", fontsize=17)
    ax.set_xticklabels([])
    sns.despine()
    ax.legend(loc=0, fontsize=16)

    ax.text(2., -.08, 'SPC', fontsize=20)
    ax.text(6.9, -.08, 'GRQ', fontsize=20)
    ax.text(12.0, -.08, 'ERQ', fontsize=20)
    plt.tight_layout()
    if save:
        plt.savefig('~/Dropbox/TheXostFx.png', dpi=900)
    return [y, wls_update, ols_update]


def track_optimization(y, inits={}, collector=[], track='residuals', depends=['xx'], wts=None, model='', ntrials=5000, maxfev=50, ftol=1.e-3, xtol=1.e-3, all_params=0, disp=True):
    """
    passess a list to the objective function (model) that
    collects each predicted or residual model output during
    the optimization routine.

    useful when testing new cost functions
    """

    lim = {'tr': (.001, .5), 'v': (.01, 4.),
           'ssv': (-4., -.01),  'a': (.01, .6)}

    p = Parameters()

    if all_params:
        p.add('a', value=inits['a'], vary=1, min=lim['a'][0], max=lim['a'][1])
        p.add('ssv', value=-abs(inits['ssv']), vary=1, min=lim['ssv'][0], max=lim['ssv'][1])
        p.add('v', value=inits['v'], vary=1, min=lim['v'][0], max=lim['v'][1])
        p.add('zperc', value=inits['z'] / inits['a'], vary=1, min=.01, max=.99)
        p.add('tr', value=inits['tr'], vary=1, min=lim['tr'][0], max=lim['tr'][1])
        p.add('z', expr="zperc*a")
    else:
        for key, val in inits.items():
            if key in depends:
                p.add(key, value=val, vary=1, min=lim[key][0], max=lim[key][1])
                continue
            p.add(key, value=val, vary=0)

    if wts is None:
        wts = np.ones_like(y)

    popt = Minimizer(recost, p, fcn_args=(y, ntrials), fcn_kws={'wts': m.wts['bsl']})
    popt.scalar_minimize(maxfev=50, full_output=True, disp=True, method='differential_evolution')
    popt.fmin(maxfev=maxfev, ftol=ftol, xtol=xtol, full_output=True, disp=disp)

    return collector


def recost_collector(theta, y=None, ntrials=5000, collector=[], track='residuals', wts=None, pGo=.5, fit=True, ssd=np.arange(.2, .45, .05)):
    """
    collects residuals or predictions during
    optimization routine
    """

    if not type(theta) == dict:
        theta = theta.valuesdict()

    a, tr, v, ssv, z = theta['a'], theta['tr'], theta['v'], -abs(theta['ssv']),  theta['z']
    nss = int((1 - pGo) * ntrials)
    dvg, dvs = cRADD.run(a, tr, v, ssv, z, ssd, nss=nss, ntot=ntrials)
    yhat = cRADD.analyze_reactive(dvg, dvs, a, tr, ssd, nss=nss)

    wtc, wte = wts[0], wts[1]
    cost = np.hstack([y[:6] - yhat[:6], wtc * y[6:11] - wtc * yhat[6:11], wte * y[11:] - wte * yhat[11:]]).astype(np.float32)

    if track == 'residuals':
        collector.append(cost)
    else:
        collector.append(yhat)

    return cost


def recost_scipy(x0, y=None, wts=None, ntrials=2000, pGo=.5, ssd=np.arange(.2, .45, .05)):

    a, tr, v, ssv, z = p[0], p[1], p[2], p[3], p[4]
    nss = int((1 - pGo) * ntrials)
    dvg, dvs = cRADD.run(a, tr, v, ssv, z, ssd, nss=nss, ntot=ntrials)
    yhat = cRADD.analyze_reactive(dvg, dvs, a, tr, ssd, nss=nss)

    if wts is None:
        cost = np.sum((y[0] - yhat[0])**2 + (y[1:6] - yhat[1:6]) ** 2 + (y[6:11] - yhat[6:11])**2 + (y[11:] - yhat[11:])**2)
    else:
        wta, wtc, wte = wts[0], wts[1], wts[2]
        cost = np.hstack([wta * (y[:6] - yhat[:6]), wtc * y[0] * (y[6:11] - yhat[6:11]), wte * y(y[11:] - yhat[11:])]).astype(np.float32)

    return cost
