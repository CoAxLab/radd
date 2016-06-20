#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
import os
import re
from numpy import array
from sklearn.neighbors import KernelDensity
from scipy.stats.mstats import mquantiles as mq
from scipy.stats.mstats_extras import mjci
from scipy import optimize
import functools


def assess_fit(finfo, lmMin=None, fitparams=None):
    """ calculate fit statistics
    """
    y, wts, yhat = fitparams['y'], fitparams['wts'], fitparams['yhat']
    # fill finfo dict with goodness-of-fit info
    finfo['chi'] = np.sum(wts * (y.flatten() - yhat)**2)
    finfo['ndata'] = len(yhat)
    finfo['cnvrg'] = lmMinimizer.success
    finfo['nfev'] = lmMinimizer.nfev
    finfo['nvary'] = len(lmMinimizer.var_names)
    finfo = pd.Series(finfo)
    chisqr = finfo.chi
    finfo['df'] = finfo.ndata - finfo.nvary
    finfo['rchi'] = chisqr / finfo.df
    finfo['logp'] = finfo.ndata * np.log(finfo.rchi)
    finfo['AIC'] = finfo.logp + 2 * finfo.nvary
    finfo['BIC'] = finfo.logp + np.log(finfo.ndata * finfo.nvary)
    return finfo

def rangl_data(data, tb=.555, quantiles=([.1, .3, .5, .7, .9]), ssd_method='all'):
    """ called by __make_dataframes__ to generate observed dataframes and iterables for
    subject fits
    """
    gac = data.query('ttype=="go"').acc.mean()
    grt = data.query('ttype=="go" & acc==1').rt.values
    ert = data.query('response==1 & acc==0').rt.values
    gq = mq(grt, prob=quantiles)
    eq = mq(ert, prob=quantiles)
    data_vector = [gac, gq, eq]
    if 'ssd' in data.columns:
        stopdf = data.query('ttype=="stop"')
        if ssd_method=='all':
            sacc=stopdf.groupby('ssd').mean()['acc'].values
        elif ssd_method=='central':
            sacc = np.array([stopdf.mean()['acc']])
        data_vector.insert(1, sacc)
    return np.hstack(data_vector)

def remove_outliers(data, sd=1.5, verbose=False):
    df = data.copy()
    ssdf = df[df.response == 0]
    godf = df[df.response == 1]
    bound = godf.rt.std() * sd
    rmslow = godf[godf['rt'] < (godf.rt.mean() + bound)]
    clean_go = rmslow[rmslow['rt'] > (godf.rt.mean() - bound)]
    clean = pd.concat([clean_go, ssdf])
    if verbose:
        pct_removed = len(clean) * 1. / len(df)
        print("len(df): %i\nbound: %s \nlen(cleaned): %i\npercent removed: %.5f" % (len(df), str(bound), len(clean), pct_removed))
    return clean

def ensure_numerical_wts(wts, flat_wts):
    # test inf
    wts[np.isinf(wts)] = np.median(wts[~np.isinf(wts)])
    flat_wts[np.isinf(flat_wts)] = np.median(flat_wts[~np.isinf(flat_wts)])
    # test nan
    wts[np.isnan(wts)] = np.median(wts[~np.isnan(wts)])
    flat_wts[np.isnan(flat_wts)] = np.median(flat_wts[~np.isnan(flat_wts)])
    return wts, flat_wts

def kde_fit_quantiles(rtquants, nsamples=1000, bw=.1):
    """ takes quantile estimates and fits cumulative density function
    returns samples to pass to sns.kdeplot()
    """
    kdefit = KernelDensity(kernel='gaussian', bandwidth=bw).fit(rtquants)
    samples = kdefit.sample(n_samples=nsamples).flatten()
    return samples

def sigmoid(p, x):
    x0, y0, c, k = p
    y = c / (1 + np.exp(k * (x - x0))) + y0
    return y

def residuals(p, x, y):
    return y - sigmoid(p, x)

def res(arr, lower=0.0, upper=1.0):
    arr = arr.copy()
    if lower > upper:
        lower, upper = upper, lower
    arr -= arr.min()
    arr *= (upper - lower) / arr.max()
    arr += lower
    return arr

def get_observed_vector(rt, quantiles=array([10, 30, 50, 70, 90])):
    """ takes array of rt values and returns binned counts (trials
    that fall between each set of quantiles in quantiles). also returns
    the total number of observations (len(rt)) and the RT values at those
    quantiles (rtquant)
    """
    inter_quantiles = array([quantiles[0] - 0] + [quantiles[i] - quantiles[i - 1] for i in range(1, len(quantiles))] + [100 - quantiles[-1]])
    rtquant = mq(rt, prob=quantiles * .01)
    ocounts = np.ceil((inter_quantiles) * .01 * len(rt)).astype(int)
    n_obs = np.sum(ocounts)
    return [ocounts, rtquant, n_obs]

def get_expected_vector(simrt, obsinfo):
    """ calculates the expected frequencies of responses for a
    set of simulated RTs, given. obsinfo is output of
    get_observed_vector() -->  [ocounts, rtquant, n_obs]
    simrt = pd.Series(simrt)
    """
    counts, q, n_obs = obsinfo[0], obsinfo[1], obsinfo[2]
    first = array([len(simrt[simrt.between(simrt.min(), q[0])]) / len(simrt)]) * n_obs
    middle = array([len(simrt[simrt.between(q[i - 1], q[i])]) / len(simrt) for i in range(1, len(q))]) * n_obs
    last = array([len(simrt[simrt.between(q[-1], simrt.max())]) / len(simrt)]) * n_obs
    expected = np.ceil(np.hstack([first, middle, last]))
    return expected

def ssrt_calc(df, avgrt=.3):
    dfstp = df.query('ttype=="stop"')
    dfgo = df.query('choice=="go"')
    pGoErr = ([idf.response.mean() for ix, idf in dfstp.groupby('idx')])
    nlist = [int(pGoErr[i] * len(idf)) for i, (ix, idf) in enumerate(df.groupby('idx'))]
    GoRTs = ([idf.rt.sort(inplace=False).values for ix,idf in dfgo.groupby('idx')])
    ssrt_list = ([GoRTs[i][nlist[i]] for i in np.arange(len(nlist))]) - avgrt
    return ssrt_list

def get_obs_quant_counts(df, quantiles=([.10, .30, .50, .70, .90])):
    if type(df) == pd.Series:
        rt = df.copy()
    else:
        rt = df.rt.copy()
    inter_quantiles = [quantiles[0] - 0] + [quantiles[i] - quantiles[i - 1] for i in range(1, len(quantiles))] + [1.00 - quantiles[-1]]
    obs_quant = mq(rt, prob=quantiles)
    observed = np.ceil((inter_quantiles) * len(rt) * .94).astype(int)
    return observed, obs_quant

def get_exp_counts(simdf, obs_quant, n_obs, quantiles=([.10, .30, .50, .70, .90])):
    if type(simdf) == pd.Series:
        simrt = simdf.copy()
    else:
        simrt = simdf.rt.copy()
    exp_quant = mq(simrt, prob=quantiles)
    oq = obs_quant
    expected = np.ceil(np.diff([0] + [pscore(simrt, oq_rt) * .01 for oq_rt in oq] + [1]) * n_obs)
    return expected, exp_quant

def get_intersection(iter1, iter2):
    """ get the intersection of two iterables ("items in-common")
    """
    intersect_set = set(iter1).intersection(set(iter2))
    return ([i for i in intersect_set])
