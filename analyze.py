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


def assess_fit(finfo):
    """ calculate fit statistics
    """

    finfo = pd.Series(finfo)
    chisqr = finfo.chi
    finfo['df'] = finfo.ndata - finfo.nvary
    finfo['rchi'] = chisqr / finfo.df

    finfo['logp'] = finfo.ndata * np.log(finfo.rchi)
    finfo['AIC'] = finfo.logp + 2 * finfo.nvary
    finfo['BIC'] = finfo.logp + np.log(finfo.ndata * finfo.nvary)

    return finfo


def finfo_to_params(finfo='./finfo.csv', pc_map=None):

    pnames = set(['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso'])
    if isinstance(finfo, str):
        finfo = pd.read_csv(finfo, header=None, names=['id', 'vals'], index_col=0)
        finfo = pd.Series(finfo.to_dict()['vals'])
    elif isinstance(finfo, dict):
        finfo = pd.Series(finfo)
    elif isinstance(finfo, pd.Series):
        pass

    plist = list(pnames.intersection(finfo.index))
    params = {k: finfo[k] for k in plist}

    if pc_map != None:
        for pkey, pclist in pc_map.items():
            params[pkey] = array([params[pkc] for pkc in pclist])
    return params


def remove_outliers(df, sd=1.5, verbose=False):

    ssdf = df[df.response == 0]
    godf = df[df.response == 1]
    bound = godf.rt.std() * sd
    rmslow = godf[godf['rt'] < (godf.rt.mean() + bound)]
    clean_go = rmslow[rmslow['rt'] > (godf.rt.mean() - bound)]

    clean = pd.concat([clean_go, ssdf])
    if verbose:
        pct_removed = len(clean) * 1. / len(df)
        print "len(df): %i\nbound: %s \nlen(cleaned): %i\npercent removed: %.5f" % (len(df), str(bound), len(clean), pct_removed)

    return clean


def rangl_data(data, data_style='re', kind='dpm', tb=.555, prob=([.1, .3, .5, .7, .9])):
    """ called by __make_dataframes__ to generate observed dataframes and iterables for
    subject fits
    """
    if data_style == 're':
        gac = data.query('ttype=="go"').acc.mean()
        sacc = data.query('ttype=="stop"').groupby('ssd').mean()['acc'].values
        grt = data.query('ttype=="go" & acc==1').rt.values
        ert = data.query('response==1 & acc==0').rt.values
        gq = mq(grt, prob=prob)
        eq = mq(ert, prob=prob)
        return np.hstack([gac, sacc, gq, eq])

    elif data_style == 'pro':
        godf = data[data.response == 1]
        godf['response'] = np.where(godf.rt < tb, 1, 0)
        data = pd.concat([godf, data[data.response == 0]])
        return 1 - data.response.mean()


def rt_quantiles(data, split_col='HL', include_zero_rts=False, tb=.555, nrt_cond=2, prob=np.array([.1, .3, .5, .7, .9])):
    """ called by __make_dataframes__ for proactive models to generate observed
    dataframes and iterables for subject fits, specifically aggregates proactive rts
    into a smaller number of conditions to offset low trial count issues
    """

    if include_zero_rts:
        godfx = data[(data.response == 1)]
    else:
        godfx = data[(data.response == 1) & (data.pGo > 0.)]
    godfx.loc[:, 'response'] = np.where(godfx.rt < tb, 1, 0)
    godf = godfx.query('response==1')
    if split_col == None:
        rts = godf[godf.rt <= tb].rt.values
        return mq(rts, prob=prob)

    rtq = []
    for i in range(1, nrt_cond + 1):
        if i not in godf[split_col].unique():
            rtq.append(array([np.nan] * len(prob)))
        else:
            rts = godf[(godf[split_col] == i) & (godf.rt <= tb)].rt.values
            rtq.append(mq(rts, prob=prob))

    return np.hstack(rtq)


def resample_data(data, data_style='re', n=120, kind='dpm', tb=.555):
    """ generates n resampled datasets using rwr()
    for bootstrapping model fits
    """
    df = data.copy()
    bootlist = list()
    if n == None:
        n = len(df)
    if data_style == 're':
        for ssd, ssdf in df.groupby('ssd'):
            boots = ssdf.reset_index(drop=True)
            orig_ix = np.asarray(boots.index[:])
            resampled_ix = rwr(orig_ix, get_index=True, n=n)
            bootdf = ssdf.irow(resampled_ix)
            bootlist.append(bootdf)
            # concatenate and return all resampled conditions
        return rangl_data(pd.concat(bootlist))
    elif data_style == 'pro':
        pvec = list()
        rtvec = list()
        df = df.copy()
        for pg, pgdf in df.groupby('pGo'):
            boots = pgdf.reset_index(drop=True)
            orig_ix = np.asarray(boots.index[:])
            resampled_ix = rwr(orig_ix, get_index=True, n=n)
            bootdf = pgdf.irow(resampled_ix)
            pnogo = rangl_data(bootdf, data_style='pro')
            bootlist.append(bootdf)
            pvec.append(pnogo)
        rts = rt_quantiles(pd.concat(bootlist))
        return np.hstack([pnogo, rts])


def rwr(X, get_index=False, n=None):
    """
    Modified from http://nbviewer.ipython.org/gist/aflaxman/6871948
    """
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)
    resample_i = np.floor(np.random.rand(n) * len(X)).astype(int)
    X_resample = (X[resample_i])
    if get_index:
        return resample_i
    else:
        return X_resample


def proactive_mj_quanterr(df, split='HL', tb=.555, prob=array([.1, .3, .5, .7, .9]), as_ratio=True):
    """ calculates weight vectors for proactive RT quantiles by estimating
    first estimating the SEM of RT quantiles across levels of <split>
    (using Maritz-Jarrett estimatation: scipy.stats.mstats_extras.mjci).
    Then representing these variances as ratios
    e.g.
          QSEM = mjci(rtvectors)
          wts = median(QSEM)/QSEM
    """
    nquant = len(prob)
    godf = df[df.response == 1].copy()
    ncond = len(godf[split].unique())

    mjcix = lambda x: mjci(x.rt, prob=prob)
    q_sem_obj = godf.groupby(['idx', split]).apply(mjcix).values
    qwts = np.nanmean(np.vstack(q_sem_obj.reshape(2, 61).mean(axis=1)), axis=1)[:, None] / np.vstack(q_sem_obj.reshape(2, 61).mean(axis=1))
    return qwts


def reactive_mj_quanterr(df, avg_ssd=True, cond='Cond', prob=array([.1, .3, .5, .7, .9]), as_ratio=True):
    """ calculates weight vectors forreactive RT quantiles by estimating
    first estimating the SEM of RT quantiles for corr. and err. responses.
    (using Maritz-Jarrett estimatation: scipy.stats.mstats_extras.mjci).
    Then representing these variances as ratios.
    e.g.
          QSEM = mjci(rtvectors)
          wts = median(QSEM)/QSEM
    """
    ncond = len(df[cond].unique())
    nquant = len(prob)

    godf = df.query('response==1')
    # Martinz & Jarrett
    mjcix = lambda x: mjci(x.rt, prob=prob)
    ica_err = np.vstack(godf.groupby(['idx', cond, 'acc']).apply(mjcix).values)

    nidx = int(len(ica_err) / ncond / 2)
    # divide by 2 again because grouped by accuracy
    qx = ica_err.reshape(ncond, nidx, nquant, ncond)
    c_a_q = np.nanmean(qx, axis=1).reshape(ncond, 2, nquant)
    # axes: cond, acc, quants
    # get mean err across first two axes and divide by vals
    qwts = np.nanmean(c_a_q, axis=2)[:, :, None] / c_a_q
    return qwts


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


def get_observed_vector(rt, prob=array([10, 30, 50, 70, 90])):
    """ takes array of rt values and returns binned counts (trials
    that fall between each set of percentiles in prob). also returns
    the total number of observations (len(rt)) and the RT values at those
    percentiles (rtquant)
    """
    inter_prob = array([prob[0] - 0] + [prob[i] - prob[i - 1] for i in range(1, len(prob))] + [100 - prob[-1]])
    rtquant = mq(rt, prob=prob * .01)
    ocounts = np.ceil((inter_prob) * .01 * len(rt)).astype(int)
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


def make_proRT_conds(data, split=None, rt_cix=None):

    if np.any(data['pGo'].values > 1):
        data['pGo'] = data['pGo'] * .01
    if np.any(data['rt'].values > 5):
        data['rt'] = data['rt'] * .001

    if split != None:
        pg = split * .01
        data['HL'] = 'x'
        data.ix[data.pGo > pg, 'HL'] = 1
        data.ix[data.pGo <= pg, 'HL'] = 2

    # get index to split rts during fits
    rt_cix = len(data.query('HL==1').pGo.unique())
    return data, rt_cix


def get_obs_quant_counts(df, prob=([.10, .30, .50, .70, .90])):

    if type(df) == pd.Series:
        rt = df.copy()
    else:
        rt = df.rt.copy()

    inter_prob = [prob[0] - 0] + [prob[i] - prob[i - 1] for i in range(1, len(prob))] + [1.00 - prob[-1]]
    obs_quant = mq(rt, prob=prob)
    observed = np.ceil((inter_prob) * len(rt) * .94).astype(int)

    return observed, obs_quant


def get_header(params=None, data_style='re', labels=[], delays=[], prob=np.array([.1, .3, .5, .7, .9]), cond=None):

    info = ['nfev', 'nvary', 'df', 'chi',
            'rchi', 'logp', 'AIC', 'BIC', 'cnvrg']
    if data_style == 're':
        cq = ['c' + str(int(n * 100)) for n in prob]
        eq = ['e' + str(int(n * 100)) for n in prob]
        qp_cols = [cond, 'Go'] + delays + cq + eq
    else:
        hi = ['hi' + str(int(n * 100)) for n in prob]
        lo = ['lo' + str(int(n * 100)) for n in prob]
        qp_cols = labels + hi + lo

    if params is not None:
        infolabels = params + info
        return [qp_cols, infolabels]
    else:
        return [qp_cols]


def get_exp_counts(simdf, obs_quant, n_obs, prob=([.10, .30, .50, .70, .90])):

    if type(simdf) == pd.Series:
        simrt = simdf.copy()
    else:
        simrt = simdf.rt.copy()
    exp_quant = mq(simrt, prob)
    oq = obs_quant
    expected = np.ceil(np.diff([0] + [pscore(simrt, oq_rt) * .01 for oq_rt in oq] + [1]) * n_obs)

    return expected, exp_quant


def weight_by_simulated_variance(opt, p, nsims=200):

    from pandas import DataFrame as DF

    data_style = opt.fitparams['data_style']
    sims = [opt.simulator.sim_fx(p) for i in range(nsims)]
    x = DF(np.asarray(sims)).std().values
    x = x.round(4)
    if data_style == 're':
        x = x.reshape(opt.ncond, 16)
        x[:, :6] = np.median(x[:, :6], axis=1)[:, None] / x[:, :6]
        x[:, 6:] = np.median(x[:, 6:], axis=1)[:, None] / x[:, 6:]
    else:
        hi = np.mean(x[:3])
        lo = np.mean(x[3:6])
        x[:6] = np.median(x[:6]) / x[:6]
        x[6:11] = np.median(x[6:11]) / x[6:11] * hi
        x[11:] = np.median(x[11:]) / x[11:] * lo

    return x.flatten()


def get_intersection(iter1, iter2):
    intersect_set = set(iter1).intersection(set(iter2))
    return ([i for i in intersect_set])


def rename_bad_cols(data):

    if 'trial_type' in data.columns:
        data.rename(columns={'trial_type': 'ttype'}, inplace=True)

    return data
