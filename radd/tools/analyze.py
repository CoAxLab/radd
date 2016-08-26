#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
import os
from numpy import array
from sklearn.neighbors import KernelDensity
from scipy.stats.mstats import mquantiles as mq
from scipy.stats.mstats_extras import mjci
from scipy import optimize
import functools
from scipy.interpolate import interp1d

def rangl_data(data, quantiles=np.arange(.1, 1.,.1), ssd_method='all'):
    """ called by DataHandler.__make_dataframes__ to generate
    observed data arrays
    """
    data = data.copy()
    gac = data.query('ttype=="go"').acc.mean()
    grt = data.query('response==1 & acc==1').rt.values
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


def idx_quant_weights(data, groups, nlevels=1, quantiles=np.arange(.1, 1.,.1), max_wt=3):
    """ calculates weight vectors for reactive RT quantiles by
    first estimating the SEM of RT quantiles for corr. and err. responses.
    (using Maritz-Jarrett estimatation: scipy.stats.mstats_extras.mjci).
    Then representing these variances as ratios.
    e.g.
          QSEM = mjci(rtvectors)
          wts = median(QSEM)/QSEM
    """
    idx_mjci = lambda x: mjci(x.rt, prob=quantiles)
    nidx = data.idx.unique().size
    nquant = quantiles.size
    nsplits = nlevels
    # get all trials with response recorded
    godf = data.query('response==1')
    # sort by ttype first so go(acc==1) occurs before stop(acc==0)
    ttype_ordered_groups = np.hstack([groups, 'ttype', 'acc']).tolist()
    godf_grouped = godf.groupby(ttype_ordered_groups)
    # apply self.idx_mjci() to estimate quantile CI's
    quant_err = np.vstack(godf_grouped.apply(idx_mjci).values)
    # reshape [nidx   x   ncond * nquant * nacc]
    idx_qerr = quant_err.reshape(nidx, nsplits * nquant * 2)
    # calculate subject median across all conditions quantiles and accuracy
    # this implicitly accounts for n_obs as the mjci estimate of sem will be lower
    # for conditions with greater number of observations (i.e., more acc==1 in godf)
    idx_medians = np.nanmedian(idx_qerr, axis=1)
    idx_qratio = idx_medians[:, None] / idx_qerr
    # set extreme values to max_wt arg val
    idx_qratio[idx_qratio >= max_wt] = max_wt
    # reshape to fit in wtsDF[:, q_cols]
    return idx_qratio.reshape(nidx * nsplits, nquant * 2)


def idx_acc_weights(data, conds=['flat'], ssd_method='all'):
    """ count number of observed responses across levels, transform into ratios
    (counts_at_each_level / np.median(counts_at_each_level)) for weight
    subject-level p(response) values in cost function.
    ::Arguments::
        df (DataFrame): group-level dataframe
        var (str): column header for variable to count responses
        conds (list): depends_on.values()
    """
    df = data.copy()
    index=['idx']
    if not 'flat' in conds:
        index = index + conds
    if 'ssd' in df.columns:
        if ssd_method=='all':
            df = df[df.ttype=='stop'].copy()
            split_by = 'ssd'
        else:
            split_by = 'ttype'
    else:
        split_by = conds
        _ = index.remove(split_by)
    df['n'] = 1
    countdf = df.pivot_table('n', index=index, columns=split_by, aggfunc=np.sum)
    idx_pwts = countdf.values / countdf.median(axis=1).values[:, None]
    if ssd_method=='all':
        go_wts = np.ones(countdf.shape[0])
        idx_pwts = np.concatenate((go_wts[:,None], idx_pwts), axis=1)
    return idx_pwts


def determine_ssd_method(stopdf):
    ssd_n = [df.size for _, df in stopdf.groupby('ssd')]
    # test if equal # of trials per ssd & return ssd_n
    all_equal_counts = ssd_n[1:] == ssd_n[:-1]
    if all_equal_counts:
        return 'all'
    else:
        return'central'


def get_model_ssds(stopdf, conds, ssd_method='all', scale=.001):
    if ssd_method == 'all':
        get_df_ssds = lambda df: df.groupby(conds).ssd.unique().values
        cond_ssds =  [get_df_ssds(df) for _,df in stopdf.groupby('idx')]
    elif ssd_method == 'central':
        mean_cond_ssd_df = stopdf.pivot_table('ssd', index='idx', columns=conds)
        cond_ssds = list(mean_cond_ssd_df.values)
    ssds = [np.sort(np.vstack(ssds))*scale for ssds in cond_ssds]
    return ssds

def load_nested_popt_dictionaries(fitdf, params={}, nlevels=1):
    mparams = list(params)
    params_dict = {}
    for mid in fitdf.idx.values:
        freeparams = mid.split('_')
        constants = [p for p in mparams if p not in freeparams]
        select_params = fitdf[fitdf.idx==mid][mparams]
        pdict = select_params.to_dict('records')[0]
        for p in constants:
            pdict[p] = pdict[p] * np.ones(nlevels)
        params_dict[mid] = pdict
    return params_dict


def fill_nan_vals(coldata):
    coldata = coldata.copy()
    coldata[coldata.isnull()] = np.nanmean(coldata.values)
    return coldata


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

def scurve_interpolate(x, y, kind='cubic'):
    interpol_fx = interp1d(x, y, kind=kind)
    xsim = np.linspace(x[0], x[-1], 10000, endpoint=True)
    ysim = interpol_fx(xsim)
    return xsim, ysim

def scurve_poly_fit(x, y, n=20):
    polysim = lambda p, pi, x, xi: p[pi]*x**xi
    ix = np.arange(n+1)
    poly_ix = zip(ix, ix[::-1])
    p = np.polyfit(x,y,n)
    xsim = np.linspace(x.min(), x.max(), 10000, endpoint=True)
    ysim = np.sum([polysim(p, pi, xsim, xi) for pi, xi in poly_ix],axis=0)
    return xsim, ysim

def fit_sigmoid(x, y):
    x = resize(x, lower=x.max(), upper=x.min())
    y = resize(y, lower=y.min(), upper=y.max())
    p0 = (np.mean(x), np.mean(y), .5, .5)
    p, pcov = optimize.leastsq(residuals, p0, args=(x, y), xtol=1.e-15, ftol=1.e-15, maxfev=20000)
    x0, y0, c, k = p
    xsim = np.linspace(x.min()-20, x.max()+20, 10000, endpoint=True)
    ysim = sigmoid(p, xsim)
    return xsim, ysim

def sigmoid(p, x):
    x0, y0, c, k = p
    y = c / (1 + np.exp(k * (x - x0))) + y0
    return y

def residuals(p, x, y):
    return y - sigmoid(p, x)

def resize(arr, lower=0.0, upper=1.0):
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


def pwts_group_error_calc(handler):
    """ get stdev across subjects (and any conds) in observedDF
    weight perr by inverse of counts for each resp. probability measure
    previously a bound method of dfhandler --> DataHandler class
    """
    groupedDF = handler.observedDF.groupby(handler.conds)
    perr = groupedDF.agg(np.nanstd).loc[:, handler.p_cols].values
    counts = groupedDF.count().loc[:, handler.p_cols].values
    nsplits = handler.nlevels * handler.nconds
    ndata = len(handler.p_cols)
    # replace stdev of 0 with next smallest value in vector
    perr[perr==0.] = perr[perr>0.].min()
    p_wt_bycount = perr * (1./counts)
    # set wts equal to ratio --> median_perr / all_perr_values
    pwts_ratio = np.nanmedian(p_wt_bycount, axis=1)[:, None] / p_wt_bycount
    # set extreme values to max_wt arg val
    pwts_ratio[pwts_ratio >= handler.max_wt] = handler.max_wt
    # shape pwts_ratio to conform to wtsDF
    idx_pwts = np.array([pwts_ratio]*handler.nidx)
    return idx_pwts.reshape(handler.nidx * nsplits, ndata)



def get_intersection(iter1, iter2):
    """ get the intersection of two iterables ("items in-common")
    """
    intersect_set = set(iter1).intersection(set(iter2))
    return ([i for i in intersect_set])
