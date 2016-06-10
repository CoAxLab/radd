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

def rangl_data(data, data_style='re', tb=.555, percentiles=([.1, .3, .5, .7, .9])):
    """ called by __make_dataframes__ to generate observed dataframes and iterables for
    subject fits
    """

    gac = data.query('ttype=="go"').acc.mean()
    sacc = data.query('ttype=="stop"').groupby('ssd').mean()['acc'].values
    grt = data.query('ttype=="go" & acc==1').rt.values
    ert = data.query('response==1 & acc==0').rt.values
    gq = mq(grt, prob=percentiles)
    eq = mq(ert, prob=percentiles)
    return np.hstack([gac, sacc, gq, eq])


def get_group_cost_weights(dfhandler, percentiles=np.array([.1, .3, .5, .7, .9]), nacc=2):
    """ calculate weights using observed variability
    across subjects (model.observedDF)
    """

    nsplits = dfhandler.nlevels * dfhandler.nconds
    nquant = percentiles.size
    quant_cols = np.asarray(percentiles*100).astype(int)
    conds = dfhandler.conds
    groups = dfhandler.groups
    data_cols = dfhandler.observedDF.loc[:, 'acc':].columns
    p_cols = data_cols[:-nquant*nacc]
    observedDF = dfhandler.observedDF.groupby(conds)


    # use martinz-jarett method to estimate quantile errs
    quantdf = mj_quanterr(df=dfhandler.data, groups=groups, percentiles=percentiles)
    perr = observedDF.agg(np.nanstd).loc[:, p_cols]
    counts = observedDF.count().loc[:, p_cols]
    p_wt_bycount = perr.values * (1./counts.values)
    pwts_ratio = np.nanmedian(p_wt_bycount, axis=1)[:, None] / p_wt_bycount

    groups = np.hstack([conds, 'ttype', 'acc']).tolist()
    avg_qerr = np.hstack(quantdf.groupby(groups).mean().values).reshape(nsplits, nquant*2)

    avg_wts = np.array([np.append(pw, qw) for pw, qw in zip(pwts_ratio, avg_qerr)])
    ndata = int(avg_wts.size / nsplits)
    flat_wts = avg_wts.reshape(nsplits, ndata).mean(axis=0)
    avg_wts, flat_wts = ensure_numerical_wts(avg_wts, flat_wts)

    return [avg_wts], [flat_wts]


def get_subject_cost_weights(dfhandler, weight_presponse=True, percentiles=np.array([.1, .3, .5, .7, .9]), nacc=2):
    """ calculate weights using observed variability
    within individual subjects
    """
    data = dfhandler.data
    groups = dfhandler.groups
    dfvals = dfhandler.dfvals
    nlevels = dfhandler.nlevels
    nconds = dfhandler.nconds
    nsplits = nlevels * nconds
    nidx = dfhandler.data.idx.unique().size
    nrows = nidx * nconds * nlevels
    nquant = percentiles.size
    quant_cols = np.asarray(percentiles*100).astype(int)

    # use martinz-jarett method to estimate quantile errs
    quantdf = mj_quanterr(df=data, groups=groups, percentiles = percentiles)
    idx_qwts = quantdf.loc[:, quant_cols].values.reshape(nidx*nacc, nsplits*nquant)

    if weight_presponse:
        idx_pwts = get_count_pwts(data, var='ssd', groups=groups)
    else:
        idx_pwts = [np.ones(idat.size - nacc*nquant) for idat in dfvals]

    cost_wts = np.array([np.append(pw, qw) for pw, qw in zip(idx_pwts, idx_qwts)])
    cost_wts = [cost_wts[i:i+nsplits] for i in range(0, nrows, nsplits)]
    flat_cost_wts = [idx_cwts.mean(axis=0) for idx_cwts in cost_wts]
    return cost_wts, flat_cost_wts


def get_count_pwts(df, var='ssd', groups=['idx']):
    """ count number of observed responses across levels of <var> and transform into
    ratios (np.median(counts_at_each_level) / counts_at_each_level) for weight subject-level
    p(response) values in cost function.

    ::Arguments::
        df (DataFrame): group-level dataframe
        var (str): column header for variable to count responses
        conds (list): depends_on.values()
    """

    if var=='ssd':
        df = df[df.ttype=='stop'].copy()

    def calc_pwts(df_i):
        countdf = df_i.groupby(var).count()
        # get counts across levels except for last (correct Go//SSD==1000)
        lvl_counts = countdf.reset_index()['response'].values
        # calculate count ratio for all values
        varwts = np.median(lvl_counts)/lvl_counts
        # append 1 to front as weight of correct P(Go|No_SS)
        acc_var_wts = np.append(1., varwts)
        return acc_var_wts

    ic_grp = df.groupby(groups)
    idx_pwts = [calc_pwts(cdf) for c, cdf in ic_grp]
    return idx_pwts


def mj_quanterr(df, groups=['idx'], percentiles=array([.1, .3, .5, .7, .9]), as_ratio=True, nacc=2):
    """ calculates weight vectors for reactive RT quantiles by
    first estimating the SEM of RT quantiles for corr. and err. responses.
    (using Maritz-Jarrett estimatation: scipy.stats.mstats_extras.mjci).
    Then representing these variances as ratios.
    e.g.
          QSEM = mjci(rtvectors)
          wts = median(QSEM)/QSEM
    """

    # Martinz & Jarrett
    mjcix = lambda x: mjci(x.rt, prob=percentiles)
    if len(groups)>1:
        conds = groups[1:]
        nlevels = np.sum([df[c].unique().size for c in conds])
    else:
        nlevels = 1
    nidx = df.idx.unique().size
    nquant = percentiles.size

    # sort by ttype first to ensure go(acc==1) occurs before stop(acc==0)
    groups = np.hstack([groups, 'ttype', 'acc']).tolist()
    quant_cols = np.asarray(percentiles*100).astype(int)
    nrows = nidx * nlevels * nacc

    godf = df.query('response==1')
    godf_grouped = godf.groupby(groups)
    categories = array([list(x) for x, xdf in godf_grouped])

    # calculate quantile errors
    quant_err = np.vstack(godf_grouped.apply(mjcix).values)
    # reshape [nidx   x   ncond * nquant * 2]
    idx_qerr = quant_err.reshape(nidx, nlevels * nquant * nacc)

    # calculate subject median across all conditions quantiles and accuracy
    # this implicitly accounts for n_obs as the mjci estimate of sem will be lower
    # for conditions with greater number of observations (i.e., more acc==1 in godf)
    idx_medians = np.nanmedian(idx_qerr, axis=1)

    # tile subject medians (nlevels*2)
    idx_tiled_medians = [np.tile(idxmed, nlevels * nacc) for idxmed in idx_medians]
    idx_tiled_medians = np.vstack(np.hstack(idx_tiled_medians))

    # divide idx median err value by quant_err vector
    idx_qerr_ratio = idx_tiled_medians / quant_err

    catdf = pd.DataFrame(categories, columns=groups, index=np.arange(nrows))
    errdf = pd.DataFrame(idx_qerr_ratio, columns=quant_cols, index=np.arange(nrows))
    quant_df = pd.concat([catdf, errdf], axis=1)

    quant_df['acc'] = pd.to_numeric(quant_df['acc'])
    quant_df_filled = quant_df.fillna({col: np.nanmean(quant_df[col]) for col in quant_cols})
    return quant_df_filled



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


def get_observed_vector(rt, percentiles=array([10, 30, 50, 70, 90])):
    """ takes array of rt values and returns binned counts (trials
    that fall between each set of percentiles in percentiles). also returns
    the total number of observations (len(rt)) and the RT values at those
    percentiles (rtquant)
    """
    inter_percentiles = array([percentiles[0] - 0] + [percentiles[i] - percentiles[i - 1] for i in range(1, len(percentiles))] + [100 - percentiles[-1]])
    rtquant = mq(rt, prob=percentiles * .01)
    ocounts = np.ceil((inter_percentiles) * .01 * len(rt)).astype(int)
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


def get_obs_quant_counts(df, percentiles=([.10, .30, .50, .70, .90])):

    if type(df) == pd.Series:
        rt = df.copy()
    else:
        rt = df.rt.copy()

    inter_percentiles = [percentiles[0] - 0] + [percentiles[i] - percentiles[i - 1] for i in range(1, len(percentiles))] + [1.00 - percentiles[-1]]
    obs_quant = mq(rt, prob=percentiles)
    observed = np.ceil((inter_percentiles) * len(rt) * .94).astype(int)
    return observed, obs_quant


def get_exp_counts(simdf, obs_quant, n_obs, percentiles=([.10, .30, .50, .70, .90])):

    if type(simdf) == pd.Series:
        simrt = simdf.copy()
    else:
        simrt = simdf.rt.copy()
    exp_quant = mq(simrt, prob=percentiles)
    oq = obs_quant
    expected = np.ceil(np.diff([0] + [pscore(simrt, oq_rt) * .01 for oq_rt in oq] + [1]) * n_obs)
    return expected, exp_quant


def get_intersection(iter1, iter2):
    """ get the intersection of two iterables ("items in-common")
    """

    intersect_set = set(iter1).intersection(set(iter2))
    return ([i for i in intersect_set])
