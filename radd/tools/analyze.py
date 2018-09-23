#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
import os
from numpy import array
from future.utils import listvalues
from sklearn.neighbors import KernelDensity
from scipy.stats.mstats import mquantiles as mq
from scipy.stats.mstats_extras import mjci
from scipy import optimize
import functools
from scipy.interpolate import interp1d
from scipy import stats
from scipy.optimize import leastsq, minimize, curve_fit


def ezdiff(rt, correct, s=1.):

    """ estimate ezDDM parameters
    """

    logit = lambda p:np.log(p/(1-p))
    pc = np.mean(correct)

    # subtract or add 1/2 an error to prevent division by zero
    if pc==1.0:
        pc=1 - 1/(2*len(correct))
    if pc==0.5:
        pc=0.5 + 1/(2*len(correct))
    MRT = np.mean(rt[correct==1])
    VRT = np.var(rt[correct==1])

    r = (logit(pc)*(((pc**2) * logit(pc)) - pc*logit(pc) + pc - 0.5))/VRT
    v = np.sign(pc-0.5)*s*(r)**0.25
    a = (s**2 * logit(pc))/v
    y = (-1*v*a)/(s**2)
    MDT = (a/(2*v))*((1-np.exp(y))/(1+np.exp(y)))
    tr = MRT-MDT

    return([a, v, tr])



def blockify_trials(data, nblocks=5, conds=None, groups=['idx']):

    datadf = data.copy()
    if conds is not None:
        if type(conds) is str:
            conds = [conds]
        groups = groups + conds

    idxdflist = []
    for dfinfo, idxdf in datadf.groupby(groups):
        ixblocks = np.array_split(idxdf.trial.values, nblocks)
        blocks = np.hstack([[i+1]*arr.size for i, arr in enumerate(ixblocks)])
        idxdf = idxdf.copy()
        colname = 'block{}'.format(nblocks)
        idxdf[colname] = blocks
        idxdflist.append(idxdf)

    return pd.concat(idxdflist)



def rangl_data(data, ssd_method='all', quantiles=np.linspace(.01,.99,15), fit_on='average'):
    """ called by DataHandler.__make_dataframes__ to generate
    observed data arrays
    """
    data = data.copy()
    if 1 in data.ttype.unique():
        goQuery = 'ttype==1.'
        stopQuery = 'ttype==0.'
    else:
        goQuery = 'ttype=="go"'
        stopQuery = 'ttype=="stop"'
    gac = data.query(goQuery).acc.mean()
    grt = data.query('response==1 & acc==1').rt.values
    ert = data.query('response==1 & acc==0').rt.values
    gq = mq(grt[grt<5.], prob=quantiles)
    eq = mq(ert[ert<5.], prob=quantiles)
    data_vector = [gac, gq, eq]
    if 'ssd' in data.columns:
        stopdf = data.query(stopQuery)
        if 'probe' in stopdf.columns:
            stopdf = stopdf[stopdf.probe==1]
        if ssd_method=='all':
            sacc=stopdf.groupby('ssd').mean()['acc'].values
        elif ssd_method=='central':
            sacc = np.array([stopdf.mean()['acc']])
        data_vector.insert(1, sacc)
    return np.hstack(data_vector)


def rangl_freq(df, quantiles=np.arange(.1, 1.,.2)):

    prob = np.asarray([0.] + quantiles.tolist() + [1.])
    nbins = quantiles.size+1

    godf = df[df.ttype=='go']
    ssdf = df[df.ttype=='stop']
    cdf = godf[godf.response==1]
    edf = ssdf[ssdf.response==1]

    qcor = mq(cdf.rt.values, prob)
    qerr = mq(edf.rt.values, prob)

    Ncor = cdf.shape[0]
    Nerr = edf.shape[0]
    prop_go = np.asarray([cdf[(cdf.rt >= qcor[i])&(cdf.rt < qcor[i+1])].shape[0] / Ncor for i in range(nbins)])
    prop_ss = np.asarray([edf[(edf.rt >= qerr[i])&(edf.rt < qerr[i+1])].shape[0] / Nerr for i in range(nbins)])

    O_cor = Ncor * prop_go
    O_err = Nerr * prop_ss
    return np.concatenate([O_cor, O_err])


def rangl_counts(df):
    Ncor = df[(df.ttype=='go')&(df.response==1)].shape[0]
    Nerr = df[(df.ttype=='stop')&(df.response==1)].shape[0]
    return np.hstack([Ncor, Nerr])


def calc_chi_square(rt, ssrt, fitparams):
    O_freq = fitparams['obs_freq']
    Nobs = fitparams['nobs']
    Ncor, Nerr = Nobs
    quantiles = fitparams['quantiles']
    ntrials = fitparams['ntrials']
    tb = fitparams['tb']
    y = fitparams['y']
    # rt, ssrt = m.sim.simulate_model(m.inits, analyze=False, get_rts=True)

    nbins = quantiles.size+1
    ssrt = ssrt.reshape(nl, -1)
    ert = rt[:, :int(ntrials/2)]

    Ncor_hat = np.asarray([rt_i[rt_i<tb].shape[0] for rt_i in rt])
    Nerr_hat= np.asarray([ert_i[ert_i<ssrt_i].shape[0] for ert_i, ssrt_i in zip(ert, ssrt)])

    qcor, qerr = np.split(y[-quantiles.size*2:], 2)
    qcor = np.hstack([[0.], qcor, [tb]])
    qerr = np.hstack([[0.], qerr, [tb]])

    prop_go = [[rti[(rti >= qcor[i])&(rti < qcor[i+1])].shape[0] / Ni for i in range(nbins)] for Ni, rti in zip(Ncor_hat, rt)]
    prop_ss = [[erti[(erti >= qerr[i])&(erti < qerr[i+1])].shape[0] / Ni for i in range(nbins)] for Ni, erti in zip(Nerr_hat, ert)]

    # Ncor, Nerr are actual counts of O_cor, O_err
    M_cor = [Ni * np.asarray(prop_go_i) for Ni, prop_ss_i in zip(Nerr, prop_go)]
    M_err = [Ni * np.asarray(prop_ss_i) for Ni, prop_ss_i in zip(Nerr, prop_ss)]
    M_freq = [np.concatenate([Mcor_i, Merr_i]) for Mcor_i, Merr_i in zip(M_cor, M_err)]

    chi2 = np.sum([np.sum((M_i - O_i)**2 / O_i) for M_i, O_i in zip(M_freq, O_freq)])

    return chi2


def calculate_qpdata(data, quantiles=np.linspace(.02,.98,10)):
    data = data[data.response==1].copy()
    cor = data[data['acc']==1]
    err = data[data['acc']==0]
    idx_qc = pd.DataFrame([mq(idxdf.rt.values, quantiles) for idx, idxdf in cor.groupby('idx')])
    idx_qe = pd.DataFrame([mq(idxdf.rt.values, quantiles) for idx, idxdf in err.groupby('idx')])
    quantMeans = np.vstack = ([idx_qc.mean(axis=0).values, idx_qe.mean(axis=0).values])
    quantErr = np.vstack([idx_qc.sem(axis=0).values, idx_qe.sem(axis=0).values])*1.96
    return quantMeans, quantErr


def idx_quant_weights_OLD(df, groups=[], nsplits=1, prob=np.arange(.1,1.,.1), max_wt=2.5):

    """ calculates weight vectors for reactive RT quantiles by
    first estimating the SEM of RT quantiles for corr. and err. responses.
    (using Maritz-Jarrett estimatation: scipy.stats.mstats_extras.mjci).
    Then representing these variances as ratios.
    e.g.
          QSEM = mjci(rtvectors)
          wts = median(QSEM)/QSEM
    """
    idx_mjci = lambda x: mjci(x.rt, prob=prob)

    nquant = prob.size

    if 'idxN' in df.columns:
        nidx = df.idxN.unique().size
    else:
        nidx = df.idx.unique().size

    # get all trials with response recorded
    godf = df.query('response==1')

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
    return idx_qratio.reshape(nsplits * nidx, nquant * 2)


def idx_quant_weights(data, conds, quantiles=np.linspace(.02,.98,10), max_wt=3.5, bwfactors=None):
    """ calculates weight vectors for reactive RT quantiles by
    first estimating the SEM of RT quantiles for corr. and err. responses.
    (using Maritz-Jarrett estimatation: scipy.stats.mstats_extras.mjci).
    Then representing these variances as ratios.
    e.g.
          QSEM = mjci(rtvectors)
          wts = median(QSEM)/QSEM
    """
    idx_mjci = lambda x: mjci(x.rt, prob=quantiles)

    if bwfactors is not None:
        if not isinstance(bwfactors, list):
            bwfactors = [bwfactors]
        conds = [c for c in conds if c not in bwfactors]

    if len(conds)<1:
        nlevels = 1
    else:
        # get all trials with response recorded
        nlevels = np.sum([data[c].unique().size for c in conds])

    godf = data.query('response==1')
    # sort by ttype first so go(acc==1) occurs before stop(acc==0)
    ttype_ordered_groups = np.hstack([conds, 'ttype', 'acc']).tolist()
    godf_grouped = godf.groupby(ttype_ordered_groups)
    # apply self.idx_mjci() to estimate quantile CI's
    quant_err = np.hstack(godf_grouped.apply(idx_mjci).values)
    # calculate subject median across all conditions quantiles and accuracy
    # this implicitly accounts for n_obs as the mjci estimate of sem will be lower
    # for conditions with greater number of observations (i.e., more acc==1 in godf)
    idx_median = np.median(quant_err)
    # idx_qratio = idx_median / quant_err
    idx_qratio = idx_median / quant_err
    # set extreme values to max_wt arg val
    idx_qratio[idx_qratio >= max_wt] = max_wt

    return idx_qratio.reshape(nlevels, -1)


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
            split_by = 'ssd'
            df = df[df.ttype=='stop'].copy()
            if 'probe' in df.columns:
                df = df[df.probe==1].copy()
        else:
            split_by = 'ttype'
    else:
        split_by = conds
        _ = index.remove(split_by)
    df['n'] = 1
    countdf = df.pivot_table('n', index=index, columns=split_by, aggfunc=np.sum)
    idx_pwts = countdf.values / countdf.median(axis=1).values[:, None]
    idx_pwts = np.ones_like(idx_pwts) #+ .5
    # idx_pwts = countdf.values / countdf.median(axis=0).values
    if ssd_method=='all':
        go_wts = np.ones(countdf.shape[0]) #* .5
        # idx_pwts[:, -5] = .5
        # idx_pwts[:, -4] = .5
        # idx_pwts[:, -3] = .5
        # idx_pwts[:, -2] = .5
        # idx_pwts[:, -1] = .5

        idx_pwts = np.concatenate((go_wts[:,None], idx_pwts), axis=1)

    return idx_pwts


def assess_fit(y, wts, yhat, nvary=5):
    # residual = wts * (yhat - y)
    # fp['yhat'] = (residual / wts) + y
    # fill finfo dict with goodness-of-fit info
    y = y.flatten()
    wts = wts.flatten()
    yhat = yhat.flatten()
    finfo = pd.Series()
    finfo['nvary'] = nvary
    finfo['chi'] = np.sum((wts*(yhat - y))**2)
    finfo['ndata'] = len(yhat)
    finfo['df'] = finfo.ndata - finfo.nvary
    finfo['rchi'] = finfo.chi / finfo.df
    finfo['logp'] = finfo.ndata * np.log(finfo.rchi)
    finfo['AIC'] = finfo.logp + 2 * finfo.nvary
    finfo['BIC'] = finfo.logp + np.log(finfo.ndata * finfo.nvary)
    return finfo


def pandaify_results(gort, ssrt, tb=.7, bootstrap=False, bootinfo={'nsubjects':25, 'ntrials':1000, 'groups':['ssd']}, ssd=np.array([[.2, .25, .3, .35, .4]]), clmap=None):
    nl, nssd, nssPer = ssrt.shape
    nl, ntrials = gort.shape
    nss = nssd * nssPer
    dfColumns=['cond', 'ttype', 'ssd', 'response', 'acc', 'rt', 'ssrt', 'trial']
    levelNames = np.hstack(listvalues(clmap)).tolist()
    # bootinfo['groups'] = bootinfo['groups'] + list(clmap)
    dfIndex = np.arange(ntrials)
    dfList = []
    for i in range(nl):
        ert = gort[i, :nss].reshape(ssrt[i].shape)
        goTrialOutcomes = np.hstack(np.where(gort[i, nss:] < tb, 1, 0))
        # ssTrialOutcomes = np.hstack(np.hstack(np.where(ert <= ssrt[i], 1, 0)))
        ssTrialOutcomes = np.hstack(np.hstack(np.where(ssrt[i] <= ert, 0, 1)))
        delays = np.append(np.hstack([[delay]*nssPer for delay in ssd[i]]), [1000]*nss)
        response = np.append(ssTrialOutcomes, goTrialOutcomes)
        responseTime = gort[i]
        ssResponseTime = np.append(np.hstack(np.hstack(ssrt[i])), [np.nan]*nss)
        ttype = np.append(np.zeros(nss), np.ones(nss))
        acc = np.where(ttype == response, 1, 0)
        cond = [levelNames[i]]*ntrials
        trial = np.arange(1, ntrials+1)
        dfData = [cond, ttype, delays, response, acc, responseTime, ssResponseTime, trial]
        df = pd.DataFrame(dict(zip(dfColumns, dfData)), index=dfIndex)
        dfList.append(df[dfColumns])
    df = pd.concat(dfList)
    df.reset_index(drop=True, inplace=True)
    # stopdf = resultsdf[resultsdf.ttype==0.]
    df.loc[(df.rt==1000.)&(df.ssrt==1000.)&(df.ttype==0.), 'response'] = 0
    df.loc[(df.rt==1000.)&(df.ssrt==1000.)&(df.ttype==0.), 'acc'] = 1

    if bootstrap:
        resultsdf = bootstrap_data(resultsdf, nsubjects=bootinfo['nsubjects'], n=bootinfo['ntrials'], groups=bootinfo['groups'])
    return resultsdf


def bootstrap_data(data, nsubjects=25, n=120, groups=['cond', 'ssd']):
    """ generates n resampled datasets using rwr()
    for bootstrapping model fits
    """
    df = data.copy()
    bootlist = []
    idxList = []
    if 'idx' in df.columns:
        df = df.drop('idx', axis=1)
    if n == None:
        n = len(df)
    # bootdfList = []
    # for level, levelDF in df.groupby(groups[0]):
    bootlist = []
    for idx in range(nsubjects):
        boots = df.reset_index(drop=True)
        orig_ix = np.asarray(boots.index[:])
        resampled_ix = rwr(orig_ix, get_index=True, n=n)
        bootdf = df.iloc[resampled_ix]
        bootdf = bootdf.copy()
        bootdf.insert(0, 'idx', idx)
        bootdf['trial'] = np.arange(bootdf.shape[0])
        bootlist.append(bootdf)
    # concatenate and return all resampled conditions
    bootdf = pd.concat(bootlist)
    bootdf.reset_index(drop=True, inplace=True)
    # bootdfList.append(bootdf)
    # bootdf = pd.concat(bootdfList)
    return bootdf


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


def estimate_timeboundary(data):
    goDF = data[data.response==1].copy()
    return np.int(np.round(goDF.rt.max(), 2)*1000) * .001


def determine_ssd_method(stopdf):
    ssd_n = [df.size for _, df in stopdf.groupby('ssd')]
    # test if equal # of trials per ssd & return ssd_n
    all_equal_counts = ssd_n[1:] == ssd_n[:-1]
    if all_equal_counts:
        return 'all'
    else:
        return'central'


def get_model_ssds(stopdf, conds, ssd_method='all', scale=.001, bwfactors=None):

    if bwfactors is not None:
        if not isinstance(bwfactors, list):
            bwfactors = [bwfactors]
        conds = [c for c in conds if c not in bwfactors]

    groups = conds + ['idx']
    if ssd_method == 'all':
        get_idx_ssd = lambda x: pd.Series(np.sort(x.ssd.unique().squeeze())) * scale
        ssdidx = stopdf.groupby(groups).apply(get_idx_ssd)
        ssdDF = pd.DataFrame(ssdidx).reset_index()
    elif ssd_method == 'central':
        get_idx_ssd = lambda x: x.ssd.mean() * scale
        ssdidx = stopdf.groupby(groups).apply(get_idx_ssd)
        ssdDF = ssdidx.reset_index()

    return ssdDF


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
    sortby = 'idx'
    if 'trial' in clean.columns:
        sortby = ['idx', 'trial']
    cleandf = clean.sort_values(by=sortby)
    return cleandf


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
    xsim = np.linspace(x[0], x[-1], 50, endpoint=True)
    ysim = interpol_fx(xsim)
    return xsim, ysim


def scurve_poly_fit(x, y, n=20):
    polysim = lambda p, pi, x, xi: p[pi]*x**xi
    ix = np.arange(n+1)
    poly_ix = zip(ix, ix[::-1])
    p = np.polyfit(x,y,n)
    xsim = np.linspace(x.min()-.05, x.max()+.08, 2500, endpoint=True)
    ysim = np.sum([polysim(p, pi, xsim, xi) for pi, xi in poly_ix],axis=0)
    return xsim, ysim


def fit_sigmoid(x, y):
    #x, y = scurve_interpolate(x, y)
    initParams = [np.mean(x), .8, .5]
    results = minimize(sigmoid, initParams, args=(x, y), method='Nelder-Mead')
    popt = results.x
    xSim = np.linspace(x.min()-.01, x.max()+.01, 1000, endpoint=True)
    ySim = 1. / (1. + np.exp(popt[0]*(xSim-popt[1])))
    return xSim, ySim


def sigmoid(params, xdata, ydata):
    k = params[0]
    x0 = params[1]
    sd = params[2]
    yPred = 1. / (1. + np.exp(k*(xdata-x0)))
    # Calculate negative log likelihood
    LL = -np.sum(stats.norm.logpdf(ydata, loc=yPred, scale=sd ))
    return(LL)


def logistic4(x, a, b, c, d):
    """4PL lgoistic equation."""
    return ((a-d)/(1.0+((x/c)**b))) + d


def logistic_residuals(p, y, x):
    """Deviations of data from fitted 4PL curve"""
    a,b,c,d = p
    err = y-logistic4(x, a, b, c, d)
    return err


def logistic_peval(x, p):
    """Evaluated value at x with current parameters."""
    a,b,c,d = p
    return logistic4(x, a, b, c, d)


def fit_logistic(x, y):
    # Initial guess for parameters
    p0 = [0., .5, 1, 1]
    popt = leastsq(logistic_residuals, p0, args=(y, x))
    xSim = np.linspace(x.min()-.1, x.max()+.1, 1000, endpoint=True)
    ySim = logistic_peval(xSim, popt[0])
    return xSim, ySim


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

def get_intersection(iter1, iter2):
    """ get the intersection of two iterables ("items in-common")
    """
    intersect_set = set(iter1).intersection(set(iter2))
    return ([i for i in intersect_set])
