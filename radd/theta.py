#!usr/bin/env python
from __future__ import division
from future.utils import listvalues
import pandas as pd
import numpy as np
from copy import deepcopy
from numpy import array
from scipy.stats.distributions import norm, gamma, uniform
from lmfit import Parameters as lmParameters


def init_distributions(pkey, kind='dpm', mu=None, sigma=None, nrvs=25, tb=.65, force_normal=False):
    """ sample random parameter sets to explore global minima (called by
    Optimizer method __hop_around__())
    """
    if mu is None:
        mu = {'a': .25, 'tr': .3, 'v': .5, 'ssv': -.5, 'sso': .05, 'z': .1, 'xb': .5,
            'vi': .3, 'vd': .5, 'C': .001, 'B':.1, 'R': .0005, 'si':.001}
    if sigma is None:
        sigma = {'a': .2, 'tr': .15, 'v': .65, 'ssv': .5, 'sso': .035, 'z': .05, 'xb': 1., 'vi': .4, 'vd': .5, 'C':.08, 'B':.4, 'R': .008, 'si':.1}

    normal_params = ['a', 'tr', 'v', 'vd', 'ssv', 'z', 'sso', 'Beta']
    uniform_params = ['vi', 'xb', 'C', 'B', 'R', 'si']

    if 'race' in kind:
        sigma['ssv'] = abs(mu['ssv'])

    bounds = get_bounds(kind=kind)[pkey]
    loc = mu[pkey]
    scale = sigma[pkey]

    # init and freeze dist shape
    if pkey in normal_params:
        dist = norm(loc, scale)
    # elif pkey in gamma_params:
    #     dist = gamma(1.0, loc, scale)
    elif pkey in uniform_params:
        dist = uniform(loc, scale)

    # generate random variates
    rvinits = dist.rvs(nrvs)
    while rvinits.min() < bounds[0]:
        # apply lower limit
        ix = rvinits.argmin()
        rvinits[ix] = dist.rvs()
    while rvinits.max() > bounds[1]:
        # apply upper limit
        ix = rvinits.argmax()
        rvinits[ix] = dist.rvs()
    if pkey =='tr':
        rvinits = np.abs(rvinits)
    rvinits[rvinits<bounds[0]] = bounds[0]
    rvinits[rvinits>bounds[1]] = bounds[1]
    return rvinits


def get_bounds(kind='dpm', a=(.1, .65), tr=(.2, .6), v=(.1, 2.), ssv=(-1.5, -.1), sso=(.005, .1), xb=(.5, 2.), si=(.001, .15), z=(.01, .9), vd=(.1, 2.1), vi=(.1, 1.), Beta = (0.5, 5.), R=(.0001, .008), B=(.1, .4), C=(.001, .08), tb=None):
    """ set and return boundaries to limit search space
    of parameter optimization in <optimize_theta>
    """
    if tb is not None:
        tr = (tr[0], tb-0.1)
    if 'irace' in kind:
        ssv = (abs(ssv[1]), abs(ssv[0]))
    bounds = {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'vd':vd, 'vi':vi, 'z': z,
        'xb': xb, 'si': si, 'sso': sso, 'C': C, 'B': B, 'R': R, 'Beta': Beta}
    return bounds


def random_inits(pkeys, ninits=1, kind='dpm', mu=None, sigma=None, as_list=False, force_normal=False):
    """ random parameter values for initiating model across range of
    parameter values.
    ::Arguments::
        pkeys (list/dict):
            list of parameter names to sample, makes list of dict keys
            if pkeys is dictionary type
    ::Returns::
        params (dict):
            dictionary with pkeys as keys with values as 1d arrays ninits long
            of randomly sampled parameter values
    """
    if isinstance(pkeys, dict):
        pkeys = list(pkeys)
    params = {}
    for pk in pkeys:
        params[pk] = init_distributions(pk, nrvs=ninits, kind=kind, mu=mu, sigma=sigma, force_normal=force_normal)
    if as_list:
        params = np.array([{pk: params[pk][i] for pk in pkeys} for i in range(ninits)])
    return params


def loadParameters(inits=None, is_flat=False, kind=None, pcmap={}):
    """ Generates and returns an lmfit Parameters object with
    bounded parameters initialized for flat or non flat model fit
    """
    lmParams = lmParameters()
    pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso', 'C', 'B', 'R']
    pfit = list(set(inits.keys()).intersection(pnames))
    bounds = get_bounds(kind=kind)
    for pkey, pclist in pcmap.items():
        if is_flat:
            break  # exit
        pfit.remove(pkey)
        if hasattr(inits[pkey], '__iter__'):
            vals = inits[pkey]
        else:
            vals = inits[pkey] * np.ones(len(pclist))
        mn = bounds[pkey][0]
        mx = bounds[pkey][1]
        for k, v in zip(pclist, vals):
            if isinstance(v, np.ndarray):
                v = np.asscalar(v)
            lmParams.add(k, value=v, vary=True, min=mn, max=mx)
    for pk in pfit:
        inits = scalarize_params(inits, pfit)
        if is_flat:
            mn = bounds[pk][0]
            mx = bounds[pk][1]
            lmParams.add(pk, value=inits[pk], vary=True, min=mn, max=mx)
        elif isinstance(inits[pk], np.ndarray):
            pk_clist = ['_'.join([pk, pcond.split('_')[1]]) for pcond in pclist]
            for pname, pvals in zip(pk_clist, inits[pk]):
                lmParams.add(pname, value=pvals, vary=False)
        else:
            lmParams.add(pk, value=inits[pk], vary=False)
    return lmParams


def loadParameters_RL(inits, pvary, pflat, nlevels=1, kind='xdpm'):
    """ Generates and returns an lmfit Parameters object with
    bounded parameters initialized for flat or non flat model fit
    """
    lmParams = lmParameters()
    bounds = get_bounds(kind=kind)
    if nlevels > 1:
        varyIDs = ['{}_{}'.format(pkey, str(l)) for l in np.arange(nlevels)]
    for pkey in pvary:
        mn = bounds[pkey][0]
        mx = bounds[pkey][1]
        for i in range(nlevels):
            varyID = pkey
            if nlevels>1:
                varyID = '{}_{}'.format(pkey, str(l))
            lmParams.add(varyID, value=inits[pkey], vary=True, min=mn, max=mx)
    for pk in pflat:
        lmParams.add(pk, value=inits[pk], vary=False)
    return lmParams


def get_default_inits(kind='dpm', depends_on={}, learn=False, ssdelay=False):
    """ if user does not provide inits dict when initializing Model instance,
    grab default dictionary of init params reasonably suited for Model kind
    """
    inits = {'a': 0.42, 'v': 1.25, 'tr': 0.2}
    pdep = list(depends_on)

    if 'dpm' in kind:
        inits['ssv'] = -1.3
    elif 'race' in kind:
        inits['ssv'] = 1.3
    if 'x' in kind and 'xb' not in list(inits):
        inits['xb'] = 1.5
    if 'si' in pdep:
        inits['si'] = .01
    if ssdelay:
        inits['sso'] = .08
    if learn:
        inits['C'] = .02
        inits['B'] = .15
        inits['R'] = .0015
    return inits


def clean_popt(params, pcmap):
    params = deepcopy(params)
    pvary = np.sort(list(pcmap))
    for i, p in enumerate(pvary):
        p_lvls = np.sort(pcmap[p])
        params[p] = array([params[p_lvl] for p_lvl in p_lvls])
    for p in list(params):
        if p in pvary.tolist():
            continue
        pval = np.mean(params[p])
        params[p] = pval * np.ones(p_lvls.size)
    return params


def scalarize_params(params, pcmap={}):
    """ scalarize all parameters in params dict """
    params = deepcopy(params)
    pvary = np.sort(list(pcmap)).tolist()
    if isinstance(params, pd.Series):
        params = params.to_dict()
    for p in list(params):
        if p in pvary:
            continue
        if hasattr(params[p], '__iter__'):
            try:
                params[p] = np.asscalar(params[p])
            except ValueError:
                params[p] = np.mean(params[p])
    return params


def pvary_levels_to_array(popt, pcmap={}):
    """ store optimized values of parameter in popt dict as array
        Example:
            from:   {'v_bsl': 1.15, 'v_pnl':1.10, ...}
            to:     {'v': array([1.15, 1.10]), ...}
    """
    for pname, pconds in pcmap.items():
        popt[pname] = np.array([popt[c] for c in pconds])
    return popt


def check_inits(inits={}, depends_on={}, kind='dpm'):
    """ ensure inits dict is appropriate for Model kind
    """
    pdep = list(depends_on)
    if 'race' in kind or 'iact' in kind:
        inits['ssv'] = abs(inits['ssv'])
    elif 'dpm' in kind:
        inits['ssv'] = -abs(inits['ssv'])
    if 'pro' in kind:
        if 'ssv' in inits.keys():
            ssv = inits.pop('ssv')
    if 'x' in kind and 'xb' not in inits.keys():
        inits['xb'] = 1.5
    if 'si' in pdep and 'si' not in inits.keys():
        inits['si'] = .01
    if 'dpm' not in kind and 'z' in inits.keys():
        discard = inits.pop('z')
    if 'x' not in kind and 'xb' in inits:
        inits.pop('xb')
    # make sure inits only contains subsets of these params
    pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso', 'C', 'B', 'R']
    pfit = list(set(list(inits)).intersection(pnames))
    return {pk: inits[pk] for pk in pfit}


def update_params(theta):
    if 't_hi' in theta.keys():
        theta['tr'] = theta['t_lo'] + np.random.uniform() * (theta['t_hi'] - theta['t_lo'])
    if 'z_hi' in theta.keys():
        theta['z'] = theta['z_lo'] + np.random.uniform() * (theta['z_hi'] - theta['z_lo'])
    if 'sv' in theta.keys():
        theta['v'] = theta['sv'] * np.random.randn() + theta['v']
    return theta


def get_intervar_ranges(theta):
    """ theta (dict): parameters dictionary
    """
    if 'st' in theta.keys():
        theta['t_lo'] = theta['tr'] - theta['st'] / 2
        theta['t_hi'] = theta['tr'] + theta['st'] / 2
    if 'sz' in theta.keys():
        theta['z_lo'] = theta['z'] - theta['sz'] / 2
        theta['z_hi'] = theta['z'] + theta['sz'] / 2
    return theta
