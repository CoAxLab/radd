#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os
from copy import deepcopy
from numpy import array
from scipy.stats.distributions import norm, gamma, uniform
from lmfit import Parameters


def random_inits(pkeys, ninits=1, kind='dpm'):
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
        pkeys = pkeys.keys()

    bnd = get_bounds(kind=kind)
    params = {pk: init_distributions(pk, bnd[pk], nrvs=ninits, kind=kind) for pk in pkeys}

    if 'vd' in params.keys():
        params['vi'] = params['vi']*params['vd']

    return params


def loadParameters(inits=None, is_flat=False, kind=None, pc_map={}):
    """ Generates and returns an lmfit Parameters object with
    bounded parameters initialized for flat or non flat model fit
    """

    ParamsObj = Parameters()
    pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
    pfit = list(set(inits.keys()).intersection(pnames))
    bounds = get_bounds(kind=kind)
    for pkey, pclist in pc_map.items():
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
            ParamsObj.add(k, value=v, vary=True, min=mn, max=mx)

    for pk in pfit:
        inits = all_params_to_scalar(inits, pfit)
        if is_flat:
            mn = bounds[pk][0]
            mx = bounds[pk][1]
            ParamsObj.add(pk, value=inits[pk], vary=True, min=mn, max=mx)
        else:
            ParamsObj.add(pk, value=inits[pk], vary=False)
    return ParamsObj


def all_params_to_scalar(params, params_list=None, exclude=[]):

    if params_list is None:
        params_list = params.keys()
    for pk in params_list:
        if pk in exclude:
            continue
        if hasattr(params[pk], '__iter__'):
            try:
                params[pk] = np.asscalar(params[pk])
            except ValueError:
                params[pk] = np.mean(params[pk])
    for exc in exclude:
        params[exc] = np.asarray(params[exc], dtype=float)
    return params


def init_distributions(pkey, bounds, tb=.65, kind='dpm', nrvs=25, loc=None, scale=None):
    """ sample random parameter sets to explore global minima (called by
    Optimizer method __hop_around__())
    """
    mu_defaults = {'a': .3, 'tr': .29, 'v': 1., 'vi': .15, 'vd': 1.,
                   'ssv': -1, 'z': .1, 'xb': 1.5, 'sso': .15}
    sigma_defaults = {'a': .15, 'tr': .1, 'v': .15, 'vi': .70, 'vd':.15,
                      'ssv': .15, 'z': .05, 'xb': .5, 'sso': .01}


    if pkey == 'si':
        return .01

    if 'race' in kind or 'iact' in kind:
        mu_defaults['ssv'] = abs(mu_defaults['ssv'])
    if loc is None:
        loc = mu_defaults[pkey]
    if scale is None:
        scale = sigma_defaults[pkey]


    # init and freeze dist shape
    if pkey in ['tr', 'v', 'vd', 'ssv', 'z', 'xb', 'sso']:
        dist = norm(loc, scale)
    elif pkey in ['vi']:
        # vi_perc ~ U(.05, .95) --> vi = vi_perc*vd
        dist = uniform(loc, scale)
    elif pkey in ['a', 'tr']:
        dist = gamma(1, loc, scale)

    # generate random variates
    rvinits = dist.rvs(nrvs)
    while rvinits.min() <= bounds[0]:
        # apply lower limit
        ix = rvinits.argmin()
        rvinits[ix] = dist.rvs()
    while rvinits.max() >= bounds[1]:
        # apply upper limit
        ix = rvinits.argmax()
        rvinits[ix] = dist.rvs()
    return rvinits


def get_bounds(kind='dpm', a=(.1, 1.0), tr=(.1, .54), v=(.1, 5.0), z=(.01, .79), ssv=(-5.0, -.1), xb=(.1, 5), si=(.001, .2), sso=(.01, .3), vd=(.1, 5.0), vi=(.01, .95)):
    """ set and return boundaries to limit search space
    of parameter optimization in <optimize_theta>
    """

    if 'irace' in kind or 'iact' in kind:
        ssv = (abs(ssv[1]), abs(ssv[0]))
    bounds = {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'vd':vd, 'vi':vi,
              'z': z, 'xb': xb, 'si': si, 'sso': sso}
    return bounds


def format_local_bounds(xmin, xmax):
    """ groups (xmin, xmax) for each parameter """
    tupler = lambda xlim: tuple([xlim[0], xlim[1]])
    return map((tupler), zip(xmin, xmax))

def format_basinhopping_bounds(basin_keys, nlevels=1):

    allbounds = get_bounds()
    xmin, xmax = [], []
    for pk in basin_keys:
        xmin.append([allbounds[pk][0]] * nlevels)
        xmax.append([allbounds[pk][1]] * nlevels)
    xmin = np.hstack(xmin).tolist()
    xmax = np.hstack(xmax).tolist()
    return xmin, xmax

def get_stepsize_scalars(keys, nlevels=1):

    scalar_dict = {'a': .5, 'tr': .1, 'v': 2., 'vi': 2., 'vd': 2.,
                   'ssv': 2., 'z': .1, 'xb': 1.5, 'sso': .1}

    stepsize_scalars = np.array([scalar_dict[k] for k in keys]*nlevels)
    stepsize_scalars = stepsize_scalars.squeeze()
    return stepsize_scalars


def get_default_inits(kind='dpm', dynamic='hyp', depends_on={}):
    """ if user does not provide inits dict when initializing Model instance,
    grab default dictionary of init params reasonably suited for Model kind
    """

    if 'dpm' in kind:
        popt = {'a': 0.53625, 'v': 1.26, 'xb': 0.87761,
                'ssv': -0.98396, 'tr': 0.17801}
    elif 'pro' in kind:
        popt = {'a': 0.48722, 'xb': 1.51129, 'tr': 0.292126, 'v': 1.718}
    elif 'race' in kind:
        popt = {'a': 0.24266,  'v': 1.05866,  'xb': 1.5,
                'ssv': 1.12441, 'tr': 0.335, 'xb': 1.46335}
    elif 'iact' in kind:
        popt = {'a': 0.44422598, 'sso': 0.2040, 'ssv': 3.02348,
                'tr': 0.21841, 'v': 1.31063, 'xb': 1.46335}
    if 'x' in kind and 'xb' not in popt.keys():
        popt['xb'] = 2.5
    return popt


def check_inits(inits={}, kind='dpm', pdep=[], pro_ss=False, fit_noise=False):
    """ ensure inits dict is appropriate for Model kind
    """
    if 'race' in kind or 'iact' in kind:
        inits['ssv'] = abs(inits['ssv'])
    elif 'dpm' in kind:
        inits['ssv'] = -abs(inits['ssv'])
    if 'pro' in kind:
        if pro_ss and 'ssv' not in inits.keys():
            inits['ssv'] = -0.9976
        elif not pro_ss and 'ssv' in inits.keys():
            ssv = inits.pop('ssv')
    if 'x' in kind and 'xb' not in inits.keys():
        inits['xb'] = 1.5
    if fit_noise or 'si' in pdep and 'si' not in inits.keys():
        inits['si'] = .01
    if 'dpm' not in kind and 'z' in inits.keys():
        discard = inits.pop('z')
    if 'x' not in kind and 'xb' in inits:
        inits.pop('xb')
    return inits


def get_proactive_params(theta, dep='v', pgo=np.arange(0, 120, 20)):
    """ takes pd.Series or dict of params
    and formats for entry to sim
    """
    if not type(theta) == dict:
        theta = theta.to_dict()['mean']
    keep = ['a', 'z', 'v', 'tr', 'ssv', 'ssd']
    keep.pop(keep.index(dep))
    pdict = {pg: theta[dep + str(pg)] for pg in pgo}
    for k in theta.keys():
        if k not in keep:
            theta.pop(k)
    return theta, pdict


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
