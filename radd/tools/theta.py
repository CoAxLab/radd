#!usr/bin/env python
from __future__ import division
from future.utils import listvalues
import pandas as pd
import numpy as np
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
        pkeys = list(pkeys)
    params = {}
    for pk in pkeys:
        params[pk] = init_distributions(pk, nrvs=ninits, kind=kind)
    if 'vd' in params.keys():
        # vi_perc ~ U(.05, .95) --> vi = vi_perc*vd
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
        inits = scalarize_params(inits, pfit)
        if is_flat:
            mn = bounds[pk][0]
            mx = bounds[pk][1]
            ParamsObj.add(pk, value=inits[pk], vary=True, min=mn, max=mx)
        else:
            ParamsObj.add(pk, value=inits[pk], vary=False)
    return ParamsObj

def scalarize_params(params, pc_map=None, is_flat=True):
    """ scalarize all parameters in params dict """
    exclude = []
    if pc_map is not None and not is_flat:
        exclude = np.sort(list(pc_map))
        p_conds = np.sort(listvalues(pc_map)).squeeze()
        for pkc in exclude:
            params[pkc] = array([params[pc] for pc in p_conds])
    for pk in list(params):
        if pk in exclude:
            continue
        if hasattr(params[pk], '__iter__'):
            try:
                params[pk] = np.asscalar(params[pk])
            except ValueError:
                params[pk] = np.mean(params[pk])
    return params

def init_distributions(pkey, kind='dpm', nrvs=25, tb=.65):
    """ sample random parameter sets to explore global minima (called by
    Optimizer method __hop_around__())
    """
    mu_defaults = {'a': .15, 'tr': .02, 'v': 1., 'ssv': -1., 'z': .1, 'xb': 1., 'sso': .15, 'vi': .15, 'vd': 1.}
    sigma_defaults = {'a': .3, 'tr': .2, 'v': .3, 'ssv': .3, 'z': .05, 'xb': .5, 'sso': .01, 'vi': .7, 'vd': .2}
    normal_params = ['tr', 'v', 'vd', 'ssv', 'z', 'xb', 'sso']
    gamma_params = ['a', 'tr']
    uniform_params = ['vi']
    if 'race' in kind:
        mu_defaults['ssv'] = abs(mu_defaults['ssv'])

    bounds = get_bounds(kind=kind)[pkey]
    loc = mu_defaults[pkey]
    scale = sigma_defaults[pkey]

    # init and freeze dist shape
    if pkey in normal_params:
        dist = norm(loc, scale)
    elif pkey in gamma_params:
        dist = gamma(.8, loc, scale)
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
    return rvinits

def get_bounds(kind='dpm', a=(.05, 1.5), tr=(.01, .5), v=(.1, 5.0), z=(.01, .79), ssv=(-5.0, -.1), xb=(.1, 5.), si=(.001, .2), sso=(.01, .5), vd=(.1, 5.0), vi=(.01, .1)):
    """ set and return boundaries to limit search space
    of parameter optimization in <optimize_theta>
    """
    if 'irace' in kind:
        ssv = (abs(ssv[1]), abs(ssv[0]))
    bounds = {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'vd':vd, 'vi':vi,
              'z': z, 'xb': xb, 'si': si, 'sso': sso}
    return bounds

def format_local_bounds(xmin, xmax):
    """ groups (xmin, xmax) for each parameter """
    tupler = lambda xlim: tuple([xlim[0], xlim[1]])
    return map((tupler), zip(xmin, xmax))

def format_basinhopping_bounds(basin_keys, nlevels=1, kind='dpm'):
    """ returns separate lists of all parameter
    min and max values """
    allbounds = get_bounds(kind=kind)
    xmin, xmax = [], []
    for pk in basin_keys:
        xmin.append([allbounds[pk][0]] * nlevels)
        xmax.append([allbounds[pk][1]] * nlevels)
    xmin = np.hstack(xmin).tolist()
    xmax = np.hstack(xmax).tolist()
    return xmin, xmax

def get_stepsize_scalars(keys, nlevels=1):
    """ returns an array of scalars used by fit.HopStep() object
    to control stepsize of basinhopping algorithm for each parameter """
    scalar_dict = {'a': .5, 'tr': .1, 'v': 1.5, 'vi': 1.5, 'vd': 1.5,
                   'ssv': 1.5, 'z': .1, 'xb': 1.5, 'sso': .1}
    stepsize_scalars = np.array([scalar_dict[k] for k in keys]*nlevels)
    if nlevels>1:
        stepsize_scalars = stepsize_scalars.squeeze()
    return stepsize_scalars

def get_default_inits(kind='dpm', dynamic='hyp', depends_on={}):
    """ if user does not provide inits dict when initializing Model instance,
    grab default dictionary of init params reasonably suited for Model kind
    """
    inits = {'a': 0.5, 'v': 1.2, 'xb': 1.5, 'tr': 0.2}
    if 'dpm' in kind:
        inits['ssv'] = -1.
    elif 'race' in kind:
        inits['ssv'] = 1
    if 'x' in kind and 'xb' not in inits.keys():
        inits['xb'] = 1.5
    return inits

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
    pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
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
