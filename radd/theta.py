#!usr/bin/env python
from __future__ import division
from future.utils import listvalues
import pandas as pd
import numpy as np
from copy import deepcopy
from numpy import array
from scipy.stats.distributions import norm, gamma, uniform
from lmfit import Parameters as lmParameters

def filter_params(p_sets, p_yhats, flat_y, flat_wts, nsamples=1500, nkeep=3):
    """ sample *nsamples* (default=5000, see set_fitparams) different
    parameter sets (param_sets) and get model yhat for each set (param_yhats)
        if fit_on==subjects flat_y shape is (n_idx X ndata)
        elseif fit_on==average flat_y shape is (1 X ndata)
    """
    # row-wise costfx, comparing y against yhat, where yhat is
    # the pred. data generated from a sampled parameter set
    costfx = lambda yhat: np.sum((flat_wts * (yhat.values - flat_y))**2, axis=1)
    # column-wise ranking function, where each p_fmin is a column from pfmin_idx
    sortvals = lambda p_fmin: p_fmin.sort_values().index
    pfmin_idx = p_yhats.apply(costfx, axis=1)
    idx_prank = pfmin_idx.apply(sortvals, axis=0)
    # get arrays of indices for each of best nkeep param_sets for each subject
    idx_p_indices = [idx_prank.loc[:nkeep-1, idxcol].values  for idxcol in idx_prank]
    # self.param_sets: subjectwise list of "best" sampled param sets
    param_sets = [p_sets[idx_pindex].tolist() for idx_pindex in idx_p_indices]
    return param_sets

def random_inits(pkeys, ninits=1, kind='dpm', mu=None, sigma=None, as_list=False):
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
        params[pk] = init_distributions(pk, nrvs=ninits, kind=kind, mu=mu, sigma=sigma)
    if 'vd' in params.keys():
        # vi_perc ~ U(.05, .95) --> vi = vi_perc*vd
        params['vi'] = params['vi']*params['vd']
    if as_list:
        params = np.array([{pk: params[pk][i] for pk in pkeys} for i in range(ninits)])
    return params

def loadParameters(inits=None, is_flat=False, kind=None, pc_map={}):
    """ Generates and returns an lmfit Parameters object with
    bounded parameters initialized for flat or non flat model fit
    """
    lmParams = lmParameters()
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
            lmParams.add(k, value=v, vary=True, min=mn, max=mx)
    for pk in pfit:
        inits = scalarize_params(inits, pfit)
        if is_flat:
            mn = bounds[pk][0]
            mx = bounds[pk][1]
            lmParams.add(pk, value=inits[pk], vary=True, min=mn, max=mx)
        else:
            lmParams.add(pk, value=inits[pk], vary=False)
    return lmParams

def clean_popt(params, pc_map):
    params = deepcopy(params)
    pvary = np.sort(list(pc_map))
    for i, p in enumerate(pvary):
        p_lvls = np.sort(pc_map[p])
        params[p] = array([params[p_lvl] for p_lvl in p_lvls])
    for p in list(params):
        if p in pvary.tolist():
            continue
        pval = np.mean(params[p])
        params[p] = pval * np.ones(p_lvls.size)
    return params

def scalarize_params(params, pc_map={}):
    """ scalarize all parameters in params dict """
    params = deepcopy(params)
    pvary = np.sort(list(pc_map)).tolist()
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

def init_distributions(pkey, kind='dpm', mu = None, sigma = None, nrvs=25, tb=.65):
    """ sample random parameter sets to explore global minima (called by
    Optimizer method __hop_around__())
    """
    if mu is None:
        mu = {'a': .3, 'tr': .06, 'v': 1., 'ssv': -1.1, 'z': .1, 'xb': 1., 'sso': .15, 'vi': .35, 'vd': .5}
    if sigma is None:
        sigma = {'a': .15, 'tr': .15, 'v': .25, 'ssv': .3, 'z': .05, 'xb': .25, 'sso': .01, 'vi': .4, 'vd': .5}
    normal_params = ['tr', 'v', 'vd', 'ssv', 'z', 'xb', 'sso']
    gamma_params = ['a', 'tr']
    uniform_params = ['vd', 'vi']
    if 'race' in kind:
        sigma['ssv'] = abs(mu['ssv'])
    bounds = get_bounds(kind=kind)[pkey]
    loc = mu[pkey]
    scale = sigma[pkey]
    # init and freeze dist shape
    if pkey in normal_params:
        dist = norm(loc, scale)
    elif pkey in gamma_params:
        dist = gamma(1.0, loc, scale)
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

def get_bounds(kind='dpm', a=(.2, .7), tr=(.01, .4), v=(.5, 1.5), z=(.01, .9), ssv=(-1.5, -.5), xb=(.2, 2.), si=(.001, .2), sso=(.01, .5), vd=(.6, 1.1), vi=(.4, .8)):
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
    nl_ones = np.ones(nlevels)
    stepsize_scalars = np.hstack([scalar_dict[k]*nl_ones for k in keys])
    if nlevels>1:
        stepsize_scalars = stepsize_scalars.squeeze()
    return stepsize_scalars

def get_default_inits(kind='dpm', depends_on={}):
    """ if user does not provide inits dict when initializing Model instance,
    grab default dictionary of init params reasonably suited for Model kind
    """
    inits = {'a': 0.5, 'v': 1., 'tr': 0.2}
    if 'dpm' in kind:
        inits['ssv'] = -1.
    elif 'race' in kind:
        inits['ssv'] = 1
    if 'x' in kind and 'xb' not in list(inits):
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


def filter_params_old(p_fmins, p_sets, nkeep=5, method='best'):
    # rank inits by costfx error low-to-high
    fmin_series = pd.Series(p_fmins)
    rankorder = fmin_series.sort_values()
    # eliminate extremely bad parameter sets
    rankorder = rankorder[rankorder<=5.0]
    if method=='random':
        # return nkeep from randomly sampled inits
        inits = p_sets[:nkeep]
        inits_err = p_fmins[:nkeep]
    elif method=='best':
        # return nkeep from inits with lowest err
        inits = [p_sets[i] for i in rankorder.index[:nkeep]]
        inits_err = rankorder.values[:nkeep]
    elif method=='lmh':
        # split index for low, med, and high err inits
        # if nkeep is odd, will sample more low than high
        if nkeep<3: nkeep=3
        ix = rankorder.index.values
        nl, nm, nh = [arr.size for arr in np.array_split(np.arange(nkeep), 3)]
        # extract indices roughly equal numbers of parameter sets with low, med, hi err
        keep_ix = np.hstack([ix[:nl], np.array_split(ix,2)[0][-nm:], ix[-nh:]])
        inits = [p_sets[i] for i in keep_ix]
        inits_err = [fmin_series[i] for i in keep_ix]
    return inits
