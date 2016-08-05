#!usr/bin/env python
from __future__ import division
from future.utils import listvalues
import pandas as pd
import numpy as np
from copy import deepcopy
from numpy import array
from scipy.stats.distributions import norm, gamma, uniform
from lmfit import Parameters as lmParameters

class Parameters(object):

    def __init__(self, kind='xdpm', inits=None, tb=None, depends_on={'all':'flat'}):
        self.kind = kind
        self.inits = inits
        self.tb = tb
        self.depends_on = depends_on
        if self.inits is None:
            self.inits = self.get_default_inits()
        self.allparams = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']

    def random_inits(self, pkeys, ninits=1, kind='dpm', mu=None, sigma=None, as_list=False, get_params=False):
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
        pkeys = list(self.inits)
        params = {}
        for pk in pkeys:
            params[pk] = init_distributions(pk, nrvs=ninits, kind=kind, mu=mu, sigma=sigma)
        if 'vd' in params.keys():
            # vi_perc ~ U(.05, .95) --> vi = vi_perc*vd
            params['vi'] = params['vi']*params['vd']
        if as_list:
            params = [{pk: params[pk][i] for pk in pkeys} for i in range(ninits)]
        if get_params:
            return params
        self.param_sets = params

    def filter_params(self, p_sets, p_fmins, nkeep=5, method='best'):
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
        return inits, np.min(inits_err)

    def loadParameters(self, params=None, is_flat=False, kind=None, pc_map={}):
        """ Generates and returns an lmfit Parameters object with
        bounded parameters initialized for flat or non flat model fit
        """
        if params is None:
            params = self.inits
        p = deepcopy(params)
        lmParams = lmParameters()
        pfit = list(set(p.keys()).intersection(self.allparams))
        bounds = self.get_bounds()
        for pkey, pclist in self.pc_map.items():
            if is_flat:
                break  # exit
            pfit.remove(pkey)
            if hasattr(p[pkey], '__iter__'):
                vals = p[pkey]
            else:
                vals = p[pkey] * np.ones(len(pclist))
            mn = bounds[pkey][0]
            mx = bounds[pkey][1]
            for k, v in zip(pclist, vals):
                if isinstance(v, np.ndarray):
                    v = np.asscalar(v)
                lmParams.add(k, value=v, vary=True, min=mn, max=mx)
        for pk in pfit:
            p = self.scalarize_params(p, pc_map=pfit, is_flat=is_flat)
            if is_flat:
                mn = bounds[pk][0]
                mx = bounds[pk][1]
                lmParams.add(pk, value=p[pk], vary=True, min=mn, max=mx)
            else:
                lmParams.add(pk, value=p[pk], vary=False)
        return lmParams

    def scalarize_params(self, params, pc_map=None, is_flat=True):
        """ scalarize all parameters in params dict """
        exclude = []
        if isinstance(params, pd.Series):
            params = params.to_dict()
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

    def init_distributions(self, pkey, kind='dpm', mu = None, sigma = None, nrvs=25, tb=.65):
        """ sample random parameter sets to explore global minima (called by
        Optimizer method __hop_around__())
        """
        if mu is None:
            mu = {'a': .15, 'tr': .02, 'v': 1., 'ssv': -1., 'z': .1, 'xb': 1., 'sso': .15, 'vi': .35, 'vd': .5}
        if sigma is None:
            sigma = {'a': .35, 'tr': .25, 'v': .5, 'ssv': .5, 'z': .05, 'xb': .5, 'sso': .01, 'vi': .4, 'vd': .5}
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

    def get_bounds(self, kind='dpm', a=(.05, 1.5), tr=(.01, .5), v=(.1, 5.0), z=(.01, .79), ssv=(-5.0, -.1), xb=(.1, 5.), si=(.001, .2), sso=(.01, .5), vd=(.6, 1.1), vi=(.4, .8)):
        """ set and return boundaries to limit search space
        of parameter optimization in <optimize_theta>
        """
        if 'irace' in kind:
            ssv = (abs(ssv[1]), abs(ssv[0]))
        self.bounds = {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'vd':vd, 'vi':vi,
                  'z': z, 'xb': xb, 'si': si, 'sso': sso}
        if get_bounds:
            return self.bounds

    def format_local_bounds(self, xmin, xmax):
        """ groups (xmin, xmax) for each parameter """
        tupler = lambda xlim: tuple([xlim[0], xlim[1]])
        return map((tupler), zip(xmin, xmax))

    def format_basinhopping_bounds(self, basin_keys, nlevels=1, kind='dpm'):
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

    def get_stepsize_scalars(self, keys, nlevels=1):
        """ returns an array of scalars used by fit.HopStep() object
        to control stepsize of basinhopping algorithm for each parameter """
        scalar_dict = {'a': .5, 'tr': .1, 'v': 1.5, 'vi': 1.5, 'vd': 1.5,
                       'ssv': 1.5, 'z': .1, 'xb': 1.5, 'sso': .1}
        stepsize_scalars = np.array([scalar_dict[k] for k in keys]*nlevels)
        if nlevels>1:
            stepsize_scalars = stepsize_scalars.squeeze()
        return stepsize_scalars

    def get_default_inits(self):
        """ if user does not provide inits dict when initializing Model instance,
        grab default dictionary of init params reasonably suited for Model kind
        """
        inits = {'a': 0.5, 'v': 1.2, 'xb': 1.5, 'tr': 0.2}
        if 'dpm' in self.kind:
            inits['ssv'] = -1.
        elif 'race' in self.kind:
            inits['ssv'] = 1
        if 'x' in self.kind:
            inits['xb'] = 1.5
        return inits

    def check_inits(self, inits={}, depends_on={}, kind='dpm'):
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
        params = [{pk: params[pk][i] for pk in pkeys} for i in range(ninits)]
    return params

def filter_params(p_sets, p_fmins, nkeep=5, method='best'):
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
    return inits, np.min(inits_err)

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

def scalarize_params(params, pc_map=None, is_flat=True):
    """ scalarize all parameters in params dict """
    exclude = []
    if isinstance(params, pd.Series):
        params = params.to_dict()
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

def init_distributions(pkey, kind='dpm', mu = None, sigma = None, nrvs=25, tb=.65):
    """ sample random parameter sets to explore global minima (called by
    Optimizer method __hop_around__())
    """
    if mu is None:
        mu = {'a': .15, 'tr': .02, 'v': 1., 'ssv': -1., 'z': .1, 'xb': 1., 'sso': .15, 'vi': .35, 'vd': .5}
    if sigma is None:
        sigma = {'a': .35, 'tr': .25, 'v': .5, 'ssv': .5, 'z': .05, 'xb': .5, 'sso': .01, 'vi': .4, 'vd': .5}
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

def get_bounds(kind='dpm', a=(.05, 1.5), tr=(.01, .5), v=(.1, 5.0), z=(.01, .79), ssv=(-5.0, -.1), xb=(.1, 5.), si=(.001, .2), sso=(.01, .5), vd=(.6, 1.1), vi=(.4, .8)):
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

def get_default_inits(kind='dpm', depends_on={}):
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
