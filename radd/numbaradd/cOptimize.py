#!/usr/local/bin/env python
from __future__ import division
from future.utils import listvalues
import numpy as np
import pandas as pd
from radd import theta
from radd.numbaradd import cmodels
from radd.tools import utils
from copy import deepcopy
from scipy.optimize import basinhopping
from numpy.random import uniform

class Optimizer(object):
    def __init__(self, model, param_sets=None, fitparams=None, basinparams=None, progress=True):
        self.model = model
        self.observed = model.observed
        self.observed_flat = model.observed_flat
        self.flat_wts = model.flat_wts
        self.cond_wts = model.cond_wts
        self.inits = deepcopy(model.inits)
        self.fitparams = model.fitparams
        self.basinparams = model.basinparams
        self.kind = model.kind
        self.progress = progress
        self.callback = None
        self.finished_sampling = False
        if param_sets is not None:
            self.param_sets = param_sets
            self.finished_sampling = True
        if self.progress:
            self.make_progress_bars()
        self.update()

    def update(self, fitparams=None, inits=None, force_conditional=False):
        if force_conditional:
            self.model.set_fitparams(force_conditional=1)
            self.fitparams = self.model.fitparams
        if fitparams is None:
            fitparams = self.fitparams
        if inits is None:
            inits = deepcopy(self.inits)
        self.nlevels = fitparams.nlevels
        self.simulator = cmodels.Simulator(inits, fitparams=fitparams)
        if not self.finished_sampling:
            self.sample_param_sets()

    def hop_around(self, params=None):
        """ initialize model with niter randomly generated parameter sets
        and perform global minimization using basinhopping algorithm
        ::Arguments::
            p (dict):
                parameter dictionary
        ::Returns::
            parameter set with the best fit
        """
        fp = self.fitparams.copy()
        ix = fp['ix']
        fp['nlevels'] = 1
        fp['y'] = self.observed_flat[ix]
        fp['wts'] = self.flat_wts[ix]
        self.update(fitparams=fp)
        if params is None:
            params = self.param_sets[ix]
        xpopt, xfmin = [], []
        if self.progress:
            self.make_progress_bars()
        for i, p in enumerate(params):
            if self.progress:
                self.ibar.update(value=i, status=i+1)
            popt, fmin = self.run_basinhopping(p=p)
            xpopt.append(popt)
            xfmin.append(fmin)
        if self.progress:
            self.gbar.clear()
            self.ibar.clear()
        # return parameters at the global basin
        return xpopt[np.argmin(xfmin)]

    def store_popt_array(self, popt_arr):
        p = {}
        p['a'], p['tr'], p['v'], p['xb'], p['ssv'], p['sso'], p['si'] = popt_arr
        return p

    def run_basinhopping(self, p=None):
        bp = self.basinparams
        inits = deepcopy(p)
        basin_keys = self.simulator.allparams
        nlevels = self.simulator.n_vals
        # make list of init values for all pkeys included in fit
        x0 = self.simulator.init_theta_array(deepcopy(inits))
        xmin, xmax = format_basinhopping_bounds(basin_keys, nlevels, self.kind)
        lbounds = format_local_bounds(xmin, xmax)
        mkwargs = {"method": bp['method'], 'bounds': lbounds, 'tol': bp['tol'], 'options': {'xtol': bp['tol'], 'ftol': bp['tol']}}
        if self.progress:
            self.callback = self.gbar.reset(get_call=True)
        # define custom take_step and accept_test functions
        accept_step = BasinBounds(xmin, xmax)
        custom_step = HopStep(basin_keys, nlevels=nlevels, stepsize=bp['stepsize'])
        # run basinhopping on simulator.basinhopping_minimizer func
        self.out = basinhopping(self.simulator.cost_fx, x0=x0, minimizer_kwargs=mkwargs, take_step=custom_step, accept_test=accept_step, T=bp['T'], stepsize=bp['stepsize'], niter_success=bp['nsuccess'], niter=bp['niter'], interval=bp['interval'], callback=self.callback)
        xopt = self.out.x
        funcmin = self.out.fun
        return xopt, funcmin

    def make_progress_bars(self):
        bp = self.basinparams
        self.progress = True
        # if not hasattr(self, 'gbar'):
        self.gbar = utils.BasinCallback(n=bp['nsuccess'])
        # if not hasattr(self, 'ibar'):
        ninits = bp['ninits']
        status='/'.join(['Inits {}', '{}'.format(ninits)])
        self.ibar = utils.PBinJ(n=ninits, color='g', status=status)

    def sample_param_sets(self):
        """ sample *nsamples* (default=5000, see set_fitparams) different
        parameter sets (param_sets) and get model yhat for each set (param_yhats)
        """
        pkeys = np.sort(list(self.inits))
        nsamples = self.basinparams['nsamples']
        nkeep = self.basinparams['ninits']
        # get columns for building flat y dataframe
        cols = self.model.observedDF.loc[:, 'acc':].columns
        p_sets = theta.random_inits(pkeys, ninits=nsamples, kind=self.model.kind, as_list=True)
        p_arrays = [self.simulator.init_theta_array(p) for p in p_sets]
        p_yhats = [self.simulator.simulate_model(p_arr) for p_arr in p_arrays]
        # dataframe with (nsampled param sets X ndata)
        p_yhats = pd.DataFrame(p_yhats, columns=cols, index=np.arange(nsamples))
        flat_y = pd.DataFrame(np.array(self.model.observed_flat), columns=cols)
        flat_wts = pd.DataFrame(np.array(self.model.flat_wts), columns=cols)
        self.param_sets = theta.filter_params(p_sets, p_yhats, flat_y, flat_wts, nsamples, nkeep)
        self.finished_sampling = True



class BasinBounds(object):
    """ sets conditions for step acceptance during
    basinhopping optimization routine
    Arguments:
        xmin (list): lower boundaries for each parameter
        xmax (list): upper boundaries for each parameter
    """
    def __init__(self, xmin, xmax):
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmin = bool(np.all(x >= self.xmin))
        tmax = bool(np.all(x <= self.xmax))
        return tmin and tmax


class HopStep(object):
    """ scale stepsize of basinhopping optimization according to
    individual parameters ranges (smaller steps for more sensitive params).
    See theta.get_stepsize_scalars() for parameter <--> scalar mapping.
    Arguments:
        keys (list): list of parameter names
        nlevels (list): list of levels per parameter
        stepsize (list): initial stepsize
    """
    def __init__(self, keys, nlevels, stepsize=0.05):
        self.stepsize_scalars = self.get_stepsize_scalars(keys, nlevels)
        self.stepsize = stepsize
        self.np = self.stepsize_scalars.size

    def get_stepsize_scalars(self, keys, nlevels):
        """ returns an array of scalars used by for controlling
        stepsize of basinhopping algorithm for each parameter
        """
        scalar_dict = {'a': .5, 'tr': .2, 'v': 1.5, 'vi': 1.1, 'vd': 1.5,
                   'ssv': 1.5, 'z': .1, 'xb': 1.5, 'sso': .1, 'si': .001}
        nl = [np.ones(nl) for nl in nlevels]
        stepsize_scalars = np.hstack([scalar_dict[k]*nl[i] for i,k in enumerate(keys)])
        stepsize_scalars = stepsize_scalars.squeeze()
        return stepsize_scalars

    def __call__(self, x):
        s = self.stepsize
        ss = self.stepsize_scalars
        x = np.array([x[i] + uniform(-ss[i]*s, ss[i]*s) for i in range(self.np)])
        return x



def format_local_bounds(xmin, xmax):
    """ groups (xmin, xmax) for each parameter """
    tupler = lambda xlim: tuple([xlim[0], xlim[1]])
    return map((tupler), zip(xmin, xmax))

def format_basinhopping_bounds(basin_keys, nlevels=1, kind='dpm'):
    """ returns separate lists of all parameter
    min and max values """
    allbounds = theta.get_bounds(kind=kind)
    xmin, xmax = [], []
    for i, pk in enumerate(basin_keys):
        xmin.append([allbounds[pk][0]] * nlevels[i])
        xmax.append([allbounds[pk][1]] * nlevels[i])
    xmin = np.hstack(xmin).tolist()
    xmax = np.hstack(xmax).tolist()
    return xmin, xmax

def get_stepsize_scalars(keys, nlevels):
    """ returns an array of scalars used by fit.HopStep() object
    to control stepsize of basinhopping algorithm for each parameter """
    scalar_dict = {'a': .5, 'tr': .2, 'v': 1.5, 'vi': 1.1, 'vd': 1.5,
                   'ssv': 1.5, 'z': .1, 'xb': 1.5, 'sso': .1}
    nl = [np.ones(nl) for nl in nlevels]
    stepsize_scalars = np.hstack([scalar_dict[k]*nl[i] for i,k in enumerate(keys)])
    stepsize_scalars = stepsize_scalars.squeeze()
    return stepsize_scalars
