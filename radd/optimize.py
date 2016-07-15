#!/usr/local/bin/env python
from __future__ import division
from future.utils import listvalues
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd import theta
from radd.models import Simulator
from lmfit import minimize, fit_report
from scipy.optimize import basinhopping
from numpy.random import uniform

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
        stepsize (list): initial stepsize
    """
    def __init__(self, keys, nlevels=1, stepsize=0.05):
        self.stepsize_scalars = theta.get_stepsize_scalars(keys, nlevels)
        self.stepsize = stepsize
        self.np = self.stepsize_scalars.size

    def __call__(self, x):
        s = self.stepsize
        ss = self.stepsize_scalars
        x = np.array([x[i] + uniform(-ss[i]*s, ss[i]*s) for i in range(self.np)])
        return x

class Optimizer(object):
    """ Optimizer class acts as interface between Model and Simulator objects.
    Structures fitting routines so that Models are first optimized with the full set of
    parameters free, data collapsing across conditions.

    The fitted parameters are then used as the initial parameters for fitting conditional
    models with only a subset of parameters are left free to vary across levels of a given
    experimental conditions.

    Parameter dependencies are specified when initializing Model object via
    <depends_on> arg (i.e.{parameter: condition})

    Handles fitting routines for models of average, individual subject, and bootstrapped data
    """
    def __init__(self, fitparams=None, basinparams=None, inits=None, kind='xdpm', depends_on=None, pc_map=None):
        self.inits=inits
        self.fitparams = fitparams
        self.basinparams = basinparams
        self.kind = kind
        self.pc_map = pc_map
        self.pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
        self.constants = deepcopy(['a', 'tr', 'v', 'xb'])

    def hop_around(self, inits, pbars=None):
        """ initialize model with niter randomly generated parameter sets
        and perform global minimization using basinhopping algorithm
        ::Arguments::
            p (dict):
                parameter dictionary
        ::Returns::
            parameter set with the best fit
        """
        callback = None
        xpopt, xfmin = [], []
        for i, p in enumerate(inits):
            if pbars is not None:
                pbars.update(name='glb_basin', i=i)
                callback = pbars.callback
            popt, fmin = self.run_basinhopping(p=p, callback=callback)
            xpopt.append(popt)
            xfmin.append(fmin)
        if self.fitparams['disp']:
            # report global minimum
            print("Finished Hopping Around:\nGLOBAL MIN = {:.9f}".format(np.min(xfmin)))
        # return parameters at the global basin
        return xpopt[np.argmin(xfmin)]

    def run_basinhopping(self, p, callback=None):
        """ uses fmin_tnc in combination with basinhopping to perform bounded global
         minimization of multivariate model
        ::Arguments::
            p (dict):
                parameter dictionary
            callback (function):
                callable function for displaying optimization progress
        """
        bp = self.basinparams
        nl = self.fitparams['nlevels']
        if nl==1:
            basin_keys = np.sort(list(p))
            xp = dict(deepcopy(p))
            basin_params = theta.scalarize_params(xp)
        else:
            basin_keys = np.sort(list(self.pc_map))
            basin_params = deepcopy(p)
        self.simulator.__prep_global__(basin_params=basin_params, basin_keys=basin_keys)
        # make list of init values for all pkeys included in fit
        x0 = np.hstack(np.hstack([basin_params[pk]*np.ones(nl) for pk in basin_keys]))
        # define parameter boundaries for all params in pc_map.keys()
        # to be used by basinhopping minimizer & tnc local optimizer
        xmin, xmax = theta.format_basinhopping_bounds(basin_keys, nlevels=nl, kind=self.kind)
        tncbounds = theta.format_local_bounds(xmin, xmax)
        tncopt = {'xtol': bp['tol'], 'ftol': bp['tol']}
        mkwargs = {"method": bp['method'], 'bounds': tncbounds, 'tol': bp['tol'], 'options': tncopt}
        # define custom take_step and accept_test functions
        accept_step = BasinBounds(xmin, xmax)
        custom_step = HopStep(basin_keys, nlevels=nl, stepsize=bp['stepsize'])
        # run basinhopping on simulator.basinhopping_minimizer func
        out = basinhopping(self.simulator.global_cost_fx, x0=x0, minimizer_kwargs=mkwargs, take_step=custom_step, accept_test=accept_step, T=bp['T'], stepsize=bp['stepsize'], niter_success=bp['nsuccess'], interval=bp['interval'], disp=bp['disp'], callback=callback)
        xopt = out.x
        funcmin = out.fun
        if nl > 1:
            xopt = [array([xopt]).reshape(len(basin_keys), nl).squeeze()]
        for i, k in enumerate(basin_keys):
            p[k] = xopt[i]
        return p, funcmin

    def gradient_descent(self, p=None):
        """ Optimizes parameters following specified parameter
        dependencies on task conditions (i.e. depends_on={param: cond})
        """
        fp = self.fitparams
        if p is None:
            p = dict(deepcopy(self.inits))
        optkws = {'disp': fp['disp'], 'xtol': fp['tol'], 'ftol': fp['tol'], 'maxfev': fp['maxfev']}
        # make lmfit Parameters object to keep track of
        # parameter names and dependencies during fir
        lmParams = theta.loadParameters(inits=p, pc_map=self.pc_map, is_flat=fp['flat'], kind=self.kind)
        lmMinimizer = minimize(self.simulator.cost_fx, lmParams, method=fp['method'], options=optkws)
        self.lmMinimizer = lmMinimizer
        #self.lmMinimizer = deepcopy(lmMinimizer)
        self.param_report = fit_report(lmMinimizer.params)
        # gen dict of lmfit optimized Parameters object
        finfo = pd.Series(lmMinimizer.params.valuesdict())
        p = finfo.to_dict()
        finfo['cnvrg'] = lmMinimizer.success
        finfo['nfev'] = lmMinimizer.nfev
        finfo['nvary'] = len(lmMinimizer.var_names)
        # get model-predicted yhat vector
        yhat = (lmMinimizer.residual / self.simulator.wts) + self.simulator.y
        # un-vectorize all parameters except conditionals
        popt = theta.scalarize_params(p, pc_map=self.pc_map, is_flat=fp['flat'])
        return finfo, popt, yhat
