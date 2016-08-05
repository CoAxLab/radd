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
from radd.tools import utils

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
    def __init__(self, simulator=None, fitparams=None, basinparams=None, kind='xdpm', pc_map=None):
        if simulator is None:
            simulator = Simulator(fitparams=fitparams, kind=kind, pc_map=pc_map)
        self.simulator = simulator
        self.fitparams = simulator.fitparams
        self.pc_map = simulator.pc_map
        self.kind = simulator.kind
        self.basinparams = basinparams
        self.pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
        self.constants = deepcopy(['a', 'tr', 'v', 'xb'])
        self.callback = None
        self.progress = False

    def update(self, get_simulator=False, **kwargs):
        kw_keys = list(kwargs)
        kw = pd.Series(kwargs)
        if 'fitparams' in kw_keys:
            self.fitparams = kw.fitparams
        if 'basinparams' in kw_keys:
            self.basinparams = kw.basinparams
        if 'pc_map' in kw_keys:
            self.pc_map = kw.pc_map
        if 'simulator' in kw_keys:
            self.simulator = kw.simulator
        if self.basinparams['progress']:
            self.make_progress_bars()
        self.simulator.update(fitparams=self.fitparams, pc_map=self.pc_map)
        if get_simulator:
            return self.simulator

    def hop_around(self, inits, pbars=None):
        """ initialize model with niter randomly generated parameter sets
        and perform global minimization using basinhopping algorithm
        ::Arguments::
            p (dict):
                parameter dictionary
        ::Returns::
            parameter set with the best fit
        """
        xpopt, xfmin = [], []
        for i, p in enumerate(inits):
            if self.progress:
                self.ibar.update(value=i, status=i)
                self.callback = self.gbar.reset(get_call=True)
                if i>0:
                    self.gbar.reset(bar=True)
            popt, fmin = self.run_basinhopping(p=p)
            xpopt.append(popt)
            xfmin.append(fmin)
        if self.progress:
            self.ibar.clear()
            self.gbar.clear()
        # return parameters at the global basin
        return xpopt[np.argmin(xfmin)]

    def run_basinhopping(self, p):
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
        nsuccess = bp['nsuccess']
        if nl==1:
            basin_keys = np.sort(list(p))
            basin_params = theta.scalarize_params(deepcopy(p), is_flat=True)
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
        out = basinhopping(self.simulator.global_cost_fx, x0=x0, minimizer_kwargs=mkwargs, take_step=custom_step, accept_test=accept_step, T=bp['T'], stepsize=bp['stepsize'], niter_success=bp['nsuccess'], niter=bp['niter'], interval=bp['interval'], callback=self.callback)
        xopt = out.x
        funcmin = out.fun
        if nl > 1:
            xopt = [array([xopt]).reshape(len(basin_keys), nl).squeeze()]
        for i, k in enumerate(basin_keys):
            p[k] = xopt[i]
        return p, funcmin

    def gradient_descent(self, p, flat=True):
        """ Optimizes parameters following specified parameter
        dependencies on task conditions (i.e. depends_on={param: cond})
        ::Arguments::
            p (dict):
                parameter dictionary
            flat (bool):
                if flat, yhat have ndim=1, else ndim>1
                and popt lmParams will have conditional param names
        ::Returns::
            yhat (array):
                model-predicted data array
            finfo (pd.Series):
                fit info (AIC, BIC, chi2, redchi, etc)
            popt (dict):
                optimized parameters dictionary
        """
        fp = self.fitparams
        optkws = {'xtol': fp['tol'], 'ftol': fp['tol'], 'maxfev': fp['maxfev']}
        # make lmfit Parameters object to keep track of
        # parameter names and dependencies during fir
        lmParams = theta.loadParameters(inits=p, pc_map=self.pc_map, is_flat=flat, kind=self.kind)
        self.lmMin = minimize(self.simulator.cost_fx, lmParams, method=fp['method'], options=optkws)
        #self.lmMinimizer = deepcopy(lmMinimizer)
        self.param_report = fit_report(self.lmMin.params)
        return self.assess_fit(flat=flat)

    def assess_fit(self, flat=True):
        """ wrapper for analyze.assess_fit calculates and stores
        rchi, AIC, BIC and other fit statistics
        ::Arguments::
            flat (bool):
                if flat, yhat have ndim=1, else ndim>1
        ::Returns::
            yhat (array), finfo (pd.Series), popt (dict)
            see gradient_descent() docstrings
        """
        fp = deepcopy(self.fitparams)
        y = self.simulator.y.flatten()
        wts = self.simulator.wts.flatten()
        # gen dict of lmfit optimized Parameters object
        popt = dict(self.lmMin.params.valuesdict())
        # un-vectorize all parameters except conditionals
        popt = theta.scalarize_params(popt, pc_map=self.pc_map, is_flat=flat)
        finfo = pd.Series(popt)
        # get model-predicted yhat vector
        fp['yhat'] = (self.lmMin.residual / wts) + y
        # fill finfo dict with goodness-of-fit info
        finfo['cnvrg'] = self.lmMin.success
        finfo['nfev'] = self.lmMin.nfev
        finfo['nvary'] = len(self.lmMin.var_names)
        finfo['chi'] = np.sum(wts*(fp['yhat'] - y)**2)
        finfo['ndata'] = len(fp['yhat'])
        finfo['df'] = finfo.ndata - finfo.nvary
        finfo['rchi'] = finfo.chi / finfo.df
        finfo['logp'] = finfo.ndata * np.log(finfo.rchi)
        finfo['AIC'] = finfo.logp + 2 * finfo.nvary
        finfo['BIC'] = finfo.logp + np.log(finfo.ndata * finfo.nvary)
        return finfo, popt, fp['yhat']

    def make_progress_bars(self):
        bp = self.basinparams
        self.progress = True
        if not hasattr(self, 'ibar'):
            ninits = bp['ninits']
            status='/'.join(['Inits {}', '{}'.format(ninits)])
            self.ibar = utils.PBinJ(n=ninits, color='g', status=status)
            self.gbar = utils.BasinCallback(n=bp['nsuccess'])
            self.callback = self.gbar.callback
