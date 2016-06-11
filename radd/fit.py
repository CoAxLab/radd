#!/usr/local/bin/env python
from __future__ import division
from future.utils import listvalues
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.tools import messages, theta
from radd.tools.analyze import assess_fit
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
        self.np = len(self.stepsize_scalars)

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
        self.pvc = deepcopy(['a', 'tr', 'v', 'xb'])


    def get_best_inits(self, pkeys=None, nbest=10, nrand_samples=500):
        """ test a large sample of random parameter values
        and submit <nbest> to hop_around() global optimization
        """
        if pkeys is None:
            pkeys = np.sort(list(self.inits))

        rinits = theta.random_inits(pkeys, ninits=nrand_samples, kind=self.kind)
        all_inits = [{pk: rinits[pk][i] for pk in pkeys} for i in range(nrand_samples)]

        fmin_all = [self.simulator.__cost_fx__(inits_i) for inits_i in all_inits]
        fmin_series = pd.Series(fmin_all)

        best_inits_index = fmin_series.sort_values().index[:nbest]
        best_inits = [all_inits[i] for i in best_inits_index]
        return best_inits


    def hop_around(self, p):
        """ initialize model with niter randomly generated parameter sets
        and perform global minimization using basinhopping algorithm
        ::Arguments::
            p (dict):
                parameter dictionary
        ::Returns::
            parameter set with the best fit
        """

        bp = self.basinparams
        pkeys = np.sort(list(p))

        # get cost fmin for default inits
        p0 = dict(deepcopy(p))
        p_init = dict(deepcopy(p))
        fmin0 = self.simulator.__cost_fx__(p_init)
        print("default inits: fmin0=%.9f" % (fmin0))

        # sample random inits and select best of
        best_inits = self.get_best_inits(pkeys, nbest=bp['nrand_inits'], nrand_samples=bp['nrand_samples'])

        xpopt, xfmin = [], []
        for params in best_inits:
            popt, fmin = self.run_basinhopping(p=params, is_flat=True)
            xpopt.append(popt)
            xfmin.append(fmin)

        # get the global basin and
        # corresponding parameter estimates
        ix_min = np.argmin(xfmin)
        popt_best = xpopt[ix_min]
        fmin_best = xfmin[ix_min]

        # compare global basin (fmin_best) to
        # fmin using default inits (fmin0)
        if fmin_best >= fmin0:
            basin_decision = "USING DEFAULT INITS: fmin_inits=%.9f, next_best=%.9f" % (fmin0, fmin_best)
            print(basin_decision)
            return p0
        else:
            basin_decision = "NEW GLOBAL MINIMUM: fmin_new=%.9f; fmin_inits=%9f)" % (fmin_best, fmin0)
            print(basin_decision)
            return popt_best


    def run_basinhopping(self, p, is_flat=True):
        """ uses fmin_tnc in combination with basinhopping to perform bounded global
         minimization of multivariate model
        ::Arguments::
            is_flat (bool <True>):
              if True, optimize all params in p
        """
        bp = self.basinparams
        fp = self.fitparams

        if is_flat:
            basin_keys = np.sort(list(p))
            xp = dict(deepcopy(p))
            basin_params = theta.all_params_to_scalar(xp)
            nl = 1
        else:
            basin_keys = np.sort(list(self.pc_map))
            basin_params = deepcopy(p)
            nl = fp['y'].ndim

        self.simulator.__prep_global__(basin_params=basin_params, basin_keys=basin_keys)
        # make list of init values for all pkeys included in fit
        x0 = np.hstack(np.hstack([basin_params[pk]*np.ones(nl) for pk in basin_keys]))

        # define parameter boundaries for all params in pc_map.keys()
        # to be used by basinhopping minimizer & tnc local optimizer
        xmin, xmax = theta.format_basinhopping_bounds(basin_keys, nlevels=nl)
        tncbounds = theta.format_local_bounds(xmin, xmax)
        mkwargs = {"method": bp['method'], 'bounds': tncbounds, 'tol': bp['tol'],
            'options': {'xtol': bp['tol'], 'ftol': bp['tol'], 'maxiter': bp['maxiter']}}

        # define custom take_step and accept_test functions
        accept_step = BasinBounds(xmin, xmax)
        custom_step = HopStep(basin_keys, nlevels=nl, stepsize=bp['stepsize'])

        # run basinhopping on simulator.basinhopping_minimizer func
        out = basinhopping(self.simulator.basinhopping_minimizer, x0=x0, minimizer_kwargs=mkwargs, take_step=custom_step, accept_test=accept_step, stepsize=bp['stepsize'], niter=bp['niter'], niter_success=bp['niter_success'], interval=bp['interval'], disp=bp['disp'])

        xopt = out.x
        funcmin = out.fun
        if nl > 1:
            xopt = [array([xopt]).reshape(len(basin_keys), nl).squeeze()]
        for i, k in enumerate(basin_keys):
            p[k] = xopt[i]
        return p, funcmin


    def gradient_descent(self, inits=None, is_flat=True):
        """ Optimizes parameters following specified parameter
        dependencies on task conditions (i.e. depends_on={param: cond})
        """

        fp = self.fitparams
        if inits is None:
            inits = dict(deepcopy(self.inits))

        opt_kws = {'disp': fp['disp'], 'xtol': fp['tol'], 'ftol': fp['tol'],
        'maxiter': fp['maxiter'], 'maxfev': fp['maxfev']}

        # GEN PARAMS OBJ & OPTIMIZE THETA
        lmParams = theta.loadParameters(inits=inits, pc_map=self.pc_map, is_flat=is_flat, kind=self.kind)
        optmod = minimize(self.simulator.__cost_fx__, lmParams, method=fp['method'], options=opt_kws)

        # gen dict of opt. params
        finfo = dict(deepcopy(optmod.params.valuesdict()))
        popt = dict(deepcopy(finfo))
        y = self.simulator.y
        wts = self.simulator.wts
        yhat_iter = [self.simulator.sim_fx(popt) for i in range(100)]
        yhat = np.mean(yhat_iter, axis=0)

        # ASSUMING THE weighted SSE is CALC BY COSTFX
        finfo['chi'] = np.sum(wts * (y.flatten() - yhat)**2)
        finfo['ndata'] = len(yhat)
        finfo['cnvrg'] = optmod.success
        finfo['nfev'] = optmod.nfev
        finfo['nvary'] = len(optmod.var_names)
        finfo = assess_fit(finfo)
        print(finfo['cnvrg'])

        popt = theta.all_params_to_scalar(popt, exclude=list(self.pc_map))
        log_arrays = {'y': y, 'yhat': yhat, 'wts': wts}
        param_report = fit_report(optmod.params)
        messages.logger(param_report=param_report, finfo=finfo, pdict=popt, depends_on=fp['depends_on'], log_arrays=log_arrays, is_flat=is_flat, kind=self.kind, fit_on=fp['fit_on'], dynamic=fp['dynamic'], pc_map=self.pc_map)

        return yhat, finfo, popt
