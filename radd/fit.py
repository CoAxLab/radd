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
from radd.tools.vis import PBinJ

class BasinFeedback(object):
    """ tracks BasinHopping progress with tqdm bars
    Arguments:
        nsuccess:
            Stop the run if the global minimum candidate remains the
            same for this number of iterations.
    """
    def __init__(self, ninits, fmin0=None):
        self.inits_bar = PBinJ(bartype='colorbar', n=ninits+1, color='blue', title='global fmin')
        self.success_bar = PBinJ(bartype='infobar', title='current fmin')
        self.basins = []
        if fmin0 is not None:
            self.inits_bar.update(new_progress=fmin0)
            self.basins.append(fmin0)

    def callback(self, x, fmin, accept):
        """ A callback function which will be called for all minima found
        Arguments:
            x (array):
                parameter values
            fmin (float):
                function value of the trial minimum, and
            accept (bool):
                whether or not that minimum was accepted
        """
        if fmin <= np.min(self.basins):
            self.inits_bar.update(new_progress=fmin)
        if accept:
            self.basins.append(fmin)
            self.success_bar.update(new_progress=fmin)

    def clear(self):
        self.inits_bar.clear()
        self.success_bar.clear()

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

    def sample_inits(self, pkeys=None):
        """ test a large sample of random parameter values
        and submit <nkeep> to hop_around() global optimization
        """
        bp = self.basinparams
        nkeep = bp['ninits']
        nsamples = bp['nsamples']
        keep_method = bp['init_sample_method']
        if pkeys is None:
            pkeys = np.sort(list(self.inits))
        rinits = theta.random_inits(pkeys, ninits=nsamples, kind=self.kind)
        all_inits = [{pk: rinits[pk][i] for pk in pkeys} for i in range(nsamples)]
        fmin_all = [self.simulator.cost_fx(inits_i, sse=True) for inits_i in all_inits]
        # rank inits by costfx error low-to-high
        fmin_series = pd.Series(fmin_all)
        rankorder = fmin_series.sort_values()
        # eliminate extremely bad parameter sets
        rankorder = rankorder[rankorder<=5.0]
        if keep_method=='random':
            # return nkeep from randomly sampled inits
            inits = all_inits[:nkeep]
            inits_err = fmin_all[:nkeep]
        elif keep_method=='best':
            # return nkeep from inits with lowest err
            inits = [all_inits[i] for i in rankorder.index[:nkeep]]
            inits_err = rankorder.values[:nkeep]
        elif keep_method=='lmh':
            # split index for low, med, and high err inits
            # if nkeep is odd, will sample more low than high
            if nkeep<3: nkeep=3
            ix = rankorder.index.values
            nl, nm, nh = [arr.size for arr in np.array_split(np.arange(nkeep), 3)]
            # extract indices roughly equal numbers of parameter sets with low, med, hi err
            keep_ix = np.hstack([ix[:nl], np.array_split(ix,2)[0][-nm:], ix[-nh:]])
            inits = [all_inits[i] for i in keep_ix]
            inits_err = [fmin_series[i] for i in keep_ix]
        return inits, inits_err

    def hop_around(self, p, inits_list=None, progress=False, callback=None):
        """ initialize model with niter randomly generated parameter sets
        and perform global minimization using basinhopping algorithm
        ::Arguments::
            p (dict):
                parameter dictionary
        ::Returns::
            parameter set with the best fit
        """
        bp = self.basinparams
        if inits_list is None:
            # sample random inits and select best of
            inits_list, fmins = self.sample_inits(pkeys=np.sort(list(p)))
            f0 = np.min(fmins)
        else:
            f0=np.min([self.simulator.cost_fx(inits, sse=True) for inits in inits_list])
        if progress and not bp['disp']:
            pbars = BasinFeedback(ninits=len(inits_list), fmin0=f0)
        xpopt, xfmin = [], []
        for i, inits in enumerate(inits_list):
            if progress and not bp['disp']:
                pbars.inits_bar.update(i+1)
                callback = pbars.callback
            popt, fmin = self.run_basinhopping(p=inits, is_flat=True, callback=callback)
            xpopt.append(popt)
            xfmin.append(fmin)
        if progress:
            pbars.clear()
        # report global minimum
        print("GLOBAL MIN = {:.9f}".format(np.min(xfmin)))
        # return parameters at the global basin
        return xpopt[np.argmin(xfmin)]

    def run_basinhopping(self, p, is_flat=True, callback=None):
        """ uses fmin_tnc in combination with basinhopping to perform bounded global
         minimization of multivariate model
        ::Arguments::
            is_flat (bool <True>):
              if True, optimize all params in p
        """
        bp = self.basinparams
        if is_flat:
            basin_keys = np.sort(list(p))
            xp = dict(deepcopy(p))
            basin_params = theta.scalarize_params(xp)
            nl = 1
        else:
            basin_keys = np.sort(list(self.pc_map))
            basin_params = deepcopy(p)
            nl = self.fitparams['y'].ndim
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

    def gradient_descent(self, inits=None, is_flat=True):
        """ Optimizes parameters following specified parameter
        dependencies on task conditions (i.e. depends_on={param: cond})
        """
        fp = self.fitparams
        if inits is None:
            inits = dict(deepcopy(self.inits))
        optkws = {'disp': fp['disp'], 'xtol': fp['tol'], 'ftol': fp['tol'], 'maxfev': fp['maxfev']}
        # make lmfit Parameters object to keep track of
        # parameter names and dependencies during fir
        lmParams = theta.loadParameters(inits=inits, pc_map=self.pc_map, is_flat=is_flat, kind=self.kind)
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
        popt = theta.scalarize_params(p, pc_map=self.pc_map, is_flat=is_flat)
        return finfo, popt, yhat
