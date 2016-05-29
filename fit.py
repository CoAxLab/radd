#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.tools import messages, theta
from radd.models import Simulator
from lmfit import minimize, fit_report
from radd.CORE import RADDCore
from scipy.optimize import basinhopping


class Optimizer(object):
    """ Optimizer class acts as interface between Model and Simulator objects.
    Structures fitting routines so that Models are first optimized with the full set of
    parameters free, data collapsing across conditions.

    The fitted parameters are then used as the initial parameters for fitting conditional
    models with only a subset of parameters are left free to vary across levels of a given
    experimental conditionself.

    Parameter dependencies are specified when initializing Model object via
    <depends_on> arg (i.e.{parameter: condition})

    Handles fitting routines for models of average, individual subject, and bootstrapped data
    """

    def __init__(self, fitparams=None, basinparams=None, inits=None, kind='xdpm', depends_on=None, niter=50, fit_whole_model=True, pc_map=None, multiopt=True, *args, **kws):

        self.inits=inits
        self.multiopt = multiopt
        self.fitparams = fitparams
        self.basinparams = basinparams
        self.kind = kind

        self.pc_map = pc_map
        self.pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
        self.pvc = deepcopy(['a', 'tr', 'v', 'xb'])
        # super(Optimizer, self).__init__(kind=kind, data=self.data, fit_on=fit_on, depends_on=depends_on, inits=inits, fit_whole_model=fit_whole_model, niter=niter)

        # initate simulator object of model being optimized
        # self.simulator = Simulator(fitparams=self.fitparams, kind=self.kind, inits=self.inits, pc_map=self.pc_map)


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

        P0X = dict(deepcopy(p))
        pkeys = p.keys()
        rinits = theta.random_inits(pkeys, ninits=bp['nrand_inits'], kind=self.kind)
        xpopt, xfmin = [], []
        for i in range(bp['nrand_inits']):
            if i == 0:
                params = dict(deepcopy(p))
            else:
                params = {pkey: rinits[pkey][i] for pkey in pkeys}
            popt, fmin = self.basinhopping_full(p=params, is_flat=True)
            xpopt.append(popt)
            xfmin.append(fmin)

        ix_min = np.argmin(xfmin)
        new_inits = xpopt[ix_min]
        fmin = xfmin[ix_min]
        self.basins = xfmin
        self.basins_popt = xpopt
        if ix_min == 0:
            self.basin_decision = "using default inits: fmin=%.9f" % xfmin[0]
            print(self.basin_decision)
            return P0X
        else:
            self.basin_decision = "found global miniumum new: fmin=%.9f; norig=%9f)" % (fmin, xfmin[0])
            self.global_inits = dict(deepcopy(new_inits))
            print(self.basin_decision)
        return new_inits


    def basinhopping_full(self, p, is_flat=True):
        """ uses fmin_tnc in combination with basinhopping to perform bounded global
         minimization of multivariate model
        ::Arguments::
            is_flat (bool <True>):
              if True, optimize all params in p
        """

        fp = self.fitparams
        bp = self.basinparams

        if is_flat:
            basin_keys = p.keys()
            xp = dict(deepcopy(p))
            basin_params = theta.all_params_to_scalar(xp)
            nlevels = 1
        else:
            basin_keys = self.pc_map.keys()
            nlevels = len(self.pc_map.values()[0])
            basin_params = deepcopy(p)

        self.simulator.__prep_global__(basin_params=basin_params, basin_keys=basin_keys, is_flat=is_flat)
        xmin, xmax = theta.format_basinhopping_bounds(basin_keys, nlevels=nlevels, kind=self.kind)
        x = np.hstack(np.hstack([basin_params[pk] for pk in basin_keys])).tolist()
        bounds = map((lambda x: tuple([x[0], x[1]])), zip(xmin, xmax))
        mkwargs = {"method": bp['method'], "bounds": bounds, 'tol': bp['tol'], 'options': {'xtol': bp['tol'], 'ftol': bp['tol'], 'maxiter': bp['maxiter']}}

        # run basinhopping on simulator.basinhopping_minimizer func
        out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=bp['stepsize'], niter_success=bp['nsuccess'], minimizer_kwargs=mkwargs, niter=bp['niter'], interval=bp['interval'], disp=bp['disp'])

        xopt = out.x
        funcmin = out.fun
        if nlevels > 1:
            xopt = array([xopt]).reshape(len(basin_keys), nlevels)
        for i, k in enumerate(basin_keys):
            p[k] = xopt[i]
        return p, funcmin


    def single_basin(self, p, niter=100, nsuccess=40):
        """ uses basinhopping and fmin_tnc to pre-optimize init cond parameters
        to individual conditions to prevent terminating in local minima
        """

        fp = self.fitparams
        bp = self.basinparams
        xbasin = []
        pkeys = self.pc_map.keys()
        allbounds = theta.get_bounds()
        bounds = [allbounds[pk] for pk in pkeys]

        if not hasattr(self, 'bdata'):
            self.bdata, self.bwts = self.__prep_basin_data__()
        if not np.all([hasattr(p[pk], '__iter__') for pk in pkeys]):
            p = self.__nudge_params__(p)

        p = theta.all_params_to_scalar(p, exclude=pkeys)
        mkwargs = {"method": bp['method'], 'bounds': [allbounds[pk] for pk in pkeys]}
        self.simulator.__prep_global__(basin_params=p, basin_keys=pkeys, is_flat=True)
        # make list of init values for all keys in pc_map,
        # for all conditions in depends_on.values()
        vals = [[p[pk][i] for pk in pkeys] for i in range(fp['nlevels'])]

        for i, x in enumerate(vals):
            self.simulator.__update__(is_flat=True, y=self.bdata[i], wts=self.bwts[i])
            out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=bp['stepsize'], minimizer_kwargs=mkwargs, interval=bp['interval'], niter=niter, niter_success=nsuccess, disp=bp['disp'])
            xbasin.append(out.x)
        for i, pk in enumerate(pkeys):
            p[pk] = array([xbasin[ci][i] for ci in range(fp['nlevels'])])

        return p


    def gradient_descent(self, y=None, wts=None, inits={}, is_flat=True):
        """ Optimizes parameters following specified parameter
        dependencies on task conditions (i.e. depends_on={param: cond})
        """

        if not hasattr(self, 'simulator'):
            self.make_simulator()

        fp = self.fitparams

        if inits is None:
            inits = dict(deepcopy(self.inits))

        self.simulator.__update__(is_flat=is_flat)
        opt_kws = {'disp': fp['disp'], 'xtol': fp['tol'], 'ftol': fp['tol'], 'maxfev': fp['maxfev']}

        # GEN PARAMS OBJ & OPTIMIZE THETA
        lmParams = theta.loadParameters(inits=inits, pc_map=self.pc_map, is_flat=is_flat, kind=self.kind)
        optmod = minimize(self.simulator.__cost_fx__, lmParams, method=fp['method'], options=opt_kws)

        # gen dict of opt. params
        finfo = dict(deepcopy(optmod.params.valuesdict()))
        popt = dict(deepcopy(finfo))
        y = self.simulator.y
        yhat = np.mean([self.simulator.sim_fx(popt) for i in xrange(100)], axis=0)
        wts = self.simulator.wts

        # ASSUMING THE weighted SSE is CALC BY COSTFX
        finfo['chi'] = np.sum(wts * (y.flatten() - yhat)**2)
        finfo['ndata'] = len(yhat)
        finfo['cnvrg'] = optmod.pop('success')
        finfo['nfev'] = optmod.pop('nfev')
        finfo['nvary'] = len(optmod.var_names)
        finfo = self.assess_fit(finfo)

        print(finfo['cnvrg'])

        popt = theta.all_params_to_scalar(popt, exclude=self.pc_map.keys())
        log_arrays = {'y': y, 'yhat': yhat, 'wts': wts}
        param_report = fit_report(optmod.params)
        messages.logger(param_report=param_report, finfo=finfo, pdict=popt, depends_on=fp['depends_on'], log_arrays=log_arrays, is_flat=is_flat, kind=self.kind, fit_on=fp['fit_on'], dynamic=fp['dynamic'], pc_map=self.pc_map)

        return yhat, finfo, popt
