#!/usr/local/bin/env python
from __future__ import division
import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd import models, theta
from radd.tools.messages import logger
from lmfit import Parameters, minimize, fit_report
from radd.CORE import RADDCore
from scipy.optimize import basinhopping


class Optimizer(RADDCore):
    """ Optimizer class acts as interface between Model and Simulator (see fit.py) objects.
    Structures fitting routines so that Models are first optimized with the full set of
    parameters free, data collapsing across conditions.

    The fitted parameters are then used as the initial parameters for fitting conditional
    models with only a subset of parameters are left free to vary across levels of a given
    experimental conditionself.

    Parameter dependencies are specified when initializing Model object via
    <depends_on> arg (i.e.{parameter: condition})

    Handles fitting routines for models of average, individual subject, and bootstrapped data
    """

    def __init__(self, dframes=None, fitparams=None, kind='dpm', inits=None, fit_on='average', depends_on=None, niter=50, fit_whole_model=True, method='nelder', pc_map=None, wts=None, multiopt=True, global_method='basinhopping', basinparams=None, *args, **kws):

        self.multiopt = multiopt
        self.fit_on = fit_on
        self.data = dframes['data']
        self.fitparams = fitparams
        self.labels = self.fitparams['labels']
        self.basinparams = basinparams
        self.global_method = global_method
        self.kind = kind
        self.dynamic = self.fitparams['dynamic']
        if fit_on in ['subjects', 'bootstrap']:
            self.prep_indx(dframes)
        self.method = method
        self.avg_y = self.fitparams['avg_y'].flatten()
        self.avg_wts = self.fitparams['avg_wts']
        self.flat_y = self.fitparams['flat_y']
        self.flat_wts = self.fitparams['flat_wts']
        self.pc_map = pc_map
        self.pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
        self.pvc = deepcopy(['a', 'tr', 'v', 'xb'])

        super(Optimizer, self).__init__(kind=kind, data=self.data, fit_on=fit_on, depends_on=depends_on, inits=inits, fit_whole_model=fit_whole_model, niter=niter)

        # initate simulator object of model being optimized
        self.simulator = models.Simulator(fitparams=self.fitparams, kind=self.kind, inits=self.inits, pc_map=self.pc_map)


    def optimize_model(self, save=True, savepth='./'):

        # make sure inits only contains subsets of these params
        pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
        pfit = list(set(self.inits.keys()).intersection(pnames))
        self.inits = {pk: self.inits[pk] for pk in pfit}

        if self.fit_on == 'average':
            self.yhat, self.fitinfo, self.popt = self.__opt_routine__()
        elif self.fit_on in ['subjects', 'bootstrap']:
            self.__indx_optimize__(save=save, savepth=savepth)

        return self.yhat, self.fitinfo, self.popt


    def optimize_flat(self, p0=None, y=None, random_init=True):
        """ optimizes flat model to data collapsing across all conditions

        ::Arguments::
        <OPTIONAL>
              random_init (bool <False>):
                    if True performs random initializaiton by sampling from parameter distributions and uses basinhopping alg. to find global minimum before entering stage 1 simplex
              p0 (dict):
                    parameter dictionary to initalize model, if None uses init params
                    passed by Model object
              y (ndarray):
                    data to be fit; must be same shape as flat_wts vector

        """

        if p0 is None:
            # p0: (Initials/Global Minimum)
            p0 = dict(deepcopy(self.inits))
        if y is None:
            y = self.flat_y

        if random_init and self.multiopt:
            # hop_around --> basinhopping_full
            p0 = self.hop_around(p0)

        # p1: STAGE 1 (Initial Simplex)
        yh1, finfo1, p1 = self.gradient_descent(y=y, wts=self.flat_wts, inits=p0, is_flat=True)
        self.flat_finfo = finfo1
        return yh1, finfo1, p1


    def optimize_conditional(self, p=None, y=None, precond=True):
        """ optimizes full model to all conditions in data

        ::Arguments::
        <OPTIONAL>
              precond (bool <True>):
                    if True performs pre-conditionalizes params (p)  using
                    basinhopping alg. to find global minimum for each condition
                    before entering final simplex
              p (dict):
                    parameter dictionary, if None uses default init params passed by Model object
              y (ndarray):
                    data to be fit; must be same shape as avg_wts vector
        """

        if p is None:
            p = dict(deepcopy(self.inits))
        if y is None:
            y = self.avg_y

        # STAGE 2: (Nudge/BasinHopping)
        p2 = self.__nudge_params__(p)
        if precond and self.multiopt:
            # pretune conditional parameters (1/time)
            p2 = self.single_basin(p2)

        # STAGE 3: (Final Simplex)
        yhat, finfo, popt = self.gradient_descent(y=y, wts=self.avg_wts, inits=p2, is_flat=False)

        return yhat, finfo, popt


    def __opt_routine__(self):
        """ main function for running optimization routine through all phases
        (flat optimization, pre-tuning with basinhopping alg., final simplex)
        """

        # p0: (Initials/Global) &  p1: STAGE 1 (Initial Simplex)
        flat_yh, flat_fi, flat_p = self.optimize_flat()
        # STAGE 2 (Nudge/BasinHopping) & STAGE 3 (Final Simplex)
        yhat, finfo, popt = self.optimize_conditional(p=flat_p)

        return yhat, finfo, popt


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
            return P0X
        else:
            self.basin_decision = "found global miniumum new: fmin=%.9f; norig=%9f)" % (fmin, xfmin[0])
            self.global_inits = dict(deepcopy(new_inits))
        return new_inits


    def basinhopping_full(self, p, is_flat=True):
        """ uses fmin_tnc in combination with basinhopping to perform bounded global
         minimization of multivariate model

         is_flat (bool <True>):
              if true, optimize all params in p
        """

        fp = self.fitparams
        bp = self.basinparams

        if is_flat:
            basin_keys = p.keys()
            xp = dict(deepcopy(p))
            basin_params = theta.all_params_to_scalar(xp)
            ncond = 1
        else:
            basin_keys = self.pc_map.keys()
            ncond = len(self.pc_map.values()[0])
            basin_params = deepcopy(p)

        self.simulator.__prep_global__(basin_params=basin_params, basin_keys=basin_keys, is_flat=is_flat)
        xmin, xmax = theta.format_basinhopping_bounds(basin_keys, kind=self.kind, ncond=ncond)

        x = np.hstack(np.hstack([basin_params[pk] for pk in basin_keys])).tolist()

        bounds = map((lambda x: tuple([x[0], x[1]])), zip(xmin, xmax))
        mkwargs = {"method": bp['method'], "bounds": bounds, 'tol': bp['tol'], 'options': {'xtol': bp['tol'], 'ftol': bp['tol'], 'maxiter': bp['maxiter']}}

        # run basinhopping on simulator.basinhopping_minimizer func
        out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=bp['stepsize'], niter_success=bp['nsuccess'], minimizer_kwargs=mkwargs, niter=bp['niter'], interval=bp['interval'], disp=bp['disp'])

        xopt = out.x
        funcmin = out.fun
        if ncond > 1:
            xopt = array([xopt]).reshape(len(basin_keys), ncond)
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
        vals = [[p[pk][i] for pk in pkeys] for i in range(fp['ncond'])]

        for i, x in enumerate(vals):
            self.simulator.__update__(is_flat=True, y=self.bdata[i], wts=self.bwts[i])
            out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=bp['stepsize'], minimizer_kwargs=mkwargs, interval=bp['interval'], niter=niter, niter_success=nsuccess, disp=bp['disp'])
            xbasin.append(out.x)
        for i, pk in enumerate(pkeys):
            p[pk] = array([xbasin[ci][i] for ci in range(fp['ncond'])])
        return p


    def gradient_descent(self, y=None, wts=None, inits={}, is_flat=True):
        """ Optimizes parameters following specified parameter
        dependencies on task conditions (i.e. depends_on={param: cond})
        """

        if not hasattr(self, 'simulator'):
            self.make_simulator()

        fp = self.fitparams
        if y is None:
            if is_flat:
                y = self.flat_y
                wts = self.flat_wts
            else:
                y = self.avg_y
                wts = self.avg_wts
        if inits is None:
            inits = dict(deepcopy(self.inits))

        self.simulator.__update__(y=y.flatten(), wts=wts.flatten(), is_flat=is_flat)
        opt_kws = {'disp': fp['disp'], 'xtol': fp['tol'], 'ftol': fp['tol'], 'maxfev': fp['maxfev']}

        # GEN PARAMS OBJ & OPTIMIZE THETA
        lmParams = theta.loadParameters(inits=inits, pc_map=self.pc_map, is_flat=is_flat, kind=self.kind)
        optmod = minimize(self.simulator.__cost_fx__, lmParams, method='nelder', options=opt_kws)

        # gen dict of opt. params
        finfo = dict(deepcopy(optmod.params.valuesdict()))
        popt = dict(deepcopy(finfo))
        yhat = np.mean([self.simulator.sim_fx(popt) for i in xrange(100)], axis=0)
        wts = self.simulator.wts

        # ASSUMING THE weighted SSE is CALC BY COSTFX
        finfo['chi'] = np.sum(wts * (y.flatten() - yhat)**2)
        finfo['ndata'] = len(yhat)
        finfo['cnvrg'] = optmod.pop('success')
        finfo['nfev'] = optmod.pop('nfev')
        finfo['nvary'] = len(optmod.var_names)
        finfo = self.assess_fit(finfo)

        popt = theta.all_params_to_scalar(popt, exclude=self.pc_map.keys())
        log_arrays = {'y': self.simulator.y, 'yhat': yhat, 'wts': wts}
        param_report = fit_report(optmod.params)
        logger(param_report=param_report, finfo=finfo, pdict=popt, depends_on=fp['depends_on'], log_arrays=log_arrays, is_flat=is_flat, kind=self.kind, fit_on=self.fit_on, dynamic=self.dynamic, pc_map=self.pc_map)

        return yhat, finfo, popt


    def prep_indx(self, dframes):

        nq = len(self.fitparams['prob'])
        nc = self.fitparams['ncond']
        self.fits = dframes['fits']
        self.fitinfo = dframes['fitinfo']
        self.indx_list = dframes['observed'].index
        self.dat = dframes['dat']
        self.get_id = lambda x: ''.join(['Idx ', str(self.indx_list[x])])

        if self.data_style == 're':
            self.get_flaty = lambda x: x.mean(axis=0)
        elif self.data_style == 'pro':
            self.get_flaty = lambda x: np.hstack([x[:nc].mean(), x[nc:].reshape(2, nq).mean(axis=0)])

    def __indx_optimize__(self, save=True, savepth='./'):

        ri = 0
        nc = self.ncond
        nquant = len(self.fitparams['prob'])
        pcols = self.fitinfo.columns

        for i, y in enumerate(self.dat):
            self.y = y
            self.fit_on = getid(i)
            self.flat_y = self.get_flaty(y)
            # optimize params iterating over subjects/bootstraps
            yhat, finfo, popt = self.__opt_routine__()

            self.fitinfo.iloc[i] = pd.Series({pc: finfo[pc] for pc in pcols})
            if self.data_style == 're':
                self.fits.iloc[ri:ri+nc, :] = yhat.reshape(nc, len(self.fits.columns))
                ri += nc
            elif self.data_style == 'pro':
                self.fits.iloc[i] = yhat
            if save:
                self.fits.to_csv(savepth + "fits.csv")
                self.fitinfo.to_csv(savepth + "fitinfo.csv")
        self.popt = self.__extract_popt_fitinfo__(self, self.fitinfo.mean())
