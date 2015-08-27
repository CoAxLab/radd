#!/usr/local/bin/env python
from __future__ import division
import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd import models
from radd.toolbox import theta
from radd.toolbox.messages import logger, basin_accept_fun
from lmfit import Parameters, minimize
from radd.CORE import RADDCore
from scipy.optimize import basinhopping, differential_evolution, brute


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

      def __init__(self, dframes=None, fitparams=None, kind='radd', inits=None, fit_on='average', depends_on=None, niter=50, fit_whole_model=True, method='nelder', pc_map=None, wts=None, multiopt=True, global_method='basinhopping', *args, **kws):

            self.multiopt=multiopt
            self.fit_on = fit_on
            self.data=dframes['data']
            self.fitparams=fitparams
            self.global_method=global_method
            self.kind=kind
            self.xbasin=[]
            self.dynamic=self.fitparams['dynamic']
            nq = len(self.fitparams['prob'])
            nc = self.fitparams['ncond']

            if fit_on in ['subjects', 'bootstrap']:

                  self.fits = dframes['fits']
                  self.fitinfo = dframes['fitinfo']
                  self.indx_list = dframes['observed'].index
                  self.dat=dframes['dat']
                  self.get_id = lambda x: ''.join(['Idx ', str(self.indx_list[x])])

                  if self.data_style=='re':
                        self.get_flaty = lambda x: x.mean(axis=0)
                  elif self.data_style=='pro':
                        self.get_flaty = lambda x:np.hstack([x[:nc].mean(),x[nc:].reshape(2,nq).mean(axis=0)])

            self.method=method
            self.avg_y=self.fitparams['avg_y'].flatten()
            self.avg_wts=self.fitparams['avg_wts']

            self.flat_y=self.fitparams['flat_y']
            self.flat_wts=self.fitparams['flat_wts']

            self.pc_map=pc_map
            self.pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
            self.pvc=deepcopy(['a', 'tr', 'v', 'xb'])

            super(Optimizer, self).__init__(kind=kind, data=self.data, fit_on=fit_on, depends_on=depends_on, inits=inits, fit_whole_model=fit_whole_model, niter=niter)


      def make_simulator(self):
            # initate simulator object of model being optimized
            self.simulator = models.Simulator(fitparams=self.fitparams, kind=self.kind, inits=self.inits, pc_map=self.pc_map)


      def optimize_model(self, save=True, savepth='./'):

            if not hasattr(self, 'simulator'):
                  self.make_simulator()

            # make sure inits only contains subsets of these params
            pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
            pfit = list(set(self.inits.keys()).intersection(pnames))
            self.inits = {pk:self.inits[pk] for pk in pfit}

            if self.fit_on=='average':
                  self.yhat, self.fitinfo, self.popt = self.__opt_routine__()
            elif self.fit_on in ['subjects', 'bootstrap']:
                  self.__indx_optimize__(save=save, savepth=savepth)

            return self.yhat, self.fitinfo, self.popt


      def __indx_optimize__(self, save=True, savepth='./'):

            ri=0; nc=self.ncond; nquant=len(self.fitparams['prob'])
            pcols=self.fitinfo.columns

            for i, y in enumerate(self.dat):
                  self.y = y
                  self.fit_on = getid(i)
                  self.flat_y = self.get_flaty(y)
                  # optimize params iterating over subjects/bootstraps
                  yhat, finfo, popt = self.__opt_routine__()

                  self.fitinfo.iloc[i]=pd.Series({pc: finfo[pc] for pc in pcols})
                  if self.data_style=='re':
                        self.fits.iloc[ri:ri+nc, :] = yhat.reshape(nc, len(self.fits.columns))
                        ri+=nc
                  elif self.data_style=='pro':
                        self.fits.iloc[i] = yhat
                  if save:
                        self.fits.to_csv(savepth+"fits.csv")
                        self.fitinfo.to_csv(savepth+"fitinfo.csv")
            self.popt = self.__extract_popt_fitinfo__(self, self.fitinfo.mean())


      def __opt_routine__(self):
            """ main function for running optimization routine through all phases
            (flat optimization, pre-tuning with basinhopping alg., final simplex)
            """

            fp = self.fitparams

            # p0: (Initials/Global Minimum)
            p0 = dict(deepcopy(self.inits))
            if self.multiopt:
                  #hop_around --> basinhopping_full
                  p0 = self.hop_around(p0)

            # p1: STAGE 1 (Initial Simplex)
            yh1, finfo1, p1 = self.gradient_descent(y=self.flat_y, wts=self.flat_wts, inits=p0, is_flat=True)

            # p2: STAGE 2 (Nudge/BasinHopping)
            p2 = self.__nudge_params__(p1)
            if self.multiopt:
                  # pretune conditional parameters (1/time)
                  p2 = self.single_basin(p2)

            # p3: STAGE 3 (Final Simplex)
            yh3, finfo3, p3 = self.gradient_descent(y=self.avg_y, wts=self.avg_wts, inits=p2, is_flat=False)
            return yh3, finfo3, p3


      def hop_around(self, p, nrand_params=5, nsuccess=40):
            """ initialize model with niter randomly generated parameter sets
            and perform global minimization using basinhopping algorithm
            ::Arguments::
                  niter (int):
                        number of randomly generated parameter sets
                  nsuccess (int):
                        tell basinhopping algorithm to exit after this many
                        iterations at which a common minimum is found
            ::Returns::
                  parameter set with the best fit
            """
            inits = deepcopy(self.inits.keys())
            bnd = theta.get_bounds(kind=self.kind, tb=self.fitparams['tb'])

            random_inits = {pkey: theta.init_distributions(pkey, bnd[pkey], nrvs=nrand_params, kind=self.kind) for pkey in inits}
            xpopt, xfmin = [], []
            for i in range(nrand_params):
                  if i==0:
                        p=dict(deepcopy(p))
                  else:
                        p={pkey: random_inits[pkey][i] for pkey in inits}
                  popt, fmin = self.basinhopping_full(p=p, is_flat=True, nsuccess=nsuccess)
                  xpopt.append(popt)
                  xfmin.append(fmin)

            ix_min = np.argmin(xfmin)
            new_inits = xpopt[ix_min]
            fmin = xfmin[ix_min]
            if ix_min==0:
                  self.basin_decision = "using default inits: fmin=%.9f" % xfmin[0]
            else:
                  self.basin_decision = "found global miniumum new: fmin=%.9f; norig=%9f)"%(fmin, xfmin[0])
                  self.global_inits=dict(deepcopy(new_inits))
            return new_inits


      def basinhopping_full(self, p, nsuccess=20, stepsize=.05, niter=100, interval=10, is_flat=True, disp=True):
            """ uses fmin_tnc in combination with basinhopping to perform bounded global
             minimization of multivariate model

             is_flat (bool <True>):
                  if true, optimize all params in p
            """
            fp = self.fitparams
            if is_flat:
                  basin_keys=p.keys()
                  bp=dict(deepcopy(p))
                  basin_params = theta.all_params_to_scalar(bp)
                  ncond=1
            else:
                  basin_keys = self.pc_map.keys()
                  ncond = len(self.pc_map.values()[0])
                  basin_params = self.__nudge_params__(p)

            self.simulator.__prep_global__(basin_params=basin_params, basin_keys=basin_keys, is_flat=is_flat)
            xmin, xmax = theta.format_basinhopping_bounds(basin_keys, kind=self.kind, ncond=ncond)

            x = np.hstack(np.hstack([basin_params[pk] for pk in basin_keys])).tolist()
            bounds = map((lambda x: tuple([x[0], x[1]])), zip(xmin, xmax))
            mkwargs = {"method": "TNC", "bounds":bounds}

            # run basinhopping on simulator.basinhopping_minimizer func
            out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=stepsize, niter_success=nsuccess, minimizer_kwargs=mkwargs, niter=niter, interval=interval, disp=disp)
            xopt = out.x
            funcmin = out.fun
            if ncond>1:
                  xopt = array([xopt]).reshape(len(basin_keys), ncond)
            for i, k in enumerate(basin_keys):
                  p[k]=xopt[i]
            return p, funcmin


      def single_basin(self, p, disp=True, interval=20, niter=100, stepsize=.05, nsuccess=40):
            """ uses basinhopping and fmin_tnc to pre-optimize init cond parameters
            to individual conditions to prevent terminating in local minima
            """

            fp = self.fitparams; xbasin = []
            pkeys = self.pc_map.keys()
            allbounds = theta.get_bounds()
            if not hasattr(self, 'bdata'):
                  self.bdata, self.bwts = self.__prep_basin_data__()
            if not np.all([hasattr(p[pk], '__iter__') for pk in pkeys]):
                  p=self.__nudge_params__(p)

            p = theta.all_params_to_scalar(p, exclude=pkeys)
            mkwargs = {"method":"TNC", 'bounds':[allbounds[pk] for pk in pkeys]}
            self.simulator.__prep_global__(basin_params=p, basin_keys=pkeys, is_flat=True)
            # make list of init values for all keys in pc_map,
            # for all conditions in depends_on.values()
            vals = [[p[pk][i] for pk in pkeys] for i in range(fp['ncond'])]

            for i, x in enumerate(vals):
                  self.simulator.__update__(is_flat=True, y=self.bdata[i], wts=self.bwts[i])
                  out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=stepsize, minimizer_kwargs=mkwargs, interval=interval, niter=niter, niter_success=nsuccess, disp=disp)
                  xbasin.append(out.x)
            for i, pk in enumerate(pkeys):
                  p[pk]=array([xbasin[ci][i] for ci in range(fp['ncond'])])
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
                        y=self.flat_y
                        wts=self.flat_wts
                  else:
                        y=self.avg_y
                        wts=self.avg_wts
            if inits is None:
                  inits=dict(deepcopy(self.inits))

            self.simulator.__update__(y=y.flatten(), wts=wts.flatten(), is_flat=is_flat)
            opt_kws = {'disp':fp['disp'], 'xtol':fp['tol'], 'ftol':fp['tol'], 'maxfev':fp['maxfev']}

            # GEN PARAMS OBJ & OPTIMIZE THETA
            lmParams=theta.loadParameters(inits=inits, pc_map=self.pc_map, is_flat=is_flat, kind=self.kind)
            optmod = minimize(self.simulator.__cost_fx__, lmParams, method='nelder', options=opt_kws)

            # gen dict of opt. params
            finfo = dict(deepcopy(optmod.params.valuesdict()))
            popt = dict(deepcopy(finfo))
            yhat = self.simulator.sim_fx(popt)#y# + self.residual
            wts = self.simulator.wts

            # ASSUMING THE WSSE is CALC BY COSTFX
            finfo['chi']=np.asscalar(optmod.residual)
            finfo['ndata']=len(yhat)
            finfo['cnvrg'] = optmod.pop('success')
            finfo['nfev'] = optmod.pop('nfev')
            finfo['nvary'] = len(optmod.var_names)
            finfo = self.assess_fit(finfo)

            log_arrays = {'y':self.simulator.y, 'yhat':yhat, 'wts':wts}
            logger(optmod=optmod, finfo=finfo, pdict=popt, depends_on=fp['depends_on'], log_arrays=log_arrays, is_flat=is_flat, kind=self.kind, fit_on=self.fit_on, xbasin=self.xbasin, dynamic=self.dynamic, pc_map=self.pc_map)

            return  yhat, finfo, popt
