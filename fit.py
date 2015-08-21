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


class Theta(dict):

      """ a class that inherits from a custom dictionary emulator
      for storing and passing information cleanly between Optimizer and
      Simulator objects (i.e., init dict, if flat, number of conditions,
      which methods to use etc.).

      This is motivated by the fact that
      fitting a single model often involves multiple stages at which this
      information is relevant but non constant.
      """
      def __init__(self, is_flat=False, ncond=1, pc_map=None):

            self.pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
            self.flat_pvc=deepcopy(['a', 'tr', 'v', 'xb'])
            self.full_pvc=list(set(self.flat_pvc).intersection(pc_map.keys()))
            self.pc_map=pc_map
            self.is_flat=is_flat
            self.ncond=ncond
            #self.__dict__ = self

      def __getattr__(self, name):
            """ get items using either of the following
            syntaxes: v=self[k]; v=self.x
            """
            if name in self:
                  return self[name]
            else:
                  raise AttributeError("No such attribute: " + name)

      def __setattr__(self, name, value):
            """ set items using either of the following
            syntaxes: self[k]=v; self.x=v
            """
            self[name] = value

      def set_params(self, inits):
            """ store a safe copy of the init params
            and fill ThisFit attr. dict with params
            """
            self.orig_inits = dict(deepcopy(inits))
            for k,v in inits.items():
                  self.__setattr__(k, v)

      def restore_inits(self):
            self.__clear__()
            for k,v in self.orig_inits.items():
                  self.__setattr__(k, v)

      def flat_vectorize_params(self, dt=.001):
            if 'si' in self.keys():
                  self.dx=np.sqrt(self['si']*dt)
            if 'xb' not in p.keys():
                  self['xb']=np.ones(1)
            for pkey in self.pvc_flat:
                  self[pkey]=np.ones(1)*self[pkey]

      def full_vectorize_params(self, dt=.001):
            full_pvc=list(set(self.flat_pvc).intersection(pc_map.keys()))
            if 'si' in self.keys():
                  self.dx=np.sqrt(self['si']*dt)
            if 'xb' not in p.keys():
                  self['xb']=np.ones(self.ncond)
            for pkey in self.pvc:
                  self[pkey]=np.ones(self.ncond)*self[pkey]
            for pkey, pkc in self.pc_map.items():
                  if pkc[0] not in p.keys():
                        self[pkey] = self[pkey]*np.ones(len(pkc))
                  else:
                        self[pkey] = array([self[pc] for pc in pkc]).astype(np.float32)
            return p



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

      def __init__(self, dframes=None, fitparams=None, kind='radd', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, method='nelder', pc_map=None, wts=None, multiopt=False, global_method='basinhopping', *args, **kws):

            self.multiopt=multiopt
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
            if self.fit_on=='average':
                  self.fit_id='AVERAGE'
                  self.yhat, self.fitinfo, self.popt = self.__opt_routine__()
            elif self.fit_on in ['subjects', 'bootstrap']:
                  self.__indx_optimize__(save=save, savepth=savepth)

            return self.yhat, self.fitinfo, self.popt


      def __indx_optimize__(self, save=True, savepth='./'):

            ri=0; nc=self.ncond; nquant=len(self.fitparams['prob'])
            pcols=self.fitinfo.columns

            for i, y in enumerate(self.dat):
                  self.y = y
                  self.fit_id = getid(i)
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

            # p0X: Initials
            p0 = dict(deepcopy(self.inits))
            # p1: STAGE 1 (Initial Simplex)
            yh1, finfo1, p1 = self.gradient_descent(y=self.flat_y, wts=self.flat_wts, inits=p0, is_flat=True)

            # p2: STAGE 2 (BasinHopping)
            p2 = self.__nudge_params__(p1)
            if self.multiopt:
                  p2, fmin = self.basinhopping_multivar(p1)
            else:
                  self.bdata, self.bwts = self.__prep_basin_data__()
                  for pkey in self.pc_map.keys():
                        p2[pkey] = self.basinhopping_univar(p=p2, pkey=pkey)

            # p3: STAGE 3 (Final Simplex)
            yh3, finfo3, p3 = self.gradient_descent(y=self.avg_y, wts=self.avg_wts, inits=p2, is_flat=False)
            return yh3, finfo3, p3


      def basinhopping_multivar(self, p, nsuccess=20, stepsize=.07, interval=10):
            """ uses L-BFGS-B in combination with basinhopping to perform bounded global
             minimization of multivariate model
            """
            fp = self.fitparams
            basin_keys = self.pc_map.keys()
            ncond = len(self.pc_map.values()[0])
            p = self.__nudge_params__(p)

            self.simulator.__prep_global__(method='basinhopping', basin_params=p, basin_keys=basin_keys, is_flat=False)
            xmin, xmax = theta.format_basinhopping_bounds(basin_keys, kind=self.kind, ncond=ncond)
            x = np.hstack(np.hstack([p[pk] for pk in basin_keys])).tolist()
            bounds = map((lambda x: tuple([x[0], x[1]])), zip(xmin, xmax))
            mkwargs = {"method": "L-BFGS-B", "bounds":bounds, 'tol':1.e-3}
            # run basinhopping on simulator.basinhopping_minimizer func
            out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=stepsize, niter_success=nsuccess, minimizer_kwargs=mkwargs, interval=interval, disp=True)
            xopt = out.x
            funcmin = out.fun
            xarr = array([xopt]).reshape(len(basin_keys), self.ncond)
            for i, k in enumerate(basin_keys):
                  p[k]=xarr[i]
            return p, funcmin


      def basinhopping_univar(self, p, pkey):
            """ uses basinhopping to pre-optimize init cond parameters
            to individual conditions to prevent terminating in local minima
            """
            fp = self.fitparams
            nc = fp['ncond']; cols=['pkey', 'popt', 'fun', 'nfev']
            self.simulator.__prep_global__(method='basinhopping', basin_key=pkey)
            mkwargs = {"method":"Nelder-Mead", 'jac':True}
            xbasin = []
            vals = p[pkey]
            for i, x in enumerate(vals):
                  p[pkey] = x
                  self.simulator.basin_params = p
                  self.simulator.y = self.bdata[i]
                  self.simulator.wts = self.bwts[i]
                  out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=.05, minimizer_kwargs=mkwargs, niter_success=20)
                  xbasin.append(out.x[0])
            if self.xbasin!=[]:
                  self.xbasin.extend(xbasin)
            else:
                  self.xbasin = xbasin
            return xbasin


      def global_min(self, inits, method='brute', basin_key=None):
            """ Performs global optimization via basinhopping, brute, or differential evolution
            algorithms.

            basinhopping method is only used to pre-tune conditional parameters after
            flat optimization before entering final simplex routine (optimize_theta).

            brute and differential evolution methods may be applied to the full parameter set
            (using original inits dictionary and pc_map)
            """

            self.simulator.__prep_global__(method=method, basin_key=basin_key)
            if method=='basinhopping':
                  keybasin = self.perform_basinhopping(p=inits, pkey=basin_key)
                  return keybasin

            pfit = list(set(inits.keys()).intersection(self.pnames))
            pbounds, params = self.slice_bounds_global(inits, pfit)
            self.simulator.y=self.y.flatten()
            self.simulator.wts = self.avg_wts
            if method=='brute':
                  self.simulator.wts = self.avg_wts
                  self.simulator.brute_params = pfit
                  self.globalmin = brute(self.simulator.brute_minimizer, pbounds, args=params)
            elif method=='differential_evolution':
                  self.simulator.diffev_params = pfit
                  self.globalmin = differential_evolution(self.simulator.diffevolution_minimizer, pbounds, args=params)

            return self.globalmin


      def gradient_descent(self, y=None, wts=None, inits={}, is_flat=True):
            """ Optimizes parameters following specified parameter
            dependencies on task conditions (i.e. depends_on={param: cond})
            """

            if not hasattr(self, 'simulator'):
                  self.make_simulator()

            fp = self.fitparams
            if y is None:
                  y=self.flat_y
                  wts=self.flat_wts
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
            self.residual = optmod.residual

            log_arrays = {'y':self.simulator.y, 'yhat':yhat, 'wts':wts}
            try:
                  logger(optmod=optmod, finfo=finfo, pdict=popt, depends_on=fp['depends_on'], log_arrays=log_arrays, kind=self.kind, fit_id=self.fit_id, xbasin=self.xbasin, dynamic=self.dynamic)
            except Exception:
                  pass

            return  yhat, finfo, popt
