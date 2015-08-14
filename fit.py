#!/usr/local/bin/env python
from __future__ import division
import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd import models
from radd.toolbox.messages import logger, basin_accept_fun
from lmfit import Parameters, minimize
from radd.CORE import RADDCore
from scipy.optimize import basinhopping, differential_evolution, brute


class Optimizer(RADDCore):
      """ Optimizer class acts as interface between Model and Simulator (see fit.py) objects.
      Structures fitting routines so that Models are first optimized with the full set of
      parameters free, data collapsing across conditions.

      wayward pines

      The fitted parameters are then used as the initial parameters for fitting conditional
      models with only a subset of parameters are left free to vary across levels of a given
      experimental conditionself.

      Parameter dependencies are specified when initializing Model object via
      <depends_on> arg (i.e.{parameter: condition})

      Handles fitting routines for models of average, individual subject, and bootstrapped data
      """

      def __init__(self, dframes=None, fitparams=None, kind='radd', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, method='nelder', pc_map=None, wts=None, multiopt=True, global_method='basinhopping', *args, **kws):

            self.data=dframes['data']
            self.fitparams=fitparams
            self.multiopt=multiopt
            self.global_method=global_method
            self.kind=kind
            self.xbasin=[]
            self.dynamic=self.fitparams['dynamic']
            nq = len(self.fitparams['prob'])
            nc = self.fitparams['ncond']

            if fit_on=='average':
                  self.y=dframes['avg_y']
                  self.flat_y=dframes['flat_y']

            elif fit_on in ['subjects', 'bootstrap']:
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
            self.wts=wts
            self.pc_map=pc_map
            self.pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si']
            self.pvc=deepcopy(['a', 'tr', 'v', 'xb'])

            super(Optimizer, self).__init__(kind=kind, data=self.data, fit_on=fit_on, depends_on=depends_on, inits=inits, fit_whole_model=fit_whole_model, niter=niter)


      def optimize_model(self, save=True, savepth='./'):

            # initate simulator object of model being optimized
            self.simulator = models.Simulator(fitparams=self.fitparams, kind=self.kind, inits=self.inits, pc_map=self.pc_map)

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

            fp = self.fitparams

            # p0X: Initials
            p0 = dict(deepcopy(self.inits))
            #yh1, finfo1, p1 = self.perform_diffevolution(inits=p0)
            # p1: STAGE 1 (Initial Simplex)
            self.simulator.ncond=1; self.simulator.wts=fp['flat_wts']
            yh1, finfo1, p1 = self.optimize_theta(y=self.flat_y, inits=p0, is_flat=True)

            # p2: STAGE 2 (BasinHopping)
            self.bdata, self.bwts = self.__prep_basin_data__()
            for pkey in self.pc_map.keys():
                  p2 = self.__nudge_params__(p1, pkey)
                  p2[pkey] = self.perform_global_min(method='basinhopping', inits=p2, basin_key=pkey)

            # p3: STAGE 3 (Final Simplex)
            self.simulator.ncond=self.ncond; self.simulator.wts=fp['wts']
            yh3, finfo3, p3 = self.optimize_theta(y=self.y, inits=p2, is_flat=False)

            return yh3, finfo3, p3


      def perform_basinhopping(self, p, pkey):
            """ uses basin hopping to pre-optimize init cond parameters
            to individual conditions to prevent terminating in local minima
            """
            fp = self.fitparams
            nc = fp['ncond']; cols=['pkey', 'popt', 'fun', 'nfev']
            #basindf=pd.DataFrame(np.zeros((nc,4)),columns=cols)
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


      def perform_global_min(self, inits, method='brute', basin_key=None):
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
            self.simulator.wts = self.wts

            if method=='brute':
                  self.simulator.wts = self.wts
                  self.simulator.brute_params = pfit
                  self.globalmin = brute(self.simulator.brute_minimizer, pbounds, args=params)

            elif method=='differential_evolution':
                  self.simulator.diffev_params = pfit
                  self.globalmin = differential_evolution(self.simulator.diffevolution_minimizer, pbounds, args=params)

            return self.globalmin


      def optimize_theta(self, y, inits, is_flat=True):
            """ Optimizes parameters following specified parameter
            dependencies on task conditions (i.e. depends_on={param: cond})
            """

            self.simulator.y = y.flatten()
            self.simulator.__update_pvc__(is_flat=is_flat)

            pnames = deepcopy(self.pnames)
            pfit = list(set(inits.keys()).intersection(pnames))
            lim = self.set_bounds()
            fp = self.fitparams

            ip = deepcopy(inits)
            lmParams=Parameters()

            for pkey, pc_list in self.pc_map.items():
                  if is_flat: break
                  pfit.remove(pkey)
                  if hasattr(ip[pkey], '__iter__'):
                        vals=ip[pkey]
                  else:
                        vals=ip[pkey]*np.ones(len(pc_list))
                  mn = lim[pkey][0]; mx=lim[pkey][1]
                  d0 = [lmParams.add(pc, value=vals[i], vary=1, min=mn, max=mx) for i, pc in enumerate(pc_list)]

            p0 = [lmParams.add(k, value=ip[k], vary=is_flat) for k in pfit]
            opt_kws = {'disp':fp['disp'], 'xtol':fp['tol'], 'ftol':fp['tol'], 'maxfev':fp['maxfev']}

            # OPTIMIZE THETA
            optmod = minimize(self.simulator.__cost_fx__, lmParams, method=self.method, options=opt_kws)

            optp = optmod.params
            finfo = {k:optp[k].value for k in optp.keys()}
            popt = deepcopy(finfo)
            self.residual = optmod.residual
            yhat = self.simulator.y + self.residual
            wts = self.simulator.wts
            log_arrays = {'y':self.simulator.y, 'yhat':yhat, 'wts':wts}

            if is_flat:
                  fitid = ' '.join([self.fit_id, 'FLAT'])
            else:
                  fitid = ' '.join([self.fit_id, 'FULL'])
            logger(optmod=optmod, finfo=finfo, depends_on=fp['depends_on'], log_arrays=log_arrays, kind=self.kind, fit_id=fitid, xbasin=self.xbasin, dynamic=self.dynamic)

            return  yhat, finfo, popt
