#!/usr/local/bin/env python
from __future__ import division
import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd import models
from radd.toolbox.messages import logger
from lmfit import Parameters, minimize
from radd.CORE import RADDCore
from scipy.optimize import basinhopping


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

      def __init__(self, dframes=None, fitparams=None, kind='radd', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, method='nelder', pc_map=None, wts=None, dynamic='hyp', multiopt=False, *args, **kws):

            self.data=dframes['data']
            self.dynamic=dynamic
            self.fitparams=fitparams
            self.multiopt=multiopt
            self.kind = kind

            if fit_on=='average':
                  self.avg_y=dframes['avg_y']
                  self.flat_y=dframes['flat_y']

            elif fit_on in ['subjects', 'bootstrap']:
                  self.fits = dframes['fits']
                  self.fitinfo = dframes['fitinfo']
                  self.indx_list = dframes['observed'].index
                  self.dat=dframes['dat']

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
                  self.yhat, self.fitinfo, self.popt = self.__opt_routine__(self.avg_y, fit_id='AVERAGE')
            elif self.fit_on in ['subjects', 'bootstrap']:
                  self.__indx_optimize__(save=save, savepth=savepth)

            return self.yhat, self.fitinfo, self.popt


      def __indx_optimize__(self, save=True, savepth='./'):

            ri=0; nc=self.ncond
            pcols=self.fitinfo.columns
            for i, y in enumerate(self.dat):

                  fit_id = ''.join(["SUBJECT ",  str(self.indx_list[i])])

                  if self.data_style=='re':
                        self.flat_y = y.mean(axis=0)
                  elif self.data_style=='pro':
                        nquant = len(self.fitparams['prob'])
                        flatgo = y[:nc].mean()
                        flatq = y[nc:].reshape(2,nquant).mean(axis=0)
                        self.flat_y = np.hstack([flatgo, flatq])

                  # optimize params iterating over subjects/bootstraps
                  yhat, finfo, popt = self.__opt_routine__(y, fit_id=fit_id)

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



      def __opt_routine__(self, y, fit_id='AVERAGE'):

            p = dict(deepcopy(self.inits))
            fp = self.fitparams

            if not fp['fit_whole_model']:
                  self.simulator.ncond = self.ncond
                  self.simulator.wts = self.wts
                  self.fit_id=''.join([fit_id, ' FULL'])
                  yhat, finfo, popt = self.optimize_theta(y=y, inits=p, is_flat=False)

            else:
                  fit_ids = [''.join([fit_id, ' FLAT']), ''.join([fit_id, ' FULL'])]
                  to_fit = [self.flat_y, y]
                  wts = [fp['flat_wts'], fp['wts']]
                  ncond=[1, self.ncond]
                  flat = [True, False]

            for i, yi in enumerate(to_fit):
                  # set attributes for fit
                  self.simulator.ncond = ncond[i]
                  self.simulator.wts = wts[i]
                  self.fit_id = fit_ids[i]
                  # run parameter, model setup and optimize
                  yhat, finfo, popt = self.optimize_theta(y=yi, inits=p, is_flat=flat[i])
                  p = deepcopy(popt)

                  if self.multiopt and i<1:
                        x = self.perform_basinhopping(p)
                        p[self.pc_map.keys()[0]] = x

            return yhat, finfo, popt



      def perform_basinhopping(self, p):
            """ uses basin hopping to pre-optimize init cond parameters
            to individual conditions to prevent terminating in local minima
            """

            fp = self.fitparams
            self.simulator.__update_pvc__(is_flat=False, basinhopping=True)
            minimizer_kwargs = {"method":"Nelder-Mead", "jac":True}

            def print_fun(x, f, accepted):
                  print("at minimum %.4f accepted %d" % (f, int(accepted)))

            for pkey, pc_list in self.pc_map.items():
                  bump = np.linspace(.98, 1.0, self.ncond)
                  if pkey in ['a', 'tr']:
                        bump = bump[::-1]
                  vals = p[pkey]*bump

            cond_basins = []
            # regroup y vectors into conditions
            if self.kind=='pro':
                  nogo = self.avg_y.flatten()[:self.ncond]
                  cond_data = [np.append(ng, self.flat_y[1:]) for ng in nogo]
            else:
                  cond_data = self.avg_y

            for i, v in enumerate(vals):
                  p[pkey] = v
                  self.simulator.minimize_simulator_params = p
                  self.simulator.y = cond_data[i]
                  basin = basinhopping(self.simulator.basinhopping_minimizer, [v], stepsize=.05, minimizer_kwargs=minimizer_kwargs, niter_success=1, callback=print_fun)
                  cond_basins.append(basin.x[0])

            return np.hstack(cond_basins)


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
                        vals=ip[pkey]*np.ones(self.ncond)
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

            logger(optmod=optmod, finfo=finfo, depends_on=fp['depends_on'], log_arrays=log_arrays, kind=self.kind, fit_id=self.fit_id)

            return  yhat, finfo, popt


      def set_bounds(self, a=(.001, 1.000), tr=(.1, .55), v=(.0001, 4.0000), z=(.001, .900), ssv=(-4.000, -.0001), xb=(.01,10), si=(.001, .2)):

            """ set and return boundaries to limit search space
            of parameter optimization in <optimize_theta>
            """

            if self.dynamic == 'exp':
                  xb = (.01, 10)
            elif self.dynamic == 'hyp':
                  #xb = (.001, .1)
                  xb = (.01, 10)

            if 'irace' in self.kind:
                  ssv=(abs(ssv[1]), abs(ssv[0]))

            bounds = {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'z': z, 'xb':xb, 'si':si}
            return bounds
