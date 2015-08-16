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
from radd.toolbox.messages import logger, global_logger
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
                  self.y=dframes['y'].flatten()
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
                  self.fits, self.fitinfo, self.popt = self.__opt_routine__()
                  # write params (Series) and fit arrays (DF)
                  self.params_io(self.popt, io='w', iostr=savepth+'p')
                  self.fits_io(self.fits, io='w', iostr=savepth+'fits')

            elif self.fit_on in ['subjects', 'bootstrap']:
                  self.__indx_optimize__(save=save, savepth=savepth)

            return self.fits, self.fitinfo, self.popt


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


      def __hop_around__(self, niter=40, nsuccess=20):
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

            inits = self.inits.keys()
            bnd = theta.get_bounds(kind=self.kind, tb=self.fitparams['tb'])
            random_inits = {pkey: theta.init_distributions(pkey, bnd[pkey], nrvs=niter) for pkey in inits}

            xpopt, xfmin = [], []
            for i in range(niter):
                  p={pkey: random_inits[pkey][i] for pkey in inits}
                  popt, fmin = self.perform_basinhopping(p=p, is_flat=True, nsuccess=nsuccess)
                  yhat = self.simulator.sim_fx(popt)
                  cost = self.simulator.__cost_fx__(popt)
                  log_arrays={'popt':popt, 'fmin':fmin, 'cost':cost, 'yhat':yhat}
                  global_logger(log_arrays)
                  xpopt.append(popt)
                  xfmin.append(fmin)

            # get the best fitting set of params
            contender = xpopt[np.argmin(xfmin)]
            # test against original inits and return better
            p1 = self.__test_global__(contender, nsuccess=nsuccess)
            return p1 #return xpopt, xfmin


      def __test_global__(self, popt, nsuccess=10):
            """ test that global optimization worked, evaluating
            costfx using basinhopping optimized parameter set against
            an initial set of parameters provided to Model
            """
            cost = lambda x: m.opt.simulator.__cost_fx__(x)

            m.opt.simulator.ncond=1;
            m.opt.simulator.wts=fp['flat_wts']
            m.opt.simulator.y=self.flat_y

            yh0 = [cost(popt) for i in range(nsuccess)]
            yh1 = [cost(self.inits) for i in range(nsuccess)]
            tests = np.array([1 if c0<c1 else 0 for c0,c1 in zip(yh0, yh1)])

            if np.sum(tests)>=(nsuccess/2.0):
                  print "great success, borat"
                  print tests[np.argmin(tests)]
                  return popt
            else:
                  print "this tie is blacknot"
                  return self.inits


      def __opt_routine__(self):
            """ main function for running optimization routine through all phases
            (global minimum using stochastic search, gradient descent on flat cost fx,
            pre-tuning conditional parameters with basinhopping alg., and polish with
            final gradient descent)
            """

            fp = self.fitparams
            # p0: Initials
            p0 = dict(deepcopy(self.inits))

            # p1: STAGE 1 (Find Global Min/BasinHopping)
            p1 = self.__hop_around__()

            # p1: STAGE 2 (Flat Simplex)
            self.simulator.ncond=1; self.simulator.wts=fp['flat_wts']
            yh2, finfo2, p2 = self.__gradient_descent__(y=self.flat_y, inits=p1, is_flat=True)

            # p2: STAGE 3 (PreTune/BasinHopping)
            p3 = self.__nudge_params__(p2)
            p3, fmin = self.__global_opt__(method='basinhopping', inits=p3)

            # p3: STAGE 4 (Final Simplex)
            self.simulator.ncond=self.ncond; self.simulator.wts=fp['wts']
            yh4, finfo4, p4 = self.__gradient_descent__(y=self.y, inits=p3, is_flat=False)

            return yh4, finfo4, p4


      def perform_basinhopping(self, p, is_flat=False, nsuccess=20, stepsize=.05):
            """ STAGE 1/3 FITTING - GLOBAL MIN: uses basinhopping to
            pre-tune init cond parameters to individual conditions to
            prevent terminating in local minimum
            """
            fp = self.fitparams
            if is_flat:
                  bdata=[self.flat_y]
                  bwts=[fp['flat_wts']]
                  ncond=1
                  basin_keys=p.keys()
                  p={k:array([v]) for k,v in p.items()}
            else:
                  basin_keys=self.pc_map.keys()
                  # get condition wise y vectors
                  bdata, bwts = self.__prep_basin_data__()
                  ncond=self.ncond

            mkwargs = {"method":"Nelder-Mead", 'jac':True}
            self.simulator.__prep_global__(method='basinhopping', basin_keys=basin_keys)

            xopt, funcmin = [], []
            for i in range(ncond):
                  params = {k: v[i] for k,v in p.items()}
                  x=[params[pk] for pk in basin_keys]
                  self.simulator.basin_params = params
                  self.simulator.y = bdata[i]
                  self.simulator.wts = bwts[i]
                  # run basinhopping on simulator.basinhopping_minimizer func
                  out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=stepsize, minimizer_kwargs=mkwargs, niter_success=nsuccess)
                  xopt.append(out.x)
                  funcmin.append(out.fun)

            for i, pk in enumerate(basin_keys):
                  p[pk]=array([xopt[c][i] for c in range(ncond)])
            if self.xbasin!=[]:
                  self.xbasin.extend(funcmin)
            else:
                  self.xbasin = funcmin

            return p, funcmin


      def __global_opt__(self, inits, method='brute', is_flat=False):
            """ Performs global optimization via basinhopping, brute, or differential evolution
            algorithms.

            basinhopping method is used for STAGE 1 fits to find global minimum of flat costfx
            and again at STAGE 3 in order to pre-tune conditional parameters after
            flat optimization before entering final simplex routine (optimize_theta).

            brute and differential evolution methods may be applied to the full parameter set
            (using original inits dictionary and pc_map)
            """

            if method=='basinhopping':
                  keybasin = self.perform_basinhopping(p=inits, is_flat=is_flat)
                  return keybasin

            self.simulator.__prep_global__(method=method)
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


      def __gradient_descent__(self, y, inits, is_flat=True):
            """ STAGE 2/4 FITTING - Flat/Final Simplex: Optimizes parameters
            following specified parameter dependencies on task conditions
            (i.e. depends_on={param: cond})
            """

            self.simulator.y = y.flatten()
            self.simulator.__update_pvc__(is_flat=is_flat)

            fp = self.fitparams
            pnames = deepcopy(self.pnames)
            pfit = list(set(inits.keys()).intersection(pnames))
            lim = theta.get_bounds(kind=self.kind, tb=fp['tb'])

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
