#!/usr/local/bin/env python
from __future__ import division
import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from radd.misc.messages import saygo
from radd.fit import Simulator
from radd.RADD import RADDCore
from lmfit import Parameters, minimize, fit_report, Minimizer


class Model(RADDCore):

      """ Main class for instantiating, fitting, and simulating models.
      Inherits from RADDCore parent class (see RADD.py).

      :: Arguments ::

            data (pandas DF):
                  data frame with columns 'idx', 'rt', 'acc', 'ttype', 'response',
                  <Condition Name> declared in depends_on values

            kind (str):
                  declares model type ['radd', 'irace', 'pro']

            inits (dict):
                  dictionary of parameters (v, a, tr, ssv, z) used to initialize model

            depends_on (dict):
                  set parameter dependencies on task conditions
                  (ex. depends_on={'v': 'Condition'})

            fit_on (str):
                  set if model fits 'average', 'subjects', 'bootstrap' data

            fit_whole_model (bool):
                  fit model holding all fixed except depends_on keys
                  or fit model with all free before fitting depends_on keys

            tb (float):
                  timeboundary: time alloted in task for making a response

      """


      def __init__(self, data=pd.DataFrame, kind='radd', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, weight_presp=True, tb=None, *args, **kws):

            self.data=data
            self.weight_presp=weight_presp
            self.is_optimized=False

            super(Model, self).__init__(data=self.data, inits=inits, fit_on=fit_on, depends_on=depends_on, niter=niter, fit_whole_model=fit_whole_model, kind=kind, tb=tb)

            self.prepare_fit()


      def optimize(self, save=True, savepth='./', ntrials=10000, ftol=1.e-4, xtol=1.e-4, maxfev=1000, niter=500, log_fits=True, disp=True, prob=np.array([.1, .3, .5, .7, .9])):
            """ Method to be used for accessing fitting methods in Optimizer class
            see Optimizer method optimize()
            """
            fp = self.set_fitparams(xtol=xtol, ftol=xtol, maxfev=maxfev, ntrials=ntrials, niter=niter, disp=disp, log_fits=log_fits, prob=prob, get_params=True)
            
            self.opt = Optimizer(dframes=self.dframes, fitparams=fp, kind=self.kind, inits=self.inits, depends_on=self.depends_on, fit_on=self.fit_on, wts=self.wts, pc_map=self.pc_map)

            self.fits, self.fitinfo, self.popt = self.opt.optimize_model(save=save, savepth=savepth)


      def simulate(self):

            try:
                  theta=self.popt
                  simulator=self.opt.simulator
            except Exception:
                  theta=self.inits
                  self.set_fitparams()
                  simulator=Simulator(fitparams=self.fitparams, kind=self.kind, inits=theta, pc_map=self.pc_map)

            theta = simulator.vectorize_params(theta, sim_info=False, as_dict=True)

            if self.kind=='radd':
                  dvg, dvs = simulator.simulate_radd(theta)
                  yhat = simulator.analyze_radd(dvg, dvs, theta)
            elif self.kind=='pro':
                  dvg = simulator.simulate_pro(theta)
                  yhat = simulator.analyze_pro(dvg, theta)
            if self.kind=='irace':
                  dvg, dvs = simulator.simulate_irace(theta)
                  yhat = simulator.analyze_irace(dvg, dvs, theta)

            return yhat


      def prepare_fit(self):

            if 'trial_type' in self.data.columns:
                  self.data.rename(columns={'trial_type':'ttype'}, inplace=True)

            # if numeric, sort first then convert to string
            if not isinstance(self.labels[0], str):
                  ints = sorted([int(l*100) for l in self.labels])
                  self.labels = [str(intl) for intl in ints]
            else:
                  self.labels = sorted(self.labels)

            params = sorted(self.inits.keys())
            self.pc_map = {}
            for d in self.depends_on.keys():
                  params.remove(d)
                  params_dep = ['_'.join([d, l]) for l in self.labels]
                  self.pc_map[d] = params_dep
                  params.extend(params_dep)
            qp_cols = self.__get_header__(params)
            self.__make_dataframes__(qp_cols)
            self.get_wts()
            self.is_prepared=saygo(depends_on=self.depends_on, labels=self.labels, kind=self.kind, fit_on=self.fit_on)



class Optimizer(RADDCore):

      """ Optimizer class acts as interface between Model and Simulator (see fit.py) objects.
      Structures fitting routines so that Models are first optimized with the full set of
      parameters free, data collapsing across conditions.

      The fitted parameters are then used as the initial parameters for fitting conditional
      models with only a subset of parameters are left free to vary across levels of a given
      experimental condition.

      Parameter dependencies are specified when initializing Model object via
      <depends_on> arg (i.e.{parameter: condition})

      Handles fitting routines for models of average, individual subject, and bootstrapped data
      """


      def __init__(self, dframes=None, kind='radd', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, method='nelder', pc_map=None, wts=None, fitparams=None, *args, **kws):

            self.fits=dframes['fits']
            self.fitinfo=dframes['fitinfo']
            self.data=dframes['data']
            self.fitparams=fitparams

            if fit_on=='average':
                  self.avg_y=dframes['avg_y']
                  self.flat_y=dframes['flat_y']
            elif fit_on in ['subjects', 'bootstrap']:
                  self.dat=dframes['dat']

            self.method=method
            self.wts=wts
            self.pc_map=pc_map

            super(Optimizer, self).__init__(kind=kind, data=self.data, fit_on=fit_on, depends_on=depends_on, inits=inits, fit_whole_model=fit_whole_model, niter=niter)


      def optimize_model(self, save=True, savepth='./'):

            if self.fitparams is None:
                  self.set_fitparams()

            self.simulator = Simulator(fitparams=self.fitparams, kind=self.kind, inits=self.inits, method=self.method, pc_map=self.pc_map)

            if self.fit_on=='average':
                  yhat, fitinfo, popt = self.__opt_routine__(self.avg_y)
                  return yhat, fitinfo, popt
            else:
                  fits, fitinfo, popt = self.__indx_optimize__(save=save, savepth=savepth, fitparams=fp)
                  return fits, fitinfo, popt


      def optimize_theta(self, y, inits, flat=False):

            """
            Optimizes parameters following specified parameter
            dependencies on task conditions (i.e. depends_on={param: cond})
            """

            self.simulator.y = y.flatten()
            self.simulator.set_costfx()

            pnames = deepcopy(self.simulator.pnames)
            lim = self.set_bounds()
            fp = self.fitparams

            ip = deepcopy(inits)
            if self.kind=='irace':
                  ip['ssv']=abs(ip['ssv'])
            elif self.kind=='radd':
                  ip['ssv']=-abs(ip['ssv'])

            theta=Parameters()
            for pkey, pc_list in self.pc_map.items():
                  if flat: break
                  self.simulator.pvc.remove(pkey)
                  pnames.remove(pkey)
                  mn = lim[pkey][0]; mx=lim[pkey][1]
                  d0 = [theta.add(pc, value=ip[pkey], vary=1, min=mn, max=mx) for pc in pc_list]

            p0 = [theta.add(k, value=ip[k], vary=flat, min=lim[k][0], max=lim[k][1]) for k in pnames]
            opt_kws = {'disp':fp['disp'], 'xtol':fp['xtol'], 'ftol':['ftol'], 'maxfev':fp['maxfev']}

            optmod = minimize(self.simulator.costfx, theta, method=self.method, options=opt_kws)

            optp = optmod.params
            finfo = {k:optp[k].value for k in optp.keys()}
            popt = deepcopy(finfo)

            finfo['chi'] = optmod.chisqr
            finfo['rchi'] = optmod.redchi
            finfo['CNVRG'] = optmod.pop('success')
            finfo['nfev'] = optmod.pop('nfev')
            finfo['AIC']=optmod.aic
            finfo['BIC']=optmod.bic

            yhat = (y.flatten() + optmod.residual)#*wts[:len(self.y)]

            if fp['log_fits']:
                  fitid = time.strftime('%H:%M:%S')
                  with open('fit_report.txt', 'a') as f:
                        f.write(str(fitid)+'\n')
                        f.write(fit_report(optmod, show_correl=False)+'\n')
                        f.write('AIC: %.8f' % optmod.aic + '\n')
                        f.write('BIC: %.8f' % optmod.bic + '\n')
                        f.write('chi: %.8f' % optmod.chisqr + '\n')
                        f.write('rchi: %.8f' % optmod.redchi + '\n')
                        f.write('Converged: %s' % finfo['CNVRG'] + '\n')
                        f.write('--'*20+'\n\n')

            return  yhat, finfo, popt



      def set_bounds(self, a=(.001, 1.000), tr=(.001, .550), v=(.0001, 4.0000), z=(.001, .900), ssv=(-4.000, -.0001)):

            """
            set and return boundaries to limit search space
            of parameter optimization in <optimize_theta>
            """
            if self.kind=='irace':
                  ssv=(abs(ssv[1]), abs(ssv[0]))

            bounds = {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'z': z}
            if self.kind=='pro':
                  ssv = bounds.pop('ssv')

            return bounds



      def __indx_optimize__(self, save=True, savepth='./'):

            ri=0; nc=self.ncond
            pcols=self.fitinfo.columns
            for i, y in enumerate(self.dat):

                  if self.kind in ['radd', 'irace']:
                        self.flat_y = y.mean(axis=0)
                  elif self.kind=='pro':
                        nquant = len(self.fitparams['prob'])
                        flatgo = y[:nc].mean(),
                        flatq = y[nc:].reshape(2,nquant).mean(axis=0)
                        self.flat_y = np.hstack([flatgo, flatq])

                  yhat, finfo, popt = self.__opt_routine__(y)
                  self.fitinfo.iloc[i]=pd.Series({pc: finfo[pc] for pc in pcols})
                  if self.kind in ['radd', 'irace']:
                        self.fits.iloc[ri:ri+nc, nc:] = yhat
                        ri+=nc
                  else:
                        self.fits.iloc[i] = yhat
                  if save:
                        self.fits.to_csv(savepth+"fits.csv")
                        self.fitinfo.to_csv(savepth+"fitinfo.csv")

            self.popt=self.fitinfo.mean()



      def __opt_routine__(self, y):

            p = dict(deepcopy(self.inits))
            fp = self.fitparams
            if not self.fit_flat:
                  self.simulator.ncond = self.ncond
                  self.simulator.wts = self.wts
                  yhat, finfo, popt = self.optimize_theta(y=y, inits=p, flat=False)
            else:
                  to_fit = [self.flat_y, y]
                  wts = [fp['flat_wts'], fp['wts']]
                  flat=[1, 0]
                  ncond=[1, self.ncond]
                  for i, yi in enumerate(to_fit):
                        self.simulator.ncond = ncond[i]
                        self.simulator.wts = wts[i]
                        yhat, finfo, popt = self.optimize_theta(y=yi, inits=p, flat=flat[i])
                        p = deepcopy(popt)

            return yhat, finfo, popt
