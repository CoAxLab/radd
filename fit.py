#!/usr/local/bin/env python
from __future__ import division
import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd import models
from lmfit import Parameters, minimize, fit_report, Minimizer
from radd.CORE import RADDCore

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

      def __init__(self, dframes=None, fitparams=None, kind='radd', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, method='nelder', pc_map=None, wts=None, dynamic='hyp', *args, **kws):

            self.fits=dframes['fits']
            self.fitinfo=dframes['fitinfo']

            self.data=dframes['data']
            self.dynamic=dynamic
            self.fitparams=fitparams

            if fit_on=='average':
                  self.avg_y=dframes['avg_y']
                  self.flat_y=dframes['flat_y']
            elif fit_on in ['subjects', 'bootstrap']:
                  self.indx_list = dframes['observed'].index
                  self.dat=dframes['dat']

            self.method=method
            self.wts=wts
            self.pc_map=pc_map

            self.pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si']
            self.pvc=deepcopy(['a', 'tr', 'v', 'xb'])

            super(Optimizer, self).__init__(kind=kind, data=self.data, fit_on=fit_on, depends_on=depends_on, inits=inits, fit_whole_model=fit_whole_model, niter=niter)


      def optimize_model(self, save=True, savepth='./'):

            if self.fitparams is None:
                  self.set_fitparams()

            self.simulator = models.Simulator(fitparams=self.fitparams, kind=self.kind, inits=self.inits, pc_map=self.pc_map)

            if self.fit_on=='average':
                  yhat, fitinfo, popt = self.__opt_routine__(self.avg_y, fit_id='AVERAGE DATA')
                  return yhat, fitinfo, popt
            else:
                  fits, fitinfo, popt = self.__indx_optimize__(save=save, savepth=savepth)
                  return fits, fitinfo, popt


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

                  # OPTIMIZE IDX MODEL
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

            self.popt=self.fitinfo.mean()



      def __opt_routine__(self, y, fit_id='AVERAGE DATA'):

            p = dict(deepcopy(self.inits))
            fp = self.fitparams

            if not self.fit_flat:
                  self.simulator.ncond = self.ncond
                  self.simulator.wts = self.wts
                  self.fit_id=''.join([fit_id, ' FULL'])
                  yhat, finfo, popt = self.optimize_theta(y=y, inits=p, flat=False)
            else:
                  fit_ids = [''.join([fit_id, ' FLAT']), ''.join([fit_id, ' FULL'])]
                  to_fit = [self.flat_y, y]
                  wts = [fp['flat_wts'], fp['wts']]
                  flat=[1, 0]
                  ncond=[1, self.ncond]

                  for i, yi in enumerate(to_fit):
                        self.simulator.ncond = ncond[i]
                        self.simulator.wts = wts[i]
                        self.fit_id = fit_ids[i]
                        yhat, finfo, popt = self.optimize_theta(y=yi, inits=p, flat=flat[i], )
                        p = deepcopy(popt)

            return yhat, finfo, popt


      def optimize_theta(self, y, inits, flat=False):

            """ Optimizes parameters following specified parameter
            dependencies on task conditions (i.e. depends_on={param: cond})
            """

            self.simulator.y = y.flatten()
            self.simulator.is_flat = flat
            self.simulator.pvc = deepcopy(self.pvc)
            pfit = list(set(inits.keys()).intersection(self.pnames))
            lim = self.set_bounds()
            fp = self.fitparams

            ip = deepcopy(inits)
            theta=Parameters()
            for pkey, pc_list in self.pc_map.items():
                  if flat: break
                  self.simulator.pvc.remove(pkey)
                  pfit.remove(pkey)
                  mn = lim[pkey][0]; mx=lim[pkey][1]
                  d0 = [theta.add(pc, value=ip[pkey], vary=1, min=mn, max=mx) for pc in pc_list]

            p0 = [theta.add(k, value=ip[k], vary=flat, min=lim[k][0], max=lim[k][1]) for k in pfit]
            opt_kws = {'disp':fp['disp'], 'xtol':fp['xtol'], 'ftol':['ftol'], 'maxfev':fp['maxfev']}

            # OPTIMIZE THETA
            optmod = minimize(self.simulator.__cost_fx__, theta, method=self.method, options=opt_kws)

            optp = optmod.params
            finfo = {k:optp[k].value for k in optp.keys()}
            popt = deepcopy(finfo)

            finfo['chi'] = optmod.chisqr
            finfo['rchi'] = optmod.redchi
            finfo['CNVRG'] = optmod.pop('success')
            finfo['nfev'] = optmod.pop('nfev')
            finfo['AIC']=optmod.aic
            finfo['BIC']=optmod.bic
            self.residual = optmod.residual
            yhat = y.flatten() + self.residual

            pkeys = self.depends_on.keys()
            pvals = self.depends_on.values()
            model_id = "MODEL: %s" % self.kind
            dep_id = "%s DEPENDS ON %s" % (pvals[0], str(tuple(pkeys)))
            wts_str = 'wts = array(['+ ', '.join(str(elem)[:6] for elem in self.simulator.wts)+'])'
            
            with open('fit_report.txt', 'a') as f:
                  f.write(str(self.fit_id)+'\n')
                  f.write(str(model_id)+'\n')
                  f.write(str(dep_id)+'\n')
                  f.write(wts_str+'\n\n')
                  f.write(fit_report(optmod, show_correl=False)+'\n\n')
                  f.write('AIC: %.8f' % optmod.aic + '\n')
                  f.write('BIC: %.8f' % optmod.bic + '\n')
                  f.write('chi: %.8f' % optmod.chisqr + '\n')
                  f.write('rchi: %.8f' % optmod.redchi + '\n')
                  f.write('Converged: %s' % finfo['CNVRG'] + '\n')
                  f.write('--'*20+'\n\n')

            return  yhat, finfo, popt



      def set_bounds(self, a=(.001, 1.000), tr=(.001, .550), v=(.0001, 4.0000), z=(.001, .900), ssv=(-4.000, -.0001), xb=(.01,10), si=(.001, .2)):

            """ set and return boundaries to limit search space
            of parameter optimization in <optimize_theta>
            """

            if self.dynamic == 'exp':
                  xb = (.01, 10)
            elif self.dynamic == 'hyp':
                  xb = (.001, .1)

            if 'irace' in self.kind:
                  ssv=(abs(ssv[1]), abs(ssv[0]))

            bounds = {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'z': z, 'xb':xb, 'si':si}
            return bounds
