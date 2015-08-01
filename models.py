#!/usr/local/bin/env python
from __future__ import division
import time
from copy import deepcopy
import numpy as np
from numpy.random import random_sample as rs
from scipy.stats.mstats import mquantiles as mq


class Simulator(object):

      """
      Core code for simulating RADD models

            * All cond, SSD, & timepoints are simulated simultaneously

            * a, tr, and v parameters are initialized as vectors,
            1 x Ncond so Optimizer can optimize a single costfx
            for multiple conditions.
      """

      def __init__(self, model=None, fitparams=None, inits=None, pc_map=None, kind='radd', prepare=True, is_flat=False):

            self.dt=.0005
            self.si=.01
            self.dx=np.sqrt(self.si*self.dt)
            self.is_flat=is_flat

            if model != None:
                  self.fitparams = model.fitparams
                  self.inits = model.inits
                  self.kind=model.kind
                  self.pc_map = model.pc_map
            else:
                  self.fitparams=fitparams
                  self.inits=inits
                  self.kind=kind
                  self.pc_map=pc_map

            if prepare:
                  self.prepare_simulator()


      def prepare_simulator(self):

            pdepends = self.fitparams['depends_on'].keys()
            self.pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb','si']
            self.pvc=deepcopy(['a', 'tr', 'v', 'xb'])

            fp=dict(deepcopy(self.fitparams))
            self.tb=fp['tb']
            self.wts=fp['wts']
            self.ncond=fp['ncond']
            self.ntot=fp['ntrials']
            self.prob=fp['prob']
            self.ssd=fp['ssd']
            self.scale=fp['scale']
            self.nssd=len(self.ssd)
            self.nss=int(.5*self.ntot)
            self.lowerb=0
            self.y=None

            if 'radd' in self.kind:
                  self.sim_fx = self.simulate_radd

            elif 'pro' in self.kind:
                  self.sim_fx = self.simulate_pro
                  self.ntot = int(self.ntot/self.ncond)

            elif 'irace' in self.kind:
                  self.sim_fx = self.simulate_irace


      def vectorize_params(self, p, as_dict=False):

            """
            ensures that all parameters are either converted to arrays
            of length ncond.

            * can also be accessed directly, outside of optimization routine,
            to generate vectorized parameter dictionaries for simulating.

            * Parameters (p[pkey]=pval) that are constant across conditions
            are broadcast as [pval]*n. Conditional parameters are treated as arrays with
            distinct values [pval_1...pval_n], one for each condition.

            * caculates drift coefficients (Pg & Ps)
            * calculates number of timepoints from tr/ssd to tb

            pc_map (dict):          keys: conditional parameter names (i.e. 'v')
                                    values: keys + condition names ('v_bsl, v_pnl')

            pvc (list):             list of non conditional parameter names

            as_dict (bool):         return vect. param dictionaries for simulating

            sim_info (bool):        return drift coeff, ntimepoints

            """

            if 'si' in p.keys():
                  self.dx=np.sqrt(p['si']*self.dt)

            if 'xb' not in p.keys():
                  p['xb']=1.0

            for pkey in self.pvc:
                  p[pkey]=np.ones(self.ncond)*p[pkey]

            for pkey, pkc in self.pc_map.items():
                  if self.ncond==1:
                        break
                  elif pkc[0] not in p.keys():
                        p[pkey] = p[pkey]*np.ones(len(pkc))
                  else:
                        p[pkey] = np.array([p[pc] for pc in pkc])

            if 'z' in p.keys():
                  self.lowerb = p['z']

            return p


      def __update_go_process__(self, p):

            Pg = 0.5*(1 + p['v']*self.dx/self.si)
            Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)
            t = np.cumsum([self.dt]*Tg.max())
            if 'x' in self.kind:
                  self.xtb = np.array([np.exp(xtb*t) for xtb in p['xb']])
            else:
                  self.xtb = np.array([np.ones(len(t)) for i in range(self.ncond)])

            return Pg, Tg

      def __update_stop_process__(self, p):

            Ps = 0.5*(1 + p['ssv']*self.dx/self.si)
            Ts = np.ceil((self.tb-self.ssd)/self.dt).astype(int)
            return Ps, Ts


      def cost_fx(self, theta):
            """
            Reactive Model (RADD) cost function
            """

            if type(theta)==dict:
                  p = dict(deepcopy(theta))
            else:
                  p = theta.valuesdict()

            p = self.vectorize_params(p)
            yhat = self.sim_fx(p, analyze=True)

            return (yhat - self.y)*self.wts[:len(self.y)]



      def simulate_radd(self, p, analyze=True):

            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = self.lowerb+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx,-self.dx).T, axis=2))
            init_ss = np.array([[DVc[:self.nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
            DVs = init_ss[:, :, :, None]+np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)

            if analyze:
                  return self.analyze_radd(DVg, DVs, p)
            else:
                  return DVg, DVs


      def simulate_pro(self, p, analyze=True):

            Pg, Tg = self.__update_go_process__(p)

            # a/tr/v Bias: ALL CONDITIONS
            DVg = self.lowerb+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2))

            if analyze:
                  return self.analyze_pro(DVg, p)
            else:
                  return DVg


      def simulate_irace(self, p, analyze=True):

            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = self.lowerb + (self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2))
            init_ss = self.lowerb*np.ones((self.ncond, self.nssd, self.nss))
            DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)
            if analyze:
                  return self.analyze_radd(DVg, DVs p)
            else:
                  return DVg, DVs


      def analyze_radd(self, DVg, DVs, p):

            dt=self.dt; nss=self.nss; ncond=self.ncond; ssd=self.ssd;
            tb=self.tb; prob=self.prob; scale=self.scale; a=p['a']; tr=p['tr']

            grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt,np.nan)

            # compute RT quantiles for correct and error resp.
            ert = np.array([ertx[i] * np.ones_like(ssrt[i]) for i in range(ncond)])
            gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*scale for rtc in grt])
            eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*scale for i in range(ncond)]
            # Get response and stop accuracy information
            gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)

            return np.hstack([np.hstack([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])


      def analyze_pro(self, DVg, p):

            dt=self.dt; ncond=self.ncond; tb=self.tb; prob=self.prob;
            scale=self.scale; a=p['a']; tr=p['tr']

            rt = (tr+(np.where((DVg.max(axis=2).T>=a).T, np.argmax((DVg.T>=a).T,axis=2)*dt, 999).T)).T

            if self.is_flat:
                  qrt = mq(rt[rt<tb], prob=prob)*scale
            else:
                  hi = np.hstack(rt[3:])
                  lo = np.hstack(rt[:3])
                  hilo = [hi[hi<tb], lo[lo<tb]]
                  # compute RT quantiles for correct and error resp.
                  qrt = np.hstack([mq(rti[rti<tb], prob=prob)*scale for rti in hilo])

            # Get response and stop accuracy information
            gac = 1-np.mean(np.where(rt<tb, 1, 0), axis=1)
            return np.hstack([gac, qrt])



      def analyze_irace(self, DVg, DVs, p):

            dt=self.dt; nss=self.nss; ncond=self.ncond; ssd=self.ssd;
            tb=self.tb; prob=self.prob; scale=self.scale; a=p['a']; tr=p['tr']

            grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ssrt = ((np.where((DVs.max(axis=3).T>=a).T,ssd[:, None]+np.argmax((DVs.T>=a).T,axis=3)*dt,np.nan).T)).T

            # compute RT quantiles for correct and error resp.
            ert = np.array([ertx[i] * np.ones_like(ssrt[i]) for i in range(ncond)])
            gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*scale for rtc in grt])
            eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*scale for i in range(ncond)]
            # Get response and stop accuracy information
            gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
            return np.hstack([np.hstack([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])
