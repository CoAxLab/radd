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

      def __init__(self, fitparams=None, inits=None, pc_map=None, kind='radd', si=.01, dt=.0005, method='nelder'):

            if fitparams!=None:
                  self.fitparams=fitparams
                  fp=dict(deepcopy(self.fitparams))
                  self.tb=fp['tb']
                  self.wts=fp['wts']
                  self.ncond=fp['ncond']
                  self.ntot=fp['ntrials']
                  self.prob=fp['prob']
                  self.ssd=fp['ssd']
                  self.nssd = len(self.ssd);
                  self.nss = int(.5*self.ntot)

            self.dt=dt
            self.si=si
            self.dx=np.sqrt(si*dt)

            self.inits=inits
            self.kind=kind
            self.pc_map=pc_map
            self.pnames=['a', 'tr', 'v', 'ssv', 'z']
            self.pvectors=['a', 'tr', 'v']
            if 'pro' in self.kind:
                  null = [self.pnames.remove(i) for i in ['ssv', 'z']]
            if 'x' in self.kind:
                  self.pnames.append('xb')
            self.pvc=deepcopy(self.pvectors)
            self.y=None

      def vectorize_params(self, p, sim_info=True, as_dict=False):

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

            for pkey in self.pvc:
                  p[pkey]=np.ones(self.ncond)*p[pkey]

            for pkey, pkc in self.pc_map.items():
                  if self.ncond==1:
                        break
                  elif pkc[0] not in p.keys():
                        p[pkey] = p[pkey]*np.ones(len(pkc))
                  else:
                        p[pkey] = np.array([p[pc] for pc in pkc])

            if as_dict:
                  return {k: p[k][0] if k in self.pvc else p[k] for k in self.pnames}

            out = [p[k] for k in self.pnames]

            if sim_info:
                  Pg = 0.5*(1 + p['v']*self.dx/self.si)
                  Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)
                  out.extend([Pg, Tg])
                  if self.kind in ['radd', 'irace']:
                        Ps = 0.5*(1 + p['ssv']*self.dx/self.si)
                        Ts = np.ceil((self.tb-self.ssd)/self.dt).astype(int)
                        out.extend([Ps, Ts])
            return out


      def set_bounds(self, a=(.001, 1.000), tr=(.001, .550), v=(.0001, 4.0000), z=(.001, .900), ssv=(-4.000, -.0001), xx=(5,15)):

            """
            set and return boundaries to limit search space
            of parameter optimization in <optimize_theta>
            """
            if kind=='irace':
                  ssv=(abs(ssv[1]), abs(ssv[0]))

            bounds = {'a':a,'tr':tr,'v':v,'ssv':ssv,'z':z,'xn':xx,'xd':xx}
            if 'x' not in self.kind:
                  x = bounds.pop('xn')
                  x = bounds.pop('xd')
            if self.kind=='pro':
                  ssv = bounds.pop('ssv')

            return bounds


      def set_costfx(self):

            if self.kind=='radd':
                  self.costfx = self.radd_costfx
            elif self.kind=='pro':
                  self.costfx = self.pro_costfx
                  self.ntot = int(self.ntot/self.ncond)
            elif self.kind=='xpro':
                  self.costfx = self.xpro_costfx
                  self.ntot = int(self.ntot/self.ncond)
            elif self.kind=='irace':
                  self.costfx = self.irace_costfx

      def radd_costfx(self, theta):
            """
            Reactive Model (RADD) cost function
            """

            if type(theta)==dict:
                  p = {k:theta[k] for k in theta.keys()}
            else:
                  p = theta.valuesdict()
            dvg, dvs = self.simulate_radd(p)
            yhat = self.analyze_radd(dvg, dvs, p)

            return (yhat - self.y)*self.wts[:len(self.y)]


      def pro_costfx(self, theta):
            """
            Proactive model cost function
            """

            if type(theta)==dict:
                  p = {k:theta[k] for k in theta.keys()}
            else:
                  p = theta.valuesdict()
            dvg = self.simulate_pro(p)
            yhat = self.analyze_pro(dvg, p)

            return (yhat - self.y)*self.wts[:len(self.y)]


      def xpro_costfx(self, theta):
            """
            ExProactive model cost function
            """

            if type(theta)==dict:
                  p = {k: theta[k] for k in theta.keys()}
            else:
                  p = theta.valuesdict()
            dvg = self.simulate_xpro(p)
            yhat = self.analyze_pro(dvg, p)

            return (yhat - self.y)*self.wts[:len(self.y)]


      def irace_costfx(self, theta):
            """
            Independent Race Model cost function
            """
            if type(theta)==dict:
                  p = {k:theta[k] for k in theta.keys()}
            else:
                  p = theta.valuesdict()
            dvg, dvs = self.simulate_irace(p)
            yhat = self.analyze_irace(dvg, dvs, p)

            return (yhat - self.y)*self.wts[:len(self.y)]


      def simulate_radd(self, p):

            a, tr, v, ssv, z, Pg, Tg, Ps, Ts = self.vectorize_params(p)
            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = z + np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2)
            init_ss = np.array([[DVc[:self.nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
            DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)
            return DVg, DVs


      def simulate_pro(self, p):

            a, tr, v, Pg, Tg = self.vectorize_params(p)
            # a/tr/v Bias: ALL CONDITIONS
            DVg = np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2)
            return DVg


      def simulate_xpro(self, p):
            a, tr, v, xb, Pg, Tg = self.vectorize_params(p)
            xtbias = np.exp(xb * np.cumsum([self.dt]*Tg.max()))
            # a/tr/v Bias: ALL CONDITIONS
            DVg = xtbias * np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2)
            return DVg


      def simulate_irace(self, p):

            a, tr, v, ssv, z, Pg, Tg, Ps, Ts = self.vectorize_params(p)
            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = z + np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2)
            init_ss = np.ones((self.ncond, self.nssd, self.nss))*z
            DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)
            return DVg, DVs


      def analyze_radd(self, DVg, DVs, p):

            a, tr, v, ssv, z = self.vectorize_params(p, sim_info=False)
            dt=self.dt; nss=self.nss; ncond=self.ncond; ssd=self.ssd; tb=self.tb; prob=self.prob

            grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt,np.nan)

            # compute RT quantiles for correct and error resp.
            ert = np.array([ertx[i] * np.ones_like(ssrt[i]) for i in range(ncond)])
            gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*10 for rtc in grt])
            eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*10 for i in range(ncond)]
            # Get response and stop accuracy information
            gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
            return np.hstack([np.hstack([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])


      def analyze_pro(self, DVg, p):

            a, tr, v, z = self.vectorize_params(p, sim_info=False)
            dt=self.dt; ncond=self.ncond; tb=self.tb; prob=self.prob

            rt = (tr+(np.where((DVg.max(axis=2).T>=a).T, np.argmax((DVg.T>=a).T,axis=2)*dt, 999).T)).T

            if len(self.y)<=(len(prob)+1):
                  gq = mq(rt[rt<tb], prob=prob)*10
            else:
                  hi = np.hstack(rt[ncond/2:])#, axis=0)
                  lo = np.hstack(rt[:ncond/2])#, axis=0)
                  hilo = [hi[hi<tb], lo[lo<tb]]
                  # compute RT quantiles for correct and error resp.
                  gq = np.hstack([mq(rti[rti<tb], prob=prob)*10 for rti in hilo])

            # Get response and stop accuracy information
            gac = 1-np.mean(np.where(rt<tb, 1, 0), axis=1)
            return np.hstack([gac, gq])


      def analyze_irace(self, DVg, DVs, p):

            a, tr, v, ssv, z = self.vectorize_params(p, sim_info=False)
            dt=self.dt; nss=self.nss; ncond=self.ncond; ssd=self.ssd; tb=self.tb; prob=self.prob

            grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ssrt = ((np.where((DVs.max(axis=3).T>=a).T,ssd[:, None]+np.argmax((DVs.T>=a).T,axis=3)*dt,np.nan).T)).T

            # compute RT quantiles for correct and error resp.
            ert = np.array([ertx[i] * np.ones_like(ssrt[i]) for i in range(ncond)])
            gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*10 for rtc in grt])
            eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*10 for i in range(ncond)]
            # Get response and stop accuracy information
            gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
            return np.hstack([np.hstack([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])
