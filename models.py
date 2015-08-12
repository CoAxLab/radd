#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
from numpy import array
from numpy.random import sample as rs
from scipy.stats.mstats import mquantiles as mq
from numpy import hstack as hs


class Simulator(object):
      """ Core code for simulating RADD models

            * All cond, SSD, & timepoints are simulated simultaneously

            * a, tr, and v parameters are initialized as vectors,
            1 x Ncond so Optimizer can optimize a single costfx
            for multiple conditions.
      """

      def __init__(self, model=None, fitparams=None, inits=None, pc_map=None, kind='radd', prepare=True, is_flat=False, is_bold=False):

            self.dt=.0005
            self.si=.01
            self.dx=np.sqrt(self.si*self.dt)
            self.is_flat=is_flat
            self.is_bold=is_bold

            if model:
                  self.fitparams = model.fitparams
                  self.inits = model.inits
                  self.kind=model.kind
                  self.pc_map = model.pc_map
            else:
                  self.fitparams = fitparams
                  self.inits=inits
                  self.kind=kind
                  self.pc_map=pc_map

            self.pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si']
            self.pvc=deepcopy(['a', 'tr', 'v', 'xb'])
            if prepare:
                  self.prepare_simulator()


      def prepare_simulator(self, flat_model=True):

            if not self.is_flat:
                  try:
                        map((lambda pkey: self.pvc.remove(pkey)), self.pc_map.keys())
                  except ValueError:
                        pass

            if not hasattr(self, 'sim_fx'):
                  fp=dict(deepcopy(self.fitparams))
                  self.tb=fp['tb']; self.wts=fp['wts']; self.ncond=fp['ncond']
                  self.ntot=fp['ntrials']; self.prob=fp['prob']; self.ssd=fp['ssd']
                  self.dynamic=fp['dynamic']; self.nssd=len(self.ssd)
                  self.nss=int(.5*self.ntot); self.base = 0; self.y=None

                  self.__init_model_functions__()
                  self.__init_analyze_functions__()


      def __init_model_functions__(self):
            """ initiates the simulation function used in
            optimization routine
            """
            nss=self.nss; ssd=self.ssd; ncond=self.ncond; nssd=self.nssd

            if 'radd' in self.kind:
                  self.sim_fx = self.simulate_reactive
                  self.analyze_fx = self.analyze_reactive
                  self.get_ssbase = lambda Ts,Tg,DVg: array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])[:,:,:,None]

            elif 'pro' in self.kind:
                  self.sim_fx = self.simulate_proactive
                  self.analyze_fx = self.analyze_proactive
                  self.ntot = int(self.ntot/self.ncond)

            elif 'irace' in self.kind:
                  self.sim_fx = self.simulate_reactive
                  self.analyze_fx = self.analyze_reactive
                  self.get_ssbase = lambda x,y,DVg: DVg[0,0,0]*np.ones((ncond, nssd, nss))[:,:,:,None]

            if self.is_bold:
                  self.get_ssbase = lambda Ts,Tg,DVg: array([[DVc[:nss/nssd, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])[:,:,:,None]



      def __init_analyze_functions__(self):
            """ initiates the analysis function used in
            optimization routine to produce the yhat vector
            """

            prob=self.prob; nss =self.nss;
            ssd=self.ssd; tb = self.tb
            #GO RT as SINGLE FUNC
            #go_rt = np.where(dvg.max(axis=2)>=p['a'][:,None], p['tr'][:,None]+np.argmax((dvg.T>=p['a']).T, axis=2)*.0005, 999)
            #go prob = np.where(go_rt<self.tb, 1, 0).mean(axis=1)

            self.get_rt = lambda x: np.where(x[0].max(axis=1)>=x[1], x[2]+np.argmax(x[0]>=x[1], axis=1)*self.dt, 999)
            self.get_ssrt = lambda dvs, delay: np.where(dvs.min(axis=3)<=0, delay+np.argmax(dvs<=0, axis=3)*self.dt, 999)
            self.get_resp = lambda x: x[0][np.where(x[1]<=x[2], 1, 0)]

            # INIT RESPONSE FUNCTIONS
            self.go_resp = lambda trace, a: np.argmax((trace.T>=a).T, axis=2)*.0005
            self.ss_resp = lambda trace, xxx: np.argmax((trace.T<=0).T, axis=3)*.0005

            if 'irace' in self.kind:
                  self.ss_resp = lambda trace, a: np.argmax((trace.T>=a).T, axis=3)*.0005

            # SET RT FUNCTIONS
            self.RT = lambda ontime, rbool: ontime[:, None]+(rbool*np.where(rbool==0, np.nan, 1))
            self.fRTQ = lambda zpd: map((lambda x:mq(x[0][x[0]<x[1]], prob)), zpd)

            # SET STATIC/DYNAMIC BASIS FUNCTIONS
            self.get_t = lambda tg: np.cumsum([self.dt]*tg.max())
            self.get_base = lambda z, a, xb: z*np.ones((self.ncond, len(self.t)))
            self.get_xtb = lambda xxx: np.ones((self.ncond, len(self.t)))

            # dynamic bias is hyperbolic
            if 'x' in self.kind and self.dynamic=='hyp':
                  self.get_t = lambda tg: np.cumsum(np.ones(tg.max()))[::-1]
                  self.get_base = lambda z, a, xb: z + (.5*a[:, None])/(1+(xb[:,None]*self.t))
            # dynamic bias is exponential
            if 'x' in self.kind and self.dynamic=='exp':
                  self.get_xtb = lambda xb: np.exp(xb[:,None]*self.t)


      def vectorize_params(self, p):
            """ ensures that all parameters are converted to arrays before simulation.

            pc_map is a dict containing parameter names as keys with values
            corresponding to the names given to that parameter in Parameters object
            (see optmize.Optimizer). i.e.

            * Parameters (p[pkey]=pval) that are constant across conditions
            are broadcast as [pval]*n. Conditional parameters are treated as arrays with
            distinct values [V1, V2...Vn], one for each condition.

            pc_map (dict):          keys: conditional parameter names (i.e. 'v')
                                    values: keys + condition names ('v_bsl, v_pnl')

                  |<--- PARAMETERS OBJECT [LMFIT] <-------- [IN]
                  |
                  |---> p = {'v_bsl': V1, 'v_pnl': V2..} --->|
                                                             |
                  |<--- pc_map = {'v': 'v_bsl', 'v_pnl'} <---|
                  |
                  |---> p['v'] = array([V1, V2]) --------> [OUT]

            ::Arguments::
                  p (dict):
                        keys are parameter names (e.g. ['v', 'a', 'tr' ... ])
                        values are parameter values, can be vectors or floats
            ::Returns::
                  p (dict):
                        same dictionary as input with all parameters as vectors

            """

            #if 'si' in p.keys():
            #      self.dx=np.sqrt(p['si']*self.dt)
            if 'xb' not in p.keys():
                  p['xb'] = 1
            if 'z' not in p.keys():
                  p['z'] = 0

            for pkey in self.pvc:
                  p[pkey]=np.ones(self.ncond)*p[pkey]

            for pkey, pkc in self.pc_map.items():
                  if self.ncond==1:
                        break
                  elif pkc[0] not in p.keys():
                        p[pkey] = p[pkey]*np.ones(len(pkc))
                  else:
                        p[pkey] = array([p[pc] for pc in pkc])

            return p


      def __sample_interactive_ssrt__(self, loc, sigma):
            """ sample SSRT from gaussian distribution

            ::Arguments::
                  loc (float):
                        mean of distribution
                  sigma (float):
                        sd of distribution
            ::Returns::
                  samples (array) from dist. (nssd, nss)
            """

            nrvs = self.nss*self.ncond
            rvs_shape = (self.nssd, nrvs/self.nssd)
            return (self.ssd + norm.rvs(loc, sigma, nrvs).reshape(rvs_shape).T).T


      def __update_params__(self, p):

            Pg = 0.5*(1 + p['v']*self.dx/self.si)
            Ps = 0.5*(1 + p['ssv']*self.dx/self.si)
            Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)
            Ts = np.ceil((self.tb-self.ssd)/self.dt).astype(int)

            return Pg, Ps, Tg, Ts

      def __update_go_process__(self, p):
            """ calculate go process params (Pg, Tg)
            and hyperbolic or exponential dynamic bias
            across time (exp/hyp specified in dynamic attr.)

            ::Arguments::
                  p (dict):
                        parameter dictionary
            ::Returns::
                  Pg (array):
                        Probability DVg(t)=+dx
                  Tg (array):
                        nTimepoints tr --> tb
            """

            Pg = 0.5*(1 + p['v']*self.dx/self.si)
            Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)
            self.t=self.get_t(Tg)
            self.base = self.get_base(p['z'], p['a'], p['xb'])
            self.xtb = self.get_xtb(p['xb'])
            return Pg, Tg


      def __update_stop_process__(self, p):
            """ calculate stop process params (Ps, Ts)

            ::Arguments::
                  p (dict):
                        parameter dictionary
            ::Returns::
                  Ps (array):
                        Probability DVs(t)=+dx
                  Tg (array):
                        nTimepoints ssd[i] --> tb
            """

            Ps = 0.5*(1 + p['ssv']*self.dx/self.si)
            if self.kind=='interactive':
                  Ts = np.ceil((self.tb-(self.ssd + p['toff']))/self.dt).astype(int)
            else:
                  Ts = np.ceil((self.tb-self.ssd)/self.dt).astype(int)

            return Ps, Ts


      def __cost_fx__(self, theta):
            """ Main cost function used for fitting all models self.sim_fx
            determines which model is simulated (determined when Simulator
            is initiated)
            """

            if type(theta)==dict:
                  p = dict(deepcopy(theta))
            else:
                  p = theta.valuesdict()

            yhat = self.sim_fx(p, analyze=True)
            cost = yhat-self.y
            wtd_cost = cost*self.wts#[:len(self.y)]
            return wtd_cost

      def simulate_reactive(self, p, analyze=True):

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = self.base[:, None]+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx,-self.dx).T, axis=2))
            DVs = self.get_ssbase(Ts,Tg,DVg) + np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)

            if analyze:
                  return self.analyze_reactive(DVg, DVs, p)
            else:
                  return [DVg, DVs]


      def simulate_proactive(self, p, analyze=True):

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)

            DVg = self.base[:, None]+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2))

            if analyze:
                  return self.analyze_proactive(DVg, p)
            else:
                  return DVg


      def simulate_irace(self, p, analyze=True):

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = self.base[:, None].T+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx, -self.dx).T, axis=2))
            init_ss = self.base*np.ones((self.ncond, self.nssd, self.nss))
            DVs = init_ss[:,:,:,None]+np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)
            if analyze:
                  return self.analyze_irace(DVg, DVs, p)
            else:
                  return [DVg, DVs]


      def analyze_reactive(self, DVg, DVs, p):

            nss = self.nss; prob = self.prob
            ssd = self.ssd; tb = self.tb
            ncond = self.ncond

            gdec = self.go_resp(DVg, p['a'])
            sdec = self.ss_resp(DVs, p['a'])
            gort = self.RT(p['tr'], gdec)
            ssrt = self.RT(ssd, sdec)

            ert = gort[:, :nss][:, None] * np.ones_like(ssrt)
            eq = self.fRTQ(zip(ert, ssrt))
            gq = self.fRTQ(zip(gort,[tb]*ncond))
            gac = np.nanmean(np.where(gort<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
            return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])


      def analyze_proactive(self, DVg, p):

            prob = self.prob; tb = self.tb
            ncond = self.ncond
            #gdec = self.go_resp(DVg, p['a'])
            #rt = self.RT(p['tr'], gdec)
            #if self.ncond==1:
            #      qrt = mq(rt[rt<tb], prob=prob)
            #else:
            #      zipped = zip([hs(rt[3:]),hs(rt[:3])],[tb]*2)
            #      qrt = hs(self.fRTQ(zipped))

            rt = (p['tr']+(np.where((DVg.max(axis=2).T>=p['a']).T, np.argmax((DVg.T>=p['a']).T,axis=2)*self.dt, np.nan).T)).T
            if ncond==1:
                  qrt = mq(rt[rt<tb], prob=prob)
            else:
                  hi = np.nanmean(rt[ncond/2:], axis=0)
                  lo = np.nanmean(rt[:ncond/2], axis=0)
                  hilo = [hi[~np.isnan(hi)], lo[~np.isnan(lo)]]
                  # compute RT quantiles for correct and error resp.
                  qrt = np.hstack([mq(rtc[rtc<tb], prob=prob) for rtc in hilo])

            # Get response and stop accuracy information
            gac = 1-np.mean(np.where(rt<tb, 1, 0), axis=1)
            #return gq, eq, gac, sacc
            #if return_traces:
            #      return DVg, DVs
            return np.hstack([gac, qrt])

            # Get response and stop accuracy information
            #gac = 1-np.mean(np.where(rt<tb, 1, 0), axis=1)
            #return hs([gac, qrt])


      def mean_pgo_rts(self, p, return_vals=True):
            """ Simulate proactive model and calculate mean RTs
            for all conditions rather than collapse across high and low
            """
            import pandas as pd
            tb = self.tb; ncond = self.ncond

            DVg = self.simulate_pro(p, analyze=False)
            gdec = self.resp_up(DVg, p['a'])

            rt = self.RT(p['tr'], gdec)
            mu = np.nanmean(rt, axis=1)
            ci = pd.DataFrame(rt.T).sem()*1.96
            std = pd.DataFrame(rt.T).std()

            self.pgo_rts = {'mu': mu, 'ci': ci, 'std':std}
            if return_vals:
                  return self.pgo_rts


      def simulate_interactive(self, p, analyze=True):

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = self.base[:, None].T+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx,-self.dx).T, axis=2))

            allcond_interacted = []
            for i, DVc in enumerate(DVg):
                  interacted = []
                  for toff in Ts:
                        DVc[:self.nss, toff:] = DVc[:self.nss, toff:] + np.cumsum(np.where((rs(self.ncond, self.ntot, Tg.max()-toff)).T<Ps), self.dx, -self.dx)
                        interacted.append(DVc)
                  allcond_interacted.append(interacted)

            DVg_interacted = np.append(allcond_interacted)

            if analyze:
                  return self.analyze_interactive(DVg_interacted, p)
            else:
                  return DVg_interacted



def RADD(model, ncond=2, prob=([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=10000, tb=0.650, dt=.0005, si=.01, return_traces=False):
      """

      Main code for simulating Reactive RADD model

      Simulates all Conditions, SSD, trials, timepoints simultaneously.
      Vectorized operations are set up so that any of the parameters can be
      a single float or a vector of floats (i.e., when simulating/fitting multiple
      conditions differentiated by the value of one or more model parameters)

      args:
            p (dict):                           model parameters [a, tr, v, ssv, z]
            ssd  (array):                       full set of stop signal delays
            nss  (int):                         number of stop trials
            ntot (int):                         number of total trials
            tb (float):                         time boundary
            ncond (int):                        number of conditions to simulate

      returns:

            DVg (Go Process):             3d array for all conditions, trials, timepoints
                                          (i.e. DVg = [nCOND [NTrials [NTime]]] )
                                          All conditions are simulated simultaneously (i.e., BSL & PNL)

            DVs (Stop Process):           4d array for all conditions, SSD, SS trials, timepoints.
                                          i.e. ( DVs = [COND [SSD [nSSTrials [NTime]]]] )
                                          All ss decision traces are initiated from DVg[Cond](t=SSD | SSD<tr)
      """
      model.make_simulator()
      sim = model.simulator

      nss = sim.nss; ntot=sim.ntot;

      dx=np.sqrt(si*dt)

      p = sim.vectorize_params(model.inits)
      #Pg, Tg = sim.__update_go_process__(p)
      #Ps, Ts = sim.__update_stop_process__(p)

      a, tr, v, ssv, z = p['a'], p['tr'], p['v'], p['ssv'], p['z']

      #Pg = 0.5*(1 + v*dx/si)
      #Ps = 0.5*(1 + ssv*dx/si)
      #Tg = np.ceil((tb-tr)/dt).astype(int)
      #Ts = np.ceil((tb-ssd)/dt).astype(int)
      Pg, Ps, Tg, Ts = sim.__update_params__(p)

      # a/tr/v Bias: ALL CONDITIONS, ALL SSD
      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
      init_ss = array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)

      grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt,np.nan)
      ert = ertx[:, None]*np.ones_like(ssrt)

      #collapse across SSD and get average ssrt vector for each condition
      # compute RT quantiles for correct and error resp.
      gq = np.vstack([mq(rtc[rtc<tb], prob=prob) for rtc in grt])
      eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob) for i in range(sim.ncond)]
      # Get response and stop accuracy information
      gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
      sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
      #return gq, eq, gac, sacc
      if return_traces:
            return DVg, DVs

      return np.hstack([np.hstack([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])



def proRADD(p, ncond=6, pGo=np.arange(.2,1.2,.2), prob=([.1, .3, .5, .7, .9]), ssd=.45, ntot=2000, tb=0.545, dt=.0005, si=.01, return_traces=False, style='DDM'):
      """

      main code for simulating Proactive RADD model

      args:
            p (dict):                           model parameters [a, tr, v, ssv, z]
            ssd  (array):                       full set of stop signal delays
            ntot (int):                         number of total trials
            tb (float):                         time boundary
            ncond (int):                        number of conditions to simulate

      returns:

            DVg (Go Process):             3d array for all conditions, trials, timepoints
                                          (i.e. DVg = [nCOND [NTrials [NTime]]] )
                                          All conditions are simulated simultaneously (i.e., BSL & PNL)

            DVs (Stop Process):           4d array for all conditions, SSD, SS trials, timepoints.
                                          i.e. ( DVs = [COND [SSD [nSSTrials [NTime]]]] )
                                          All ss decision traces are initiated from DVg[Cond](t=SSD | SSD<tr)
      """

      dx=np.sqrt(si*dt)

      a, tr, v, ssv, z = p['a'], p['tr'], p['v'], p['ssv'], p['z']

      if np.ndim(tr)==0:
            tr=np.ones(ncond)*tr
      if np.ndim(a)==0:
            a=np.ones(ncond)*a
      if np.ndim(v)==0:
            v=np.ones(ncond)*v
      if np.ndim(ssd)==0:
            ssd = np.ones(ncond)*ssd

      nssd = len(ssd); nss = int(.5*ntot)

      Pg = 0.5*(1 + v*dx/si)
      Ps = 0.5*(1 + ssv*dx/si)
      Tg = np.ceil((tb-tr)/dt).astype(int)
      Ts = np.ceil((tb-ssd)/dt).astype(int)

      # a/tr/v Bias: ALL CONDITIONS
      DVg = z + np.cumsum(np.where((rs((ncond, int(ntot/ncond), Tg.max())).T < Pg), dx, -dx).T, axis=2)
      grt = (tr+(np.where((DVg.max(axis=2).T>=a).T, np.argmax((DVg.T>=a).T,axis=2)*dt,np.nan).T)).T

      hi = np.nanmean(grt[:ncond/2], axis=0)
      lo = np.nanmean(grt[ncond/2:], axis=0)

      hilo = [hi[~np.isnan(hi)], lo[~np.isnan(lo)]]

      # compute RT quantiles for correct and error resp.
      gq = np.hstack([mq(rtc[rtc<tb], prob=prob)*10 for rtc in hilo])
      # Get response and stop accuracy information
      gac = 1-np.mean(np.where(grt<tb, 1, 0), axis=1)
      #return gq, eq, gac, sacc
      if return_traces:
            return DVg, DVs
      return np.hstack([gac, gq])
