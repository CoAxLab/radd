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

      def __init__(self, model=None, fitparams=None, inits=None, pc_map=None, kind='radd'):

            self.dt=.0005
            self.si=.01
            self.dx=np.sqrt(self.si*self.dt)

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

            self.__prepare_simulator__()


      def __prepare_simulator__(self):

            fp=dict(deepcopy(self.fitparams))
            self.tb=fp['tb']
            self.wts=fp['wts']
            self.ncond=fp['ncond']
            self.ntot=fp['ntrials']
            self.prob=fp['prob']
            self.ssd=fp['ssd']
            self.dynamic=fp['dynamic']
            self.nssd=len(self.ssd)
            self.nss=int(.5*self.ntot)
            self.rt_cix=fp['rt_cix']

            self.base=0
            self.y=None

            if not hasattr(self, 'pvc'):
                  self.__update_pvc__()

            self.__init_model_functions__()
            self.__init_analyze_functions__()


      def __update_pvc__(self, is_flat=False):

            if self.ncond==1 or is_flat==True:
                  self.pvc=deepcopy(['a', 'tr', 'v', 'xb'])
            else:
                  self.pvc = ['a', 'tr', 'v', 'xb']
                  map((lambda pkey: self.pvc.remove(pkey)), self.pc_map.keys())


      def __prep_global__(self, method='basinhopping', basin_keys=None):

            if method=='basinhopping':
                  if not isinstance(basin_keys, list):
                        basin_keys = [basin_keys]
                  self.basin_keys=basin_keys
                  self.basin_params = {}
                  self.ncond = 1

            elif  method=='differential_evolution':
                  self.ncond = len(self.pc_map.values()[0])
                  self.diffev_params=[]

            elif method=='brute':
                  self.ncond = len(self.pc_map.values()[0])
                  self.brute_params=[]

            self.__update_pvc__()


      def __init_model_functions__(self):
            """ initiates the simulation function used in
            optimization routine
            """

            if 'radd' in self.kind:
                  self.sim_fx = self.simulate_radd
                  self.analyze_fx = self.analyze_radd
                  #self.get_ssbase = lambda Ts,Tg,DVg: array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])[:,:,:,None]

            elif 'pro' in self.kind:
                  self.sim_fx = self.simulate_pro
                  self.analyze_fx = self.analyze_pro
                  self.ntot = int(self.ntot/self.ncond)

            elif 'irace' in self.kind:
                  self.sim_fx = self.simulate_irace
                  self.analyze_fx = self.analyze_irace
                  #self.get_ssbase = lambda x,y,DVg: DVg[0,0,0]*np.ones((ncond, nssd, nss))[:,:,:,None]

            # SET STATIC/DYNAMIC BASIS FUNCTIONS
            self.temporal_dynamics = lambda p, t: np.ones((self.ncond, len(t)))

            if 'x' in self.kind and self.dynamic=='hyp':
                  # dynamic bias is hyperbolic
                  self.temporal_dynamics = lambda p, t: np.cosh(p['xb'][:,None]*t)

            elif 'x' in self.kind and self.dynamic=='exp':
                  # dynamic bias is exponential
                  self.temporal_dynamics = lambda p, t: np.exp(p['xb'][:,None]*t)


      def __init_analyze_functions__(self):
            """ initiates the analysis function used in
            optimization routine to produce the yhat vector
            """

            prob=self.prob; nss=self.nss;
            ssd=self.ssd; tb = self.tb

            self.resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=2)*.0005
            self.resp_lo = lambda trace: np.argmax((trace.T<=0).T, axis=3)*.0005

            self.RT = lambda ontime, rbool: ontime[:, None]+(rbool*np.where(rbool==0, np.nan, 1))
            self.RTQ = lambda zpd: map((lambda x:mq(x[0][x[0]<x[1]], prob)), zpd)
            #self.ss_resp = lambda trace, a: np.argmax((trace.T>=a).T, axis=3)*.0005


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
                        #if pkey in ['a', 'tr']:
                        #      pkc = pkc[::-1]
                        p[pkey] = array([p[pc] for pc in pkc]).astype(np.float32)

            if 'z' in p.keys():
                  self.base = p['z']

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
            """ update Pg (probability of DVg +dx) and Tg (num go process timepoints)
            for go process and get get dynamic bias signal if 'x' model
            """
            Pg = 0.5*(1 + p['v']*self.dx/self.si)
            Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)
            t = np.cumsum([self.dt]*Tg.max())
            self.xtb = self.temporal_dynamics(p, t)
            return Pg, Tg


      def __update_stop_process__(self, p):
            """ update Ps (probability of DVs +dx) and Ts (num ss process timepoints)
            for stop process
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
            return (yhat - self.y)*self.wts[:len(self.y)].astype(np.float32)


      def simulate_radd(self, p, analyze=True):
            """ Simulate the dependent process model (RADD)

            ::Arguments::
                  p (dict):
                        parameter dictionary. values can be single floats
                        or vectors where each element is the value of that
                        parameter for a given condition
                  analyze (bool <True>):
                        if True (default) return rt and accuracy information
                        (yhat in cost fx). If False, return Go and Stop proc.
            ::Returns::
                  yhat of cost vector (ndarray)
                  or Go & Stop processes in list (list of ndarrays)
            """

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            DVg = self.base+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx,-self.dx).T, axis=2))
            # INITIALIZE DVs FROM DVg(t=SSD)
            init_ss = array([[DVc[:self.nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
            DVs = init_ss[:, :, :, None]+np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)

            if analyze:
                  return self.analyze_radd(DVg, DVs, p)
            else:
                  return [DVg, DVs]



      def simulate_pro(self, p, analyze=True):
            """ Simulate the proactive competition model
            (see simulate_radd() for I/O details)
            """

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)

            DVg = self.base+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2))

            if analyze:
                  return self.analyze_pro(DVg, p)
            else:
                  return DVg


      def simulate_irace(self, p, analyze=True):
            """ Simulate the independent race model
            (see simulate_radd() for I/O details)
            """

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = self.base[:, None].T+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx, -self.dx).T, axis=2))
            DVs = self.base[:, None].T + np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)
            if analyze:
                  return self.analyze_irace(DVg, DVs, p)
            else:
                  return [DVg, DVs]


      def analyze_radd(self, DVg, DVs, p):
            """ get rt and accuracy of go and stop process for simulated
            conditions generated from simulate_radd
            """

            nss = self.nss; prob = self.prob
            ssd = self.ssd; tb = self.tb
            ncond = self.ncond

            gdec = self.resp_up(DVg, p['a'])
            sdec = self.resp_lo(DVs)
            gort = self.RT(p['tr'], gdec)
            ssrt = self.RT(ssd, sdec)

            ert = gort[:, :nss][:, None] * np.ones_like(ssrt)
            eq = self.RTQ(zip(ert, ssrt))
            gq = self.RTQ(zip(gort,[tb]*ncond))
            gac = np.nanmean(np.where(gort<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)

            return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(self.ncond)])


      def analyze_pro(self, DVg, p):
            """ get proactive rt and accuracy of go process for simulated
            conditions generated from simulate_pro
            """
            prob=self.prob;  ssd=self.ssd;
            tb=self.tb;  ncond=self.ncond;
            ix=self.rt_cix;

            gdec = self.resp_up(DVg, p['a'])
            rt = self.RT(p['tr'], gdec)

            if self.ncond==1:
                  qrt = mq(rt[rt<tb], prob=prob)
            else:
                  zpd = zip([hs(rt[ix:]), hs(rt[:ix])], [tb]*2)
                  qrt = hs(self.RTQ(zpd))
            # Get response and stop accuracy information
            gac = 1-np.mean(np.where(rt<tb, 1, 0), axis=1)

            return hs([gac, qrt])


      def diffevolution_minimizer(self, z, *params):
            """ find global mininum using differential evolution

            ::Arguments::
                  z (list):
                        list of slice objects or tuples
                        boundaries for each parameter
                  *params:
                        iterable of parameter point estimates
            ::Returns::
                  weighted cost
            """

            p = {pkey: params[i] for i, pkey in enumerate(self.diffev_params)}
            yhat = self.sim_fx(p, analyze=True)
            cost = (yhat - self.y)*self.wts
            return cost.flatten()

      def brute_minimizer(self, z, *params):
            """ find global mininum using brute force
            (see differential_evolution for I/O details)
            """

            p = {pkey: params[i] for i, pkey in enumerate(self.brute_params)}
            yhat = self.sim_fx(p, analyze=True)
            cost = (yhat - self.y)*self.wts
            return cost.flatten()


      def basinhopping_minimizer(self, x):
            """ used specifically by fit.perform_basinhopping() for Model
            objects with multiopt attr.

            parameters are pre-optimized to individual conditions (ncond=1) between
            flat and final conditional parameter fits.

            assigns parameter vector prior to entering this function.
            the "x" argument is a list containing a single float which is
            passed by scipy.basinhopping(). The residual is then passed
            to scipy.fmin() and minimized with simplex.

            Each optimized parameter is stored in a vector which is then
            used to initiate conditional parameters in the final
            fitting routine.  (See __opt_routine__ and perform_basinhopping
            methods of Optimizer object)
            """

            p = self.basin_params
            for i, k in enumerate(self.basin_keys):
                  p[k] = x[i]
            #for pkey in self.pvc:
            #      p[pkey]=np.ones(self.ncond)*p[pkey]

            #Pg, Tg = self.__update_go_process__(p)
            yhat = self.sim_fx(p)

            cost = ((yhat-self.y)*self.wts)**2
            return cost.flatten()



      def analyze_irace(self, DVg, DVs, p):
            """ get rt and accuracy of go and stop process for simulated
            conditions generated from simulate_radd
            """
            dt=self.dt; nss=self.nss; ncond=self.ncond; ssd=self.ssd;
            tb=self.tb; prob=self.prob; scale=self.scale; a=p['a']; tr=p['tr']

            grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ssrt = ((np.where((DVs.max(axis=3).T>=a).T,ssd[:, None]+np.argmax((DVs.T>=a).T,axis=3)*dt,np.nan).T)).T

            # compute RT quantiles for correct and error resp.
            ert = array([ertx[i] * np.ones_like(ssrt[i]) for i in range(ncond)])
            gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*scale for rtc in grt])
            eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*scale for i in range(ncond)]
            # Get response and stop accuracy information
            gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
            return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])


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

      return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])



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
      gq = hs([mq(rtc[rtc<tb], prob=prob)*10 for rtc in hilo])
      # Get response and stop accuracy information
      gac = 1-np.mean(np.where(grt<tb, 1, 0), axis=1)
      #return gq, eq, gac, sacc
      if return_traces:
            return DVg, DVs
      return hs([gac, gq])
