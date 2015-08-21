#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import hstack as hs
from numpy import newaxis as na
from scipy.stats.mstats import mquantiles as mq

class Simulator(object):
      """ Core code for simulating RADD models

            * All cond, SSD, & timepoints are simulated simultaneously

            * a, tr, and v parameters are initialized as vectors,
            1 x Ncond so Optimizer can optimize a single costfx
            for multiple conditions.
      """

      def __init__(self, model=None, fitparams=None, inits=None, pc_map=None, kind='radd', dt=.001, si=.01):

            self.dt=dt
            self.si=si
            self.dx=np.sqrt(self.si*self.dt)
            if model:
                  self.fitparams = model.fitparams
                  self.inits = model.inits
                  self.kind = model.kind
                  self.pc_map = model.pc_map
            else:
                  self.fitparams = fitparams
                  self.inits = inits
                  self.kind = kind
                  self.pc_map = pc_map
            self.__prepare_simulator__()


      def __prepare_simulator__(self):

            fp=dict(deepcopy(self.fitparams))
            self.tb=fp['tb']
            self.ncond=fp['ncond']
            self.ntot=fp['ntrials']
            self.prob=fp['prob']
            self.ssd=fp['ssd']
            self.dynamic=fp['dynamic']
            self.nssd=len(self.ssd)
            self.nss=int(.5*self.ntot)
            self.rt_cix=fp['rt_cix']
            self.is_flat=False
            self.base=0
            if not hasattr(self, 'pvc'):
                  self.__update__()

            self.__init_model_functions__()
            self.__init_analyze_functions__()


      def __update__(self, is_flat=False, y=None, wts=None):

            fp = self.fitparams
            if is_flat:
                  self.pvc=deepcopy(['a', 'tr', 'v', 'xb'])
                  if y is None:
                        y=fp['flat_y']
                        wts=fp['flat_wts']
                  self.ncond=1
            else:
                  self.pvc = ['a', 'tr', 'v', 'xb']
                  map((lambda pkey: self.pvc.remove(pkey)), self.pc_map.keys())
                  if y is None:
                        y=fp['avg_y']
                        wts=fp['avg_wts']
                  self.ncond=fp['ncond']

            self.y=y.flatten()
            self.wts=wts.flatten()

      def __prep_global__(self, method='basinhopping', basin_params={}, basin_keys=[], is_flat=False):

            if method=='basinhopping':
                  self.basin_keys=basin_keys
                  self.basin_params=basin_params
            elif  method=='differential_evolution':
                  self.ncond = len(self.pc_map.values()[0])
                  self.diffev_params=[]
            elif method=='brute':
                  self.ncond = len(self.pc_map.values()[0])
                  self.brute_params=[]

            self.__update__(is_flat=is_flat)

      def basinhopping_minimizer(self, x):
            """ used specifically by fit.perform_basinhopping() for Model
            objects with multiopt attr (See __opt_routine__ and perform_basinhopping
            methods of Optimizer object)
            """
            p = self.basin_params

            # segment 'x' into equal len arrays (one array,
            # ncond vals long per free parameter) in basin_keys
            px = [array(x[i::1]) for i in range(1)]

            for i, pk in enumerate(self.basin_keys):
                  p[pk]=px[i]
            yhat = self.sim_fx(p)
            cost = np.sum((self.wts*(yhat-self.y)**2))
            if hasattr(cost, '__iter__'):
                  return cost[0]
            return cost


      def __init_model_functions__(self):
            """ initiates the simulation function used in
            optimization routine
            """
            if any(m in self.kind for m in ['radd', 'sab']):
                  self.sim_fx = self.simulate_radd
                  self.analyze_fx = self.analyze_reactive
            elif 'pro' in self.kind:
                  self.sim_fx = self.simulate_pro
                  self.analyze_fx = self.analyze_proactive
                  self.ntot = int(self.ntot/self.ncond)
            elif 'irace' in self.kind:
                  self.sim_fx = self.simulate_irace
                  self.analyze_fx = self.analyze_reactive
            elif 'iact' in self.kind:
                  self.sim_fx = self.simulate_interactive
                  self.analyze_fx = self.analyze_proactive

            # SET STATIC/DYNAMIC BASIS FUNCTIONS
            self.temporal_dynamics = lambda p, t: np.ones((self.ncond, len(t)))
            if 'x' in self.kind and self.dynamic=='hyp':
                  # dynamic bias is hyperbolic
                  self.temporal_dynamics = lambda p, t: np.cosh(p['xb'][:,na]*t)
            elif 'x' in self.kind and self.dynamic=='exp':
                  # dynamic bias is exponential
                  self.temporal_dynamics = lambda p, t: np.exp(p['xb'][:,na]*t)


      def __init_analyze_functions__(self):
            """ initiates the analysis function used in
            optimization routine to produce the yhat vector
            """

            prob=self.prob; nss=self.nss;
            ssd=self.ssd; tb = self.tb; dt=self.dt

            self.resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=2)*dt
            self.ss_resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=3)*dt
            self.resp_lo = lambda trace: np.argmax((trace.T<=0).T, axis=3)*dt
            self.RT = lambda ontime, rbool: ontime[:,na]+(rbool*np.where(rbool==0, np.nan, 1))
            self.RTQ = lambda zpd: map((lambda x:mq(x[0][x[0]<x[1]], prob)), zpd)


      def vectorize_params(self, p):
            """ ensures that all parameters are converted to arrays before simulation. see
            doc strings for prepare_fit() method of Model class (in build.py) for details
            regarding pc_map and logic for fitting models with parameters that depend on
            experimental conditions

            ::Arguments::
                  p (dict):
                        keys are parameter names (e.g. ['v', 'a', 'tr' ... ])
                        values are parameter values, can be vectors or floats
            ::Returns::
                  p (dict):
                        dictionary with all parameters as vectors
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
                        p[pkey] = array([p[pc] for pc in pkc]).astype(np.float32)
            return p


      def __update_go_process__(self, p):
            """ update Pg (probability of DVg +dx) and Tg (num go process timepoints)
            for go process and get get dynamic bias signal if 'x' model
            """
            Pg = 0.5*(1 + p['v']*self.dx/self.si)
            Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)
            t = np.cumsum([self.dt]*Tg.max())
            self.xtb = self.temporal_dynamics(p, t)
            return Pg, Tg


      def __update_stop_process__(self, p, sso=0):
            """ update Ps (probability of DVs +dx) and Ts (num ss process timepoints)
            for stop process
            """
            if 'sso' in p.keys():
                  sso=p['sso']
            Ps = 0.5*(1 + p['ssv']*self.dx/self.si)
            Ts = np.ceil((self.tb-(self.ssd+sso))/self.dt).astype(int)
            return Ps, Ts

      def __update_interactive_params__(self, p):
            # add ss interact delay to SSD
            Ps, Ts = self.__update_stop_process__(p)
            Pg = 0.5*(1 + p['v']*self.dx/self.si)
            Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)
            nt = np.max(np.hstack([Tg,Ts]))
            t = np.cumsum([self.dt]*nt)
            self.xtb = self.temporal_dynamics(p, t)
            return Pg, Tg, Ps, Ts, nt


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
            return np.sum(self.wts*(self.y-yhat)**2)
            #return (yhat - self.y)*self.wts[:len(self.y)].astype(np.float32)


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
            nssd = self.nssd; nss= self.nss;
            nc = self.ncond; dx=self.dx; ntot=self.ntot
            DVg = self.xtb[:,na]*np.cumsum(np.where((rs((nc, ntot, Tg.max())).T<Pg),dx,-dx).T, axis=2)
            if 'radd' in self.kind:
                  DVg+=p['z']
            # INITIALIZE DVs FROM DVg(t=SSD)
            init_ss = array([[DVg[i,:nss,ix] for ix in np.where(Ts<Tg[i],Tg[i]-Ts,0)] for i in range(nc)])
            DVs = init_ss[:,:,:,None]+np.cumsum(np.where(rs((nss, Ts.max()))<Ps,dx,-dx), axis=1)
            if analyze:
                  return self.analyze_reactive(DVg, DVs, p)
            else:
                  return [DVg, DVs]


      def simulate_pro(self, p, analyze=True):
            """ Simulate the proactive competition model
            (see simulate_radd() for I/O details)
            """
            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            nc = self.ncond; dx=self.dx; ntot=self.ntot

            DVg = self.xtb[:,na]*np.cumsum(np.where((rs((nc,ntot,Tg.max())).T < Pg),dx,-dx).T, axis=2)
            if analyze:
                  return self.analyze_proactive(DVg, p)
            return DVg


      def simulate_irace(self, p, analyze=True):
            """ simulate the independent race model
            (see selfulate_radd() for I/O details)
            """
            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)
            nssd = self.nssd; nss= self.nss;
            nc = self.ncond; dx=self.dx; ntot=self.ntot

            DVg = self.xtb[:,None]*np.cumsum(np.where((rs((nc, ntot, Tg.max())).T<Pg),dx,-dx).T, axis=2)
            # INITIALIZE DVs FROM DVg(t=0)
            DVs = np.cumsum(np.where(rs((nc, nssd, nss, Ts.max()))<Ps, dx,-dx), axis=3)
            if analyze:
                  return self.analyze_reactive(DVg, DVs, p)
            return [DVg, DVs]


      def simulate_interactive(self, p, analyze=True):
            """ simulates a version of the interactive race model in which
            the stop signal directly inhibits the go process after a given delay
            SSD + SSO (ss onset) (see simulate_radd() for I/O details)
            """
            nc=self.ncond; ntot=self.ntot; dx=self.dx;
            nss = self.nss; nssd=self.nssd;
            ssd=self.ssd; nssd_i=int(nss/nssd)

            p = self.vectorize_params(p)
            # nt is the maximum n of timepoints np.max(hs([Tg, Ts]))
            Pg, Tg, Ps, Ts, nt = self.__update_interactive_params__(p)
            gomoments = np.where((rs((nc,ntot,nt)).T<Pg), dx,-dx).T
            ssmoments = np.where(rs((nc,nssd,nssd_i,nt))<Ps, dx, -dx)

            diff = Tg[:, na] - Ts
            # fill ssmoments[:time to go onset] with zeros for all SSD
            null = [[ssmoments[ci,ssdi,:,:diff[ci, ssdi]].fill(0) for ssdi in range(nssd)] for ci in range(nc)]
            # fill gomoments[:time to go onset] with zeros
            null = [[gomoments[ci,:nss,:(nt-Tg[ci])].fill(0)] for ci in range(nc)]

            # accumulate gomoments/ssmoments
            DVg = self.xtb[:,na]*np.cumsum(gomoments, axis=2)
            DVs = np.cumsum(ssmoments, axis=3)

            # extract ss trials from full set of go processes and subtract DVs
            ssDVg = DVg[:,:nss,:].reshape(nc,nssd,nssd_i,nt)
            # ss cancels go process here
            ssDVg = ssDVg - DVs
            # DVg is now only No ss Go Trials
            DVg = DVg[:,nss:,:]
            if analyze:
                  return self.analyze_interactive(DVg, ssDVg, p)
            return [DVg, ssDVg]



      def analyze_reactive(self, DVg, DVs, p):
            """ get rt and accuracy of go and stop process for simulated
            conditions generated from simulate_radd
            """
            nss = self.nss; prob = self.prob
            ssd = self.ssd; tb = self.tb
            nc = self.ncond; nssd=self.nssd

            if 'sso' in p.keys():
                  ssd = ssd + p['sso']
            gdec = self.resp_up(DVg, p['a'])

            if 'irace' in self.kind:
                  sdec = self.ss_resp_up(DVs, p['a'])
            else:
                  sdec = self.resp_lo(DVs)
            gort = self.RT(p['tr'], gdec)
            ssrt = self.RT(ssd, sdec)
            ert = gort[:,:nss][:, None]*np.ones_like(ssrt)

            eq = self.RTQ(zip(ert, ssrt))
            gq = self.RTQ(zip(gort,[tb]*nc))
            gacc = np.nanmean(np.where(gort<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
            return hs([hs([i[ii] for i in [gacc,sacc,gq,eq]]) for ii in range(nc)])


      def analyze_proactive(self, DVg, p):
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
            gacc = 1-np.mean(np.where(rt<tb, 1, 0), axis=1)
            return hs([gacc, qrt])


      def analyze_interactive(self, DVg, ssDVg, p):
            """ get rt and accuracy of go and stop process for selfulated
            conditions generated from selfulate_radd
            """
            nss = self.nss; prob = self.prob; ssd = self.ssd; tb = self.tb;
            ncond=self.ncond; nssd=self.nssd; nss_di = int(nss/nssd)
            sscancel = ssd + p['sso']
            # Go process (No SS Trials)
            gdec = self.resp_up(DVg, p['a'])
            gort = self.RT(p['tr'], gdec)
            # Go process SS Trials
            ssgdec = self.ss_resp_up(ssDVg, p['a'])
            ssgdec = ssgdec.reshape(ncond, nssd*nss_di)
            ss_gort = self.RT(p['tr'], ssgdec).reshape(ncond,nssd,nss_di)
            # ssGo and Go rt quantiles
            eq = self.RTQ(zip(ss_gort,[tb]*ncond))
            gq = self.RTQ(zip(gort,[tb]*ncond))
            gacc = np.nanmean(np.where(gort<tb, 1, 0), axis=1)
            sacc = np.where(ss_gort<sscancel[:,na], 0, 1).mean(axis=2)

            return hs([hs([i[ii] for i in [gacc,sacc,gq,eq]]) for ii in range(ncond)])


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
