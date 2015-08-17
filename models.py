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
            self.nss=int(self.ntot*.5)
            #self.nss=int(self.ntot/self.nssd)
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
            else:
                  self.ncond = len(self.pc_map.values()[0])
                  self.global_params=[]

            self.__update_pvc__()


      def __init_model_functions__(self):
            """ initiates the simulation and main analysis function used in
            optimization routine
            """

            if 'radd' in self.kind:
                  self.sim_fx = self.simulate_radd
                  self.analyze_fx = self.analyze_reactive

            elif 'pro' in self.kind:
                  self.sim_fx = self.simulate_pro
                  self.analyze_fx = self.analyze_proactive
                  self.ntot = int(self.ntot/self.ncond)

            elif 'irace' in self.kind:
                  self.sim_fx = self.simulate_irace
                  self.analyze_fx = self.analyze_reactive

            elif 'interact' in self.kind:
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
            """ initiates the RT, go, and stop threshold detection functions
            used in optimization routine to produce the yhat vector
            """

            prob=self.prob; nss=self.nss;
            ssd=self.ssd; tb = self.tb

            self.resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=2)*.0005
            self.ss_resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=3)*.0005
            self.resp_lo = lambda trace: np.argmax((trace.T<=0).T, axis=3)*.0005

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
                  p['xb']=np.ones(self.ncond)

            for pkey in self.pvc:
                  p[pkey]=np.ones(self.ncond)*p[pkey]
            for pkey, pkc in self.pc_map.items():
                  if self.ncond==1:
                        break
                  elif pkc[0] not in p.keys():
                        p[pkey] = p[pkey]*np.ones(len(pkc))
                  else:
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


      def __update_interactive_params__(self, p):

            # add ss interact delay to SSD
            SSO=self.ssd + p['sso']

            Pg = 0.5*(1 + p['v']*self.dx/self.si)
            Ps = 0.5*(1 + p['ssv']*self.dx/self.si)

            Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)
            Ts = np.ceil((self.tb-SSO)/self.dt).astype(int)

            nt = np.max(np.hstack([Tg,Ts]))
            t = np.cumsum([self.dt]*nt)
            self.xtb = self.temporal_dynamics(p, t)

            return Pg, Tg, Ps, Ts, nt


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
                  return self.analyze_reactive(DVg, DVs, p)
            else:
                  return [DVg, DVs]


      def simulate_pro(self, p, analyze=True):
            """ Simulate the proactive competition model
            (see simulate_radd() for I/O details)
            """
            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)

            DVg = self.base+(self.xtb[:,na]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2))

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

            DVg = self.base+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx,-self.dx).T, axis=2))
            # INITIALIZE DVs FROM DVg(t=0)
            DVs = self.base+np.cumsum(np.where(rs((self.ncond, self.nssd, self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=3)

            if analyze:
                  return self.analyze_reactive(DVg, DVs, p)
            else:
                  return [DVg, DVs]


      def simulate_interactive(self, p, analyze=True):
            """ simulates a version of the interactive race model in which
            the stop signal directly inhibits the go process after a given delay
            SSD + SSO (ss onset) (see simulate_radd() for I/O details)
            """
            ncond=self.ncond; ntot=self.ntot; dx=self.dx;
            base=self.base; nss = self.nss; nssd=self.nssd;
            ssd=self.ssd;

            p = self.vectorize_params(p)
            # nt is the maximum n of timepoints np.max(hs([Tg, Ts]))
            Pg, Tg, Ps, Ts, nt = self.__update_interactive_params__(p)
            gomoments = np.where((rs((ncond, ntot, nt)).T<Pg), dx,-dx).T
            ssmoments = np.where(rs((ncond, nssd, int(nss/nssd), nt))<Ps, dx, -dx)

            diff = Tg[:, na] - Ts
            # fill ssmoments[:time to go onset] with zeros for all SSD
            null = [[ssmoments[ci, ssdi, :, :diff[ci, ssdi]].fill(0) for ssdi in range(nssd)] for ci in range(ncond)]
            # fill gomoments[:time to go onset] with zeros
            null = [[gomoments[ci, :nss, :(nt-Tg[ci])].fill(0)] for ci in range(ncond)]

            # accumulate gomoments/ssmoments
            DVg = (base+self.xtb[:,na]*np.cumsum(gomoments, axis=2))
            DVs = base+np.cumsum(ssmoments, axis=3)

            # extract ss trials from full set of go processes and subtract DVs
            ssDVg = DVg[:,:nss,:].reshape(ncond, nssd, int(nss/nssd), nt)
            # ss cancels go process here
            ssDVg = ssDVg - DVs
            # DVg is now only No ss Go Trials
            DVg = DVg[:,nss:,:]

            # NOTE  in contrast w/ RADD/IRM, where full DVg can be
            # used to estimate go RT and Accuracy (i.e. rt<tb)
            # DVg is directly influenced by DVs on ss trials.
            # Here, DVg are all the NoSS Go trials and ssDVg is
            # all the SS-Go trials in which the SS strongly suppresses
            # the Go process at t=SSD+SSO. In the analysis, ssDVg is used
            # to estimate the proportion of Go processes that make it to
            # threshold to get the stop accuracy and Error Go RT
            # DVg is treated as is in RADD and IRM models, except for the fact
            # that it contains half the simulated trials (makes no difference
            # in the calculations of Go RT and Accuracy between models given that
            # these Go Acc. only depends on the DVg reaching thresh. prior to tb
            # and is not influenced by SS. Also, the no. of Go trials simulated
            # is twice the number of SS process trials in RADD and IRM and the
            # mean estimates of the quantiles and accuracy asymptotes well before
            # ntot simulations)

            if analyze:
                  return self.analyze_interactive(DVg, ssDVg, p)
            return [DVg, ssDVg]



      def analyze_reactive(self, DVg, DVs, p):
            """ get rt and accuracy of go and stop process for simulated
            conditions generated from simulate_radd
            """

            nss = self.nss; prob = self.prob
            ssd = self.ssd; tb = self.tb
            ncond = self.ncond

            gdec = self.resp_up(DVg, p['a'])
            if 'radd' in self.kind:
                  sdec = self.resp_lo(DVs)
            elif 'irace' in self.kind:
                  sdec = self.ss_resp_up(DVs, p['a'])

            gort = self.RT(p['tr'], gdec)
            ssrt = self.RT(ssd, sdec)

            ert = gort[:, :nss][:, None] * np.ones_like(ssrt)
            eq = self.RTQ(zip(ert, ssrt))
            gq = self.RTQ(zip(gort,[tb]*ncond))
            gacc = np.nanmean(np.where(gort<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)

            return hs([hs([i[ii] for i in [gacc,sacc,gq,eq]]) for ii in range(self.ncond)])


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

            eq = self.RTQ(zip(ss_gort,[tb]*ncond))
            gq = self.RTQ(zip(gort,[tb]*ncond))
            gacc = np.nanmean(np.where(gort<tb, 1, 0), axis=1)
            sacc = np.where(ss_gort<sscancel[:,na], 0, 1).mean(axis=2)

            return hs([hs([i[ii] for i in [gacc,sacc,gq,eq]]) for ii in range(ncond)])



      def global_minimizer(self, bounds, *params):
            """ find global mininum using minimization algorithm
            specified by Optimizer object (see Optimizer.__global_opt__)
            ::Arguments::
                  z (list):
                        list of slice objects or tuples boundaries for each
                        parameter (not used w/in function, but is automatically
                        passed by brute/differential evol. minimizers)
                  *params:
                        iterable of parameter point estimates
            ::Returns::
                  weighted cost
            """
            p = {pkey: params[i] for i, pkey in enumerate(self.global_params)}
            yhat = self.sim_fx(p, analyze=True)
            cost = (yhat - self.y)*self.wts
            return cost.flatten()


      def basinhopping_minimizer(self, x):
            """ Accessed by Optimizer object which initiates basin_keys (list
            of parameter names to draw from < x >, in order of basin_keys and
            stores values in a parameter dictionary basin_params to be used
            by simulator and analysis functions.
            ::Arguments::
                  x (list):
                        iterable of parameter point estimates
            ::Returns::
                  weighted cost

            """
            p = self.basin_params
            for i, k in enumerate(self.basin_keys):
                  p[k] = x[i]
            yhat = self.sim_fx(p)
            cost = ((yhat-self.y)*self.wts)**2
            return cost.flatten()
