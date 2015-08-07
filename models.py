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

      def __init__(self, model=None, fitparams=None, inits=None, pc_map=None, kind='radd', prepare=True, is_flat=False):

            self.dt=.0005
            self.si=.01
            self.dx=np.sqrt(self.si*self.dt)
            self.is_flat=is_flat

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



      def prepare_simulator(self):

            pdepends = self.fitparams['depends_on'].keys()

            fp=dict(deepcopy(self.fitparams))
            self.tb=fp['tb']; self.wts=fp['wts']; self.ncond=fp['ncond']
            self.ntot=fp['ntrials']; self.prob=fp['prob']; self.ssd=fp['ssd']
            self.scale=fp['scale']; self.dynamic=fp['dynamic']; self.nssd=len(self.ssd)
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


      def __init_analyze_functions__(self):
            """ initiates the analysis function used in
            optimization routine to produce the yhat vector
            """

            prob=self.prob; nss =self.nss;
            ssd=self.ssd; tb = self.tb
            #GO RT as SINGLE FUNC
            #go_rt = np.where(dvg.max(axis=2)>=p['a'][:,None], p['tr'][:,None]+np.argmax((dvg.T>=p['a']).T, axis=2)*.0005, 999)
            #go prob = np.where(go_rt<self.tb, 1, 0).mean(axis=1)

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
            self.get_base = lambda z, xxx: z*np.ones(len(self.t))
            self.get_xtb = lambda tg, xb: array([np.ones(len(self.t)) for i in range(self.ncond)])

            # dynamic bias is hyperbolic
            if 'x' in self.kind and self.dynamic=='hyp':
                  self.get_t = lambda tg: np.cumsum(np.ones(tg.max()))[::-1]
                  self.get_base = lambda z, zpd: z+map((lambda x:(.5*x[0])/(1+(x[1]*self.t))), zpd)[0]

            # dynamic bias is exponential
            if 'x' in self.kind and self.dynamic=='exp':
                  self.get_xtb = lambda tg, xb: array([np.exp(xtb*self.t) for xtb in xb])


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
            self.base = self.get_base(p['z'], zip(p['a'],p['xb']))
            self.xtb = self.get_xtb(Tg, p['xb'])
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
            return (yhat - self.y)*self.wts[:len(self.y)]


      def simulate_reactive(self, p, analyze=True):

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = self.base[:, None].T+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx,-self.dx).T, axis=2))
            DVs = self.get_ssbase(Ts,Tg,DVg) + np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)

            if analyze:
                  return self.analyze_reactive(DVg, DVs, p)
            else:
                  return [DVg, DVs]


      def simulate_proactive(self, p, analyze=True):

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)

            DVg = self.base[:, None].T+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2))

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

            gdec = self.go_resp(DVg, p['a'])
            rt = self.RT(p['tr'], gdec)
            if self.ncond==1:
                  qrt = mq(rt[rt<tb], prob=prob)
            else:
                  zipped = zip([hs(rt[3:]),hs(rt[:3])],[tb]*2)
                  qrt = hs(self.fRTQ(zipped))

            # Get response and stop accuracy information
            gac = 1-np.mean(np.where(rt<tb, 1, 0), axis=1)
            return hs([gac, qrt])


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
