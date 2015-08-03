#!/usr/local/bin/env python
from __future__ import division
import time
from copy import deepcopy
import numpy as np
from numpy import array
from numpy.random import random_sample as rs
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

            if model != None:
                  self.fitparams = model.fitparams
                  self.inits = model.inits
                  self.kind=model.kind
                  self.pc_map = model.pc_map
            else:
                  self.fitparams = fitparams
                  self.inits=inits
                  self.kind=kind
                  self.pc_map=pc_map


            if prepare:
                  self.prepare_simulator()


      def prepare_simulator(self):

            pdepends = self.fitparams['depends_on'].keys()
            self.pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si']
            self.pvc=deepcopy(['a', 'tr', 'v', 'xb'])

            fp=dict(deepcopy(self.fitparams))
            self.tb=fp['tb']
            self.wts=fp['wts']
            self.ncond=fp['ncond']
            self.ntot=fp['ntrials']
            self.prob=fp['prob']
            self.ssd=fp['ssd']
            self.scale=fp['scale']
            self.dynamic=fp['dynamic']
            self.nssd=len(self.ssd)
            self.nss=int(.5*self.ntot)
            self.lowerb = 0
            self.y=None

            self.__init_analyze_functions__()
            self.__simulate_functions__()


      def __simulate_functions__(self):

            if 'radd' in self.kind:
                  self.sim_fx = self.simulate_radd

            elif 'pro' in self.kind:
                  self.sim_fx = self.simulate_pro
                  self.ntot = int(self.ntot/self.ncond)

            elif 'irace' in self.kind:
                  self.sim_fx = self.simulate_irace



      def __init_analyze_functions__(self):

            prob=self.prob; nss =self.nss;
            ssd=self.ssd; tb = self.tb

            #if self.fitparams['split']=='HML':
            #      self.ziprt=lambda rt: zip([rt[-1],hs(rt[3:-1]),hs(rt[:3])],[tb]*3)
            #elif self.fitparams['split']=='HL':
            #      self.ziprt=lambda rt: zip([hs(rt[3:]),hs(rt[:3])],[tb]*2)

            self.resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=2)*.0005
            self.resp_lo = lambda trace: np.argmax((trace.T<=0).T, axis=3)*.0005
            self.RT = lambda ontime, rbool: ontime[:, None]+(rbool*np.where(rbool==0, np.nan, 1))
            self.fRTQ = lambda zpd: map((lambda x:mq(x[0][x[0]<x[1]], prob)), zpd)


      def vectorize_params(self, p, as_dict=False):

            """ ensures that all parameters are converted to arrays before simulation.

            pc_map is a dict containing parameter names as keys with values
            corresponding to the names given to that parameter in Parameters object
            (see optmize.Optimizer). i.e.

            This extra step is needed to maintain flexibility when
            working with Parameters objects from lmfit which require
            parameters to be extracted by name. In order to fit arbitrarily
            complex models, parameter arrays that represent different conditions
            must be handled inside the optimization routine

            * can also be accessed directly, outside of optimization routine,
            to generate vectorized parameter dictionaries for simulating.

            * Parameters (p[pkey]=pval) that are constant across conditions
            are broadcast as [pval]*n. Conditional parameters are treated as arrays with
            distinct values [pval_1...pval_n], one for each condition.

            pc_map (dict):          keys: conditional parameter names (i.e. 'v')
                                    values: keys + condition names ('v_bsl, v_pnl')

                  |<--- PARAMETERS OBJECT [LMFIT] <-------- [IN]
                  |
                  |---> p = {'v_bsl': V1, 'v_pnl': V2..} --->|
                                                             |
                  |<--- pc_map = {'v': 'v_bsl', 'v_pnl'} <---|
                  |
                  |---> p['v'] = array([VAL1, VAL2]) ----> [OUT]


            pvc (list):             list of non conditional parameter names

            as_dict (bool):         return vect. param dictionaries for simulating

            sim_info (bool):        return drift coeff, ntimepoints

            """

            if 'si' in p.keys():
                  self.dx=np.sqrt(p['si']*self.dt)
            if 'xb' not in p.keys():
                  p['xb']=1.0
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


      def __sample_interactive_ssrt__(loc, sigma):
            nrvs = self.nss*self.ncond
            rvs_shape = (self.nssd, nrvs/self.nssd)
            return (self.ssd + norm.rvs(loc, sigma, nrvs).reshape(rvs_shape).T).T


      def __update_go_process__(self, p):

            Pg = 0.5*(1 + p['v']*self.dx/self.si)
            Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)

            t = np.cumsum([self.dt]*Tg.max())
            if 'x' in self.kind and self.dynamic=='exp':
                  # dynamic bias is exponential
                  self.xtb = array([np.exp(xtb*t) for xtb in p['xb']])
                  self.lowerb = np.ones(len(t))*p['z']
            elif 'x' in self.kind and self.dynamic=='hyp':
                  # dynamic bias is hyperbolic
                  t = np.cumsum(np.ones(Tg.max()))[::-1]
                  self.lowerb = p['z'] + map((lambda x: (.5*x[0])/(1+(x[1]*t))), zip(p['a'],p['xb']))[0]
                  self.xtb = array([np.ones(len(t)) for i in range(self.ncond)])
            else:
                  self.xtb = array([np.ones(len(t)) for i in range(self.ncond)])
                  self.lowerb = np.ones(len(t))*p['z']
            return Pg, Tg


      def __update_stop_process__(self, p):

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


      def simulate_radd(self, p, analyze=True):

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = p['z'] + (self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx,-self.dx).T, axis=2))
            init_ss = array([[DVc[:self.nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
            DVs = init_ss[:, :, :, None]+np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)

            if analyze:
                  return self.analyze_radd(DVg, DVs, p)
            else:
                  return [DVg, DVs]


      def simulate_pro(self, p, analyze=True):

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)

            DVg = self.lowerb[:, None].T+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2))

            if analyze:
                  return self.analyze_pro(DVg, p)
            else:
                  return DVg



      def simulate_irace(self, p, analyze=True):

            p = self.vectorize_params(p)
            Pg, Tg = self.__update_go_process__(p)
            Ps, Ts = self.__update_stop_process__(p)

            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = self.lowerb[:, None].T+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx, -self.dx).T, axis=2))
            init_ss = self.lowerb*np.ones((self.ncond, self.nssd, self.nss))
            DVs = init_ss[:,:,:,None]+np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)
            if analyze:
                  return self.analyze_irace(DVg, DVs, p)
            else:
                  return [DVg, DVs]


      def analyze_radd(self, DVg, DVs, p):

            nss = self.nss; prob = self.prob
            ssd = self.ssd; tb = self.tb
            ncond = self.ncond

            gdec = self.resp_up(DVg, p['a'])
            sdec = self.resp_lo(DVs)
            gort = self.RT(p['tr'], gdec)
            ssrt = self.RT(ssd, sdec)

            ert = gort[:, :nss][:, None] * np.ones_like(ssrt)
            eq = self.fRTQ(zip(ert, ssrt))
            gq = self.fRTQ(zip(gort,[tb]*ncond))
            gac = np.nanmean(np.where(gort<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)

            return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(self.ncond)])


      def analyze_pro(self, DVg, p):

            nss = self.nss; prob = self.prob
            ssd = self.ssd; tb = self.tb
            ncond = self.ncond

            gdec = self.resp_up(DVg, p['a'])
            rt = self.RT(p['tr'], gdec)

            if ncond==1:
                  qrt = mq(rt[rt<tb], prob=prob)
            else:
                  zipped = zip([hs(rt[3:]),hs(rt[:3])],[tb]*2)
                  qrt = hs(self.fRTQ(zipped))

            # Get response and stop accuracy information
            gac = 1-np.mean(np.where(rt<tb, 1, 0), axis=1)
            return hs([gac, qrt])



      def analyze_irace(self, DVg, DVs, p):

            nss = self.nss; prob = self.prob
            ssd = self.ssd; tb = self.tb
            ncond = self.ncond

            gdec = self.resp_up(DVg, p['a'])
            sdec = self.resp_up(DVs, p['a'])
            gort = self.RT(p['tr'], gdec)
            ssrt = self.RT(ssd, sdec)

            ert = gort[:, :nss][:, None] * np.ones_like(ssrt)
            eq = self.fRTQ(zip(ert, ssrt))
            gq = self.fRTQ(zip(gort,[tb]*ncond))
            gac = np.nanmean(np.where(gort<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)

            return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(self.ncond)])


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
            DVg = self.lowerb[:, None].T+(self.xtb[:,None]*np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T<Pg), self.dx,-self.dx).T, axis=2))

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
