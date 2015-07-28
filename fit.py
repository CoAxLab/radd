#!/usr/local/bin/env python
from __future__ import division
import time
from copy import deepcopy
import numpy as np
from lmfit import Parameters, minimize, fit_report, Minimizer
from numpy.random import random_sample as rs
from scipy.stats.mstats import mquantiles as mq


class Simulator(object):

      """
      Core code for simulating RADD models

            * All Cond and SSD are simulated simultaneously

            * a, tr, and v parameters can be initialized as
                  vectors , 1 x Ncond so that optimize_theta()
                  fits the entire model all at once.

            * optimize_theta returns AIC, BIC, or Chi2 values for the full
                  model fit, allowing different models to be
                  compared with standard complexity penalized
                  goodness-of-fit metrics
      """

      def __init__(self, fitparams=None, inits=None, pc_map=None, kind='reactive', style='RADD', si=.01, dt=.0005,  method='nelder'):

            if fitparams!=None:
                  self.fitparams=fitparams
                  fp=dict(deepcopy(self.fitparams))
                  self.tb=fp['tb']
                  self.log_fits=fp['log_fits']
                  self.disp=fp['disp']
                  self.wts=fp['wts']
                  self.flat_wts = fp['flat_wts']
                  self.ncond=fp['ncond']
                  self.ntot=fp['ntrials']
                  self.prob=fp['prob']
                  self.ssd=fp['ssd']
                  self.xtol=fp['xtol']
                  self.ftol=fp['ftol']
                  self.maxfev=fp['maxfev']
                  self.nssd = len(self.ssd);
                  self.nss = int(.5*self.ntot)

            self.dt=dt
            self.si=si
            self.dx=np.sqrt(si*dt)

            self.style=style
            self.inits=inits
            self.kind=kind
            self.method=method
            self.pc_map=pc_map
            self.pvectors=['a', 'tr', 'v']
            self.pvc=deepcopy(self.pvectors)
            self.flat=False

      def set_bounds(self, a=(.001, 1.000), tr=(.001, .550), v=(.0001, 4.0000), z=(.001, .900), ssv=(-4.000, -.0001)):

            """
            set and return boundaries to limit search space
            of parameter optimization in <optimize_theta>
            """
            if self.style=='IP':
                  ssv=(abs(ssv[1]), abs(ssv[0]))

            return {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'z': z}


      def optimize_theta(self, y, inits, flat=False):

            """
            The main function for optimizing parameters of reactive stop signal model.
            Based on the parameters provided and parameters names included in "bias" list
            this function will minimize a weighted cost function (see recost) comparing observed
            and simulated Go accuracy, Stop accuracy (for each SSD condition), reaction time
            quantiles (.10q, .30q, .50q, .70q, .90q) for correct and error responses for a set
            of experimental conditions.

            Reccommended use of this function is by initiating a build.Model object and executing
            the fit_model() method.

                  Example:

                        model = build.Model(data=pd.DataFrame, inits=param_dict, depends_on={'v': 'Cond'}, kind='reactive', prepare=1)
                        model.fit_model(*args, **kwargs)

            Based on specified parameter dependencies on task conditions (i.e. depends_on={param: cond})
            in build.Model, the bias list will be populated with parameter ids ('a', 'tr', or 'v'). These id's will determine how the lmfit Parameters() class is populated, containing free parameters for each of <ncond> levels of cond i.e. bias=['v']; ncond=3; p=Parameters(); p['v0', 'v1', 'v3'] = inits['v']

            When fitting, all instances of a bias parameter are initialized at the same value provided in inits dict.  The fitting routine will optimize each separately since each condition is simulated separately based on each of the <ncond> parameter id's in the Parameters() object, producing distinct vectors of the Go process, go rt, err rt, stop curve, etc.. (all values included in the cost function are represented separately for observed conditions and simulated conditions)

            args:

                  y (np.array [nCondx16):             observed values entered into cost fx
                                                      see build.Model for format info

                  inits (dict):                       parameter dictionary including
                                                      keys: a, tr, v, ssv, z

                  bias (list):                        list containing parameter names that have
                                                      dependencies task conditions being simulated
                                                      can include a, tr, and/or v.

                  ncond (int):                        number of conditions; determines how many
                                                      instances of parameter id in bias list
                                                      are included in lmfit Parameters() object

                  wts (np.array [2x10])               weights to be applied (separately) to
                                                      correct and error RT quantiles. Can be estimated
                                                      using get_wts() method of build.Model object

            """

            self.y = y.flatten()
            self.pvc=deepcopy(self.pvectors)
            self.flat=flat
            lim = self.set_bounds()

            if self.wts is None:
                  self.wts=np.ones_like(self.y)

            if self.flat:
                  wts = self.flat_wts
                  free = True
            else:
                  wts = self.wts
                  free = False

            if inits!=None:
                  ip = deepcopy(inits)
            else:
                  ip = deepcopy(self.inits)

            if 'pGo' in ip.keys(): del ip['pGo']
            if 'ssd' in ip.keys(): del ip['ssd']

            if self.style=='IP':
                  ip['ssv']=abs(ip['ssv'])
            elif self.style=='RADD':
                  ip['ssv']=-abs(ip['ssv'])

            theta=Parameters();
            for pkey, pc_list in self.pc_map.items():
                  if self.flat: break
                  self.pvc.remove(pkey)
                  bv = ip.pop(pkey)
                  mn = lim[pkey][0]; mx = lim[pkey][1]
                  d0 = [theta.add(pc, value=bv, vary=True, min=mn, max=mx) for pc in pc_list]

            p0 = [theta.add(k, value=v, vary=self.flat, min=lim[k][0], max=lim[k][1]) for k, v in ip.items()]
            opt_kws = {'disp':self.disp, 'xtol':self.xtol, 'ftol':self.ftol, 'maxfev':self.maxfev}

            optmod = minimize(self.cost_fx, theta, method=self.method, kws={'wts':wts}, options=opt_kws)

            optp = optmod.params
            finfo = {k:optp[k].value for k in optp.keys()}
            popt = deepcopy(finfo)

            finfo['chi'] = optmod.chisqr
            finfo['rchi'] = optmod.redchi
            finfo['CNVRG'] = optmod.pop('success')
            finfo['nfev'] = optmod.pop('nfev')
            try:
                  finfo['AIC']=optmod.aic
                  finfo['BIC']=optmod.bic
            except Exception:
                  finfo['AIC']=1000.0
                  finfo['BIC']=1000.0

            yhat = (self.y + optmod.residual)#*wts[:len(self.y)]

            if self.log_fits:
                  fitid = time.strftime('%H:%M:%S')
                  with open('fit_report.txt', 'a') as f:
                        f.write(str(fitid)+'\n')
                        f.write(fit_report(optmod, show_correl=False)+'\n')
                        f.write('AIC: %.8f' % optmod.aic + '\n')
                        f.write('BIC: %.8f' % optmod.bic + '\n')
                        f.write('--'*20+'\n\n')

            return  yhat, finfo, popt



      def cost_fx(self, theta, wts=None):

            """
            simulate data via <simulate_full> and return weighted
            cost between observed (y) and simulated values (yhat).

            returned vector is implicitly used by lmfit minimize
            routine invoked in <optimize_theta> which then submits the
            SSE of the already weighted cost to a Nelder-Mead Simplex
            optimization.


            args:
                  theta (dict):           param dict
                  y (np.array):           NCond x 16 array of observed
                                          values entered into cost f(x)
                  wts (np.array)          weights separately applied to
                                          correct and error RT quantile
                                          comparison
            returns:
                  cost:                   weighted difference bw
                                          observed (y) and simulated (yhat)
            """

            if type(theta)==dict:
                  p = {k:theta[k] for k in theta.keys()}
            else:
                  p = theta.valuesdict()

            if self.kind=='proactive':
                  dvg, dvs = self.core_radd(p)
                  yhat = self.analyze_proactive(dvg, p)

            elif self.kind=='reactive':
                  if self.style=='RADD':
                        dvg, dvs = self.core_radd(p)
                  elif self.style=='IP':
                        dvg, dvs = self.ip_radd(p)
                  yhat = self.analyze_reactive(dvg, dvs, p)

            wtd_res = (self.y - yhat)*wts[:len(self.y)]

            return wtd_res


      def vectorize_params(self, p, sim_info=True, as_dict=False):

            for fp in self.pvc:
                  p[fp]=np.ones(self.ncond)*p[fp]

            for pkey, pkc in self.pc_map.items():
                  if self.ncond==1:
                        break
                  elif pkc[0] not in p.keys():
                        p[pkey] = p[pkey]*np.ones(len(pkc))
                  else:
                        p[pkey] = np.array([p[pc] for pc in pkc])
            if as_dict:
                  allp = deepcopy(self.pvectors)
                  allp.extend(['ssv', 'z'])
                  return {k: p[k][0] if k in self.pvc else p[k] for k in allp}

            out = [p['a'][0], p['tr'][0], p['v'][0], p['ssv'], p['z']]

            if sim_info:
                  Pg = 0.5*(1 + p['v']*self.dx/self.si)
                  Ps = 0.5*(1 + p['ssv']*self.dx/self.si)
                  Tg = np.ceil((self.tb-p['tr'])/self.dt).astype(int)
                  Ts = np.ceil((self.tb-self.ssd)/self.dt).astype(int)
                  out.extend([Pg, Ps, Tg, Ts])

            return out


      def core_radd(self, p=None):

            if p is None:
                  p=self.inits
            a, tr, v, ssv, z, Pg, Ps, Tg, Ts = self.vectorize_params(p)
            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = z + np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2)
            init_ss = np.array([[DVc[:self.nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
            DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)

            return DVg, DVs


      def pro_radd(self, p=None):

            if p is None:
                  p=self.inits
            a, tr, v, ssv, z, Pg, Ps, Tg, Ts = self.vectorize_params(p)
            ntrials = int(self.ntot/self.ncond)
            # a/tr/v Bias: ALL CONDITIONS
            DVg = z + np.cumsum(np.where((rs((self.ncond, ntrials, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2)

            return DVg


      def ip_radd(self, p=None):

            if p is None:
                  p=self.inits
            a, tr, v, ssv, z, Pg, Ps, Tg, Ts = self.vectorize_params(p)
            # a/tr/v Bias: ALL CONDITIONS, ALL SSD
            DVg = z + np.cumsum(np.where((rs((self.ncond, self.ntot, Tg.max())).T < Pg), self.dx, -self.dx).T, axis=2)
            init_ss = np.ones((self.ncond, self.nssd, self.nss))*z
            DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((self.nss, Ts.max()))<Ps, self.dx, -self.dx), axis=1)

            return DVg, DVs


      def analyze_reactive(self, DVg, DVs, p=None):

            if p is None:
                  p=self.inits
            a, tr, v, ssv, z = self.vectorize_params(p, sim_info=False)
            dt=self.dt; nss=self.nss; ncond=self.ncond; ssd=self.ssd; tb=self.tb; prob=self.prob

            grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
            ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T

            if self.style=='RADD':
                  ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt,np.nan)
            else:
                  ssrt = ((np.where((DVs.max(axis=3).T>=a).T,ssd[:, None]+np.argmax((DVs.T>=a).T,axis=3)*dt,np.nan).T)).T

            # compute RT quantiles for correct and error resp.
            ert = np.array([ertx[i] * np.ones_like(ssrt[i]) for i in range(ncond)])
            gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*10 for rtc in grt])
            eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*10 for i in range(ncond)]
            # Get response and stop accuracy information
            gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
            sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)

            return np.hstack([np.hstack([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])


      def analyze_proactive(self, DVg, p=None):

            if p is None:
                  p=self.inits
            a, tr, v, ssv, z = self.vectorize_params(p, sim_info=False)
            dt=self.dt; nss=self.nss; ncond=self.ncond; ssd=self.ssd; tb=self.tb; prob=self.prob

            grt = (tr+(np.where((DVg.max(axis=2).T>=a).T, np.argmax((DVg.T>=a).T,axis=2)*dt,np.nan).T)).T
            hi = np.nanmean(grt[:ncond/2], axis=0)
            lo = np.nanmean(grt[ncond/2:], axis=0)
            hilo = [hi[~np.isnan(hi)], lo[~np.isnan(lo)]]
            if self.flat:
                  hilo=[np.hstack(hilo)]
            # compute RT quantiles for correct and error resp.
            gq = np.hstack([mq(rtc[rtc<tb], prob=prob)*10 for rtc in hilo])
            # Get response and stop accuracy information
            gac = 1-np.mean(np.where(grt<tb, 1, 0), axis=1)
            #return gq, eq, gac, sacc
            return np.hstack([gac, gq])
