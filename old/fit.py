#!/usr/local/bin/env python
from __future__ import division
import time
from copy import deepcopy
import numpy as np
from lmfit import Parameters, minimize, fit_report, Minimizer
from numpy.random import random_sample as rs
from scipy.stats.mstats import mquantiles as mq



def set_bounds(a=(.001, 5.000), tr=(.001, .650), v=(.0001, 10.0000), z=(.001, .900), ssv=(-10.000, -.0001), style='DDM'):

      """
      set and return boundaries to limit search space
      of parameter optimization in <optimize_theta>
      """
      if style=='IRM':
            ssv=(abs(ssv[1]), abs(ssv[0]))

      return {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'z': z}


def optimize_theta(y, inits={}, pc_map={}, wts=None, ncond=2, pGo=.5, kind='radd', style='DDM', ssd=np.arange(.2, .45, .05), prob=np.array([.1, .3, .5, .7, .9]), ntrials=5000, maxfev=5000, ftol=1.e-3, xtol=1.e-3, disp=True, log_fits=True, tb=.650, method='nelder'):

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

                  model = build.Model(data=pd.DataFrame, inits=param_dict, depends_on={'v': 'Cond'}, kind='radd', prepare=1)
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

      ip = inits.copy()
      lim = set_bounds(style=style)
      if 'pGo' in ip.keys(): del ip['pGo']
      if 'ssd' in ip.keys(): del ip['ssd']

      if style=='IRM':
            ip['ssv']=abs(ip['ssv'])
      elif style=='DDM':
            ip['ssv']=-abs(ip['ssv'])

      popt=Parameters()

      for pkey, pc_list in pc_map.items():
            if ncond==1:
                  vary=1
                  break
            else:
                  vary=0
            bv = ip.pop(pkey)
            mn = lim[pkey][0]; mx = lim[pkey][1]
            d0 = [popt.add(pc, value=bv, vary=True, min=mn, max=mx) for pc in pc_list]

      p0 = [popt.add(k, value=v, vary=vary, min=lim[k][0], max=lim[k][1]) for k, v in ip.items()]
      fcn_kws={'y': y, 'pc_map': pc_map, 'wts': wts, 'ntrials': ntrials, 'ncond': ncond, 'kind':kind, 'prob':prob, 'ssd':ssd, 'tb':tb}
      opt_kws = {'disp':disp, 'xtol':xtol, 'ftol':ftol, 'maxfev': maxfev}

      optmod = minimize(cost_fx, popt, method=method, kws=fcn_kws, options=opt_kws)

      optp = optmod.params
      finfo = {k:optp[k].value for k in optp.keys()}
      fitp = deepcopy(finfo)

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

      yhat = (y.flatten() + optmod.residual)*wts[:len(y)]

      if log_fits:
            fitid = time.strftime('%H:%M:%S')
            with open('fit_report.txt', 'a') as f:
                  f.write(str(fitid)+'\n')
                  f.write(fit_report(optmod, show_correl=False)+'\n')
                  f.write('AIC: %.8f' % optmod.aic + '\n')
                  f.write('BIC: %.8f' % optmod.bic + '\n')
                  f.write('--'*20+'\n\n')

      return finfo, fitp, yhat


def cost_fx(popt, y, pc_map={}, ncond=2, wts=None, ntrials=2000, kind='radd', tb=0.650, ssd=np.arange(.2, .45, .05), prob=np.array([.1, .3, .5, .7, .9]), style='DDM'):

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
      [dx,a,tr,v,ssv,z,ssd,nssd,nss,Pg,Ps,Tg,Ts]
      if type(popt)==dict:
            p = {k:popt[k] for k in popt.keys()}
      else:
            p = popt.valuesdict()

      if ncond>1:
            for pkey, pkc in pc_map.items():
                  p[pkey] = np.array([p[pc] for pc in pkc])

      yhat = RADD(p, prob=prob, ncond=ncond, ssd=ssd, ntot=ntrials, tb=tb)

      c = (y - yhat)*wts[:len(y)]

      return c


def RADD(p, ncond=2, prob=np.array([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=2000, tb=0.650, dt=.0005, si=.01, return_traces=False, style='DDM'):
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

      # a/tr/v Bias: ALL CONDITIONS, ALL SSD
      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
      init_ss = np.array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)

      grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt,np.nan)
      ert = np.array([ertx[i] * np.ones_like(ssrt[i]) for i in range(ncond)])

      #collapse across SSD and get average ssrt vector for each condition
      # compute RT quantiles for correct and error resp.
      gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*10 for rtc in grt])
      eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*10 for i in range(ncond)]
      # Get response and stop accuracy information
      gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
      sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
      #return gq, eq, gac, sacc
      if return_traces:
            return DVg, DVs
      return np.hstack([np.hstack([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])


def proRADD(p, ncond=6, pGo=np.arange(.2,1.2,.2), prob=np.array([.1, .3, .5, .7, .9]), ssd=.45, ntot=2000, tb=0.545, dt=.0005, si=.01, return_traces=False, style='DDM'):
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



def ipRADD(p, ncond=2, prob=np.array([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=2000, tb=0.650, dt=.0005, si=.01, return_traces=False, style='DDM'):
      """

      Main code for simulating Independent Race

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

      dx=np.sqrt(si*dt)

      a, tr, v, ssv, z = p['a'], p['tr'], p['v'], abs(p['ssv']), p['z']

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

      # a/tr/v Bias: ALL CONDITIONS, ALL SSD
      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
      init_ss = np.ones((ncond, nssd, nss))*z
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)

      grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      ert = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      ssrt = ((np.where((DVs.max(axis=3).T>=a).T,ssd[:, None]+np.argmax((DVs.T>=a).T,axis=3)*dt,np.nan).T)).T
      ert = np.array([ert[i] * np.ones_like(ssrt[i]) for i in range(len(ert))])

      #collapse across SSD and get average ssrt vector for each condition
      # compute RT quantiles for correct and error resp.
      gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*10 for i, rtc in enumerate(grt)])
      eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*10 for i in range(ncond)]
      # Get response and stop accuracy information
      gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
      sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)

      if return_traces:
            return DVg, DVs
      return np.hstack([np.hstack([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])
