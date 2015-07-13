#!/usr/local/bin/env python
from __future__ import division
import time
import numpy as np
from lmfit import Parameters, minimize, fit_report
from numpy.random import random_sample as rs
from scipy.stats.mstats import mquantiles as mq

"""
Working code for fitting reactive data:
This module is essentially an optimized version of the
core fitting functions in fitre.py and the
model simulations in RADD.py

      * All Cond and SSD are simulated simultaneously

      * a, tr, and v parameters can be initialized as
            vectors , 1 x Ncond so that optimize()
            fits the entire model all at once.

      * optimize returns AIC, BIC, or Chi2 values for the full
            model fit, allowing different models to be
            compared with standard complexity penalized
            goodness-of-fit metrics

RUNTIME TESTS:
------------------------------------------
50 iterations of simulating 10,000 trials:
------------------------------------------
fitre + RADD:
1 loops, best of 3: 1min 32s per loop

fit:
1 loops, best of 3: 12.9 s per loop
-----------------------------------------

------------------------------------------
50 iterations of simulating, analyzing,
storing, and executing cost function,
each sim with 10,000 trials:

1 loops, best of 3: 1min 2s per loop
...on a wimpy macbook air
------------------------------------------
"""

def optimize(y, inits={}, bias=['xx'], wts=None, ncond=1, model='radd', ntrials=5000, maxfev=5000, ftol=1.e-3, xtol=1.e-3, all_params=0, disp=True, fitid=None, log_fits=True, method='nelder'):

      ip = inits.copy()
      lim = set_bounds()
      if 'pGo' in ip.keys(): del ip['pGo']
      if 'ssd' in ip.keys(): del ip['ssd']
      ip['ssv']=-abs(ip['ssv'])

      popt=Parameters()
      if not all_params:
            vary=0
            for bk in bias:
                  bv = ip.pop(bk)
                  mn = lim[bk][0]; mx = lim[bk][1]
                  d0 = [popt.add(bk+str(i), value=bv, vary=1, min=mn, max=mx) for i in range(ncond)]

      else:
            vary=1

      if method=='differential_evolution':
            pass
      else:
            aval = ip.pop('a'); zval = ip.pop('z')
            popt.add('a', value=aval, vary=1, min=lim['a'][0], max=lim['a'][1])
            popt.add('zperc', value=zval/aval, vary=1)
            popt.add('z', expr="zperc*a")

      p0 = [popt.add(k, value=v, vary=vary, min=lim[k][0], max=lim[k][1]) for k, v in ip.items()]

      f_kws = {'wts':wts, 'ncond':ncond, 'ntrials':ntrials}
      opt_kws = {'disp':disp, 'xtol':xtol, 'ftol':ftol, 'maxfev':maxfev}
      optmod = minimize(recost, popt, args=(y, bias), method=method, kws=f_kws, options=opt_kws)

      params = popt.valuesdict()
      params['chi'] = optmod.chisqr
      params['rchi'] = optmod.redchi

      try:
            params['AIC']=optmod.aic
            params['BIC']=optmod.bic

      except Exception:

            params['AIC']=1000.0
            params['BIC']=1000.0

      yhat = np.vstack(y) + optmod.residual

      if log_fits:
            if fitid is None:
                  fitid = time.strftime('%H:%M:%S')
            with open('fit_report.txt', 'a') as f:
                  f.write(str(fitid)+'\n')
                  f.write(fit_report(optmod, show_correl=False)+'\n')
                  f.write('AIC: %.8f' % optmod.aic)
                  f.write('BIC: %.8f' % optmod.bic)
                  f.write('--'*20+'\n\n')

      return params, yhat


def recost(theta, y, bias=['v'], ntrials=2000, wts=None):

      p = {k:theta[k] for k in theta.keys()}

      ssd=np.arange(.2, .45, .05);
      prob=np.array([.1, .3, .5, .7, .9])

      cond = {pk: p.pop(pk) for pk in p.keys() if pk[-1].isdigit()}
      ncond = len(cond.keys())
      for i in range(ncond):
            p[cond.keys()[i][:-1]] = np.array(cond.values())

      if 'tr' not in bias:
            p['tr'] = np.array([p['tr']]*2)

      yhat = simulate_full(p['a'], p['tr'], p['v'], -abs(p['ssv']),  p['z'], prob=prob, ncond=ncond, ssd=ssd, ntot=ntrials)
      #wtc, wte = wts[0], wts[1]
      #y=np.vstack(y)
      #cost = np.vstack(y) - yhat
      #cost = np.hstack([y[:6]-yhat[:6], wtc*y[6:11]-wtc*yhat[6:11], wte*y[11:]-wte*yhat[11:]]).astype(np.float32)

      return yhat


def set_bounds(a=(.01, .6), tr=(.001, .5), v=(.01, 4.), z=(.001, .9), ssv=(-4., -.01)):

      return {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'z': z}



def simulate_full(a, tr, v, ssv, z, analyze=True, ncond=1, prob=np.array([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=2000, tb=0.650, dt=.0005, si=.01):
      """

      Simulates all Conditions, SSD, trials, timepoints simultaneously.
      Vectorized operations are set up so that any of the parameters can be
      a single float or a vector of floats (i.e., when simulating/fitting multiple
      conditions differentiated by the value of one or more model parameters)

      args:
            a, tr, v, ssv, z (float/array):     model parameters
            ssd  (array):                       full set of stop signal delays
            nss  (int):                         number of stop trials
            ntot (int):                         number of total trials
            tb (float):                         time boundary

      returns:

            DVg (Go Process):             3d array for all conditions, trials, timepoints
                                          (i.e. DVg = [COND [NTrials [NTime]]] )
                                          All conditions are simulated simultaneously (i.e., BSL & PNL)

            DVs (Stop Process):           4d array for all conditions, SSD, SS trials, timepoints.
                                          i.e. ( DVs = [COND [SSD [nSSTrials [NTime]]]] )
                                          All ss decision traces are initiated from DVg[Cond](t=SSD | SSD<tr)

      Output can be passed to <analyze_reactive_full()> to extract
      expected values to be entered into the cost f(x)
      """


      nssd = len(ssd)
      nss = int(.5*ntot)
      dx=np.sqrt(si*dt)
      Pg = 0.5*(1 + v*dx/si)
      Tg = np.ceil((tb-tr)/dt).astype(int)

      Ps = 0.5*(1 + ssv*dx/si)
      Ts = np.ceil((tb-ssd)/dt).astype(int)

      # a/tr/v Bias: ALL CONDITIONS, ALL SSD
      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
      init_ss = np.array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)

      # ALL CONDITIONS, ALL SSD
      grt = (tr + (np.where(DVg[:, nss:, :].max(axis=2)>=a, np.argmax(DVg[:, nss:, :]>=a, axis=2)*dt, np.nan).T)).T
      ert = (tr + (np.where(DVg[:, :nss, :].max(axis=2)>=a, np.argmax(DVg[:, :nss, :]>=a, axis=2)*dt, np.nan).T)).T
      ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt, np.nan)

      #collapse across SSD and get average ssrt vector for each condition
      c_ssrt = ssrt.mean(axis=1)

      # compute RT quantiles for correct and error resp.
      gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*10 for rtc in grt])
      eq = np.array([mq(np.extract(ert[i]<c_ssrt[i], ert[i]), prob=prob)*10 for i in range(ncond)])

      # Get response and stop accuracy information
      gac = np.mean(np.where(grt<tb, 1, 0), axis=1)
      sacc = np.array([1 - np.where(ert[i]<ssrt[i], 1, 0).mean(axis=1) for i in range(ncond)])

      yhat = [gac, sacc, gq, eq]
      yhat_grouped = np.array([np.hstack([i[ii] for i in yhat]) for ii in range(ncond)])

      return yhat_grouped



def sim_trbias_full():

      # Onset-Bias: ALL CONDITIONS, ALL SSD

      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
      init_ss = np.array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)

      # ALL CONDITIONS, ALL SSD
      grt = (tr + (np.where(DVg[:, nss:, :].max(axis=2)>=a, np.argmax(DVg[:, nss:, :]>=a, axis=2)*dt, np.nan).T)).T
      ert = (tr + (np.where(DVg[:, :nss, :].max(axis=2)>=a, np.argmax(DVg[:, :nss, :]>=a, axis=2)*dt, np.nan).T)).T
      ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt, np.nan)


def sim_vbias_full():

      # Drift-Rate Bias: ALL CONDITIONS, ALL SSD
      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg)).T < Pg), dx, -dx).T, axis=2)
      init_ss = np.array([[DVc[:nss, ix] for ix in np.where(Ts<Tg, Tg-Ts, 0)] for DVc in DVg])
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)

      # ALL CONDITIONS, ALL SSD
      grt = np.where(DVg[:, nss:, :].max(axis=2)>=a, tr + np.argmax(DVg[:, nss:, :]>=a, axis=2)*dt, np.nan)
      ert = np.where(DVg[:, :nss, :].max(axis=2)>=a, tr + np.argmax(DVg[:, :nss, :]>=a, axis=2)*dt, np.nan)
      ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt, np.nan)



def resim_only(theta, bias='v', ncond=1, ntrials=2000, wts=None, pGo=.5, ssd=np.arange(.2, .45, .05), p=np.array([.1, .3, .5, .7, .9])):

      p = {k:theta[k] for k in theta.keys()}

      p[bias] = np.array([theta[bias+str(i)] for i in range(ncond)])
      a, tr, v, ssv, z  = theta['a'], theta['tr'], theta['v'], -abs(theta['ssv']),  theta['z']

      nss = int((1-pGo)*ntrials)
      yhat = simulate_full(a, tr, v, ssv, z, p=p, ncond=ncond, ssd=ssd, nss=nss, ntot=ntrials)
      #wtc, wte = wts[0], wts[1]
      #cost = np.hstack([y[:6]-yhat[:6], wtc*y[6:11]-wtc*yhat[6:11], wte*y[11:]-wte*yhat[11:]]).astype(np.float32)

      return yhat


#      def simulate(a, tr, v, ssv, z, ssd=np.arange(.2, .45, .05), nss=1000, ntot=2000, tb=0.650, dt=.0005, si=.01):
#            """
#
#            Simulates all SSD, trials, timepoints simultaneously
#            for a single condition
#
#            args:
#
#                  a, tr, v, ssv, z (float):     model parameters
#                  ssd  (np array):              full set of stop signal delays
#                  nss  (int):                   number of stop trials
#                  ntot (int):                   number of total trials
#                  tb (float):                   time boundary
#
#            returns:
#
#                  DVg:  2d array (ntrials x ntimepoints) for all trials
#
#                  DVs:  3d array (Num SSD x nSS trials x ntimepoints) for all stop signal trials
#                        All nss SS trials for all SSD conditions (DVs). All ss decision traces
#                        are initiated from DVg(t=SSD) if SSD<tr
#
#            Output can be passed to  <analyze_reactive()>  for summary measures
#            """
#
#            dx=np.sqrt(si*dt)
#            nssd = len(ssd)
#
#            Pg = 0.5*(1 + v*dx/si)
#            Tg = np.ceil((tb-tr)/dt).astype(int)
#
#            Ps = 0.5*(1 + ssv*dx/si)
#            Ts = np.ceil((tb-ssd)/dt).astype(int)
#
#            # SINGLE CONDITION, ALL SSD
#            DVg = z + np.cumsum(np.where(rs((ntot, Tg)) < Pg, dx, -dx), axis=1)
#            init_ss = np.array([DVg[:, :nss, ix] for ix in np.where(Ts<Tg, Tg-Ts, 0)])
#            DVs = init_ss[:, :, None] + np.cumsum(np.where(rs((nssd, nss, Ts.max()))<Ps, dx, -dx), axis=2)
#
#            return DVg, DVs
#
#
#      def analyze_reactive_single(DVg, DVs, a,  tr, ssd, nss=1000, tb=.650, dt=.0005, p=np.array([.1, .3, .5, .7, .9])):
#
#            """
#            Takes Go and Stop process vectors from run() output and
#            extracts Go/Stop Accuracy, and Correct and Incorrect RT Quantiles
#            for the single condition simulated in run(), including all SSDs
#
#            """
#
#            # SINGLE CONDITION, ALL SSD
#            grt = np.where(DVg[nss:, :].max(axis=1)>=a, tr + np.argmax(DVg[nss:, :]>=a, axis=1)*dt, np.nan)
#            ert = np.where(DVg[:nss, :].max(axis=1)>=a, tr + np.argmax(DVg[:nss, :]>=a, axis=1)*dt, np.nan)
#            ssrt = np.where(np.any(DVs<=0, axis=2), ssd[:, None]+np.argmax(DVs<=0, axis=2)*dt, np.nan)
#
#            # compute RT quantiles for correct and error resp.
#            gq = mq(grt[grt<tb], prob=p)
#            eq = mq(np.hstack([np.extract(ert<ssi, ert) for ssi in ssrt]), prob=p)
#
#            # Get response and stop accuracy information
#            gac = np.where(grt<tb,1,0).mean()
#            sacc = 1 - np.where(ert<ssrt, 1, 0).mean(axis=1)
#
#            return np.hstack([gac, sacc, gq*10, eq*10])
#
#
#
#      def analyze_reactive_full(DVg, DVs, a,  tr, ssd, nss=1000, tb=.650, dt=.0005, p=np.array([.1, .3, .5, .7, .9])):
#
#            """
#            Takes Go and Stop process vectors from simulate_full() output and
#            extracts Go/Stop Accuracy, and Correct and Incorrect RT Quantiles
#            for all conditions, SSDs simulated in simulate_full()
#
#            """
#
#            # ALL CONDITIONS, ALL SSD
#            grt = np.where(DVg[:, nss:, :].max(axis=2)>=a, tr + np.argmax(DVg[:, nss:, :]>=a, axis=2)*dt, np.nan)
#            ert = np.where(DVg[:, :nss, :].max(axis=2)>=a, tr + np.argmax(DVg[:, :nss, :]>=a, axis=2)*dt, np.nan)
#            ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt, np.nan)
#
#            #collapse across SSD and get average ssrt vector for each condition
#            c_ssrt = ssrt.mean(axis=1)
#
#            # compute RT quantiles for correct and error resp.
#            gq = np.vstack([mq(rtc[rtc<tb], prob=p) for rtc in grt])
#            eq = [mq(np.extract(ert[i]<c_ssrt[i], ert[i]), prob=p) for i in range(ncond)]
#
#            # Get response and stop accuracy information
#            gac = np.where(grt<tb, 1, 0).mean(axis=1)
#            sacc = np.array([1 - np.where(ert[i]<ssrt[i], 1, 0).mean(axis=1) for i in range(ncond)])
#
#            np.hstack([gac, sacc, gq*10, eq*10])
