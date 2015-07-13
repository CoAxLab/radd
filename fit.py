#!/usr/local/bin/env python
from __future__ import division
import time
import numpy as np
from lmfit import Parameters, minimize, fit_report
from numpy.random import random_sample as rs
from scipy.stats.mstats import mquantiles as mq

"""
Main code for fitting reactive data:
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
------------------------------------------------------------------
10 iterations of simulating 10,000 trials:
==================================================================

fitre + RADD: first loop iterates 10 runs, second loop iterates
conditions, ssre_minfunc iterates over 5 SSD's with a for loop
------------------------------------------------------------------
# get param dict and make an array for drift-rates for two cond


      p = {k:v for k,v in redf_store['rebsl_boot_popt'].items()}
      v_cond = np.array([p['v']*1.05,  p['v']*.95])

      ***********************************************************

      '%''%'timeit
      # 10 runs
      for i in range(10):
            # 2 Conditions
            for i in range(2):
                  #update drift-rate, sim 5000 trials
                  p['v'] = v_cond[i]
                  yhat2 = fitre.ssre_minfunc(p, y2, ntrials=5000)


      <OUTPUT> 1 loops, best of 3: 1min 21s per loop

      ***********************************************************


==================================================================
fit: first loop iterates 10 runs, recost calls fit.simulate_full
which vectorizes Go and Stop processes across 2 conditions & 5 SSD
------------------------------------------------------------------


      # include drift-rate for two conditions in param dict
      p = {k:v for k,v in redf_store['rebsl_boot_popt'].items()}
      p['v0'], p['v1'] = np.array([p['v']*1.05,  p['v']*.95])

      ***********************************************************

      '%''%'timeit
      # 10 runs
      for i in range(10):
            # 2 Conditions x 5000 trials
            yhat = fit.recost(p, y, ntrials=10000)


      <OUTPUT> 1 loops, best of 3: 7.82 s per loop

      ***********************************************************


"""


def optimize(y, inits={}, bias=['xx'], wts=None, ncond=1, ntrials=5000, maxfev=5000, ftol=1.e-3, xtol=1.e-3, all_params=0, disp=True, fitid=None, log_fits=True, method='nelder'):

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

      ip = inits.copy()
      lim = set_bounds()
      if 'pGo' in ip.keys(): del ip['pGo']
      if 'ssd' in ip.keys(): del ip['ssd']
      ip['ssv']=-abs(ip['ssv'])

      popt=Parameters()

      for bk in bias:
            bv = ip.pop(bk)
            mn = lim[bk][0]; mx = lim[bk][1]
            d0 = [popt.add(bk+str(i), value=bv, vary=1, min=mn, max=mx) for i in range(ncond)]

      p0 = [popt.add(k, value=v, vary=0, min=lim[k][0], max=lim[k][1]) for k, v in ip.items()]

      f_kws = {'wts':wts, 'ncond':ncond, 'ntrials':ntrials}
      opt_kws = {'disp':disp, 'xtol':xtol, 'ftol':ftol}#, 'maxfev':maxfev}
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

      yhat =  np.vstack(y) + optmod.residual.reshape(ncond, int(len(optmod.residual)/ncond))

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

      """
      simulate data via <simulate_full> and return weighted
      cost between observed (y) and simulated values (yhat).

      returned vector is implicitly used by lmfit minimize
      routine invoked in <optimize> which then submits the
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

      p = {k:theta[k] for k in theta.keys()}

      ssd=np.arange(.2, .45, .05);
      prob=np.array([.1, .3, .5, .7, .9])

      cond = {pk: p.pop(pk) for pk in p.keys() if pk[-1].isdigit()}
      ncond = len(cond.keys())
      for i in range(ncond):
            p[cond.keys()[i][:-1]] = np.array(cond.values())

      if 'tr' not in bias:
            p['tr'] = np.array([p['tr']]*ncond)

      yhat = simulate_full(p['a'], p['tr'], p['v'], -abs(p['ssv']),  p['z'], prob=prob, ncond=ncond, ssd=ssd, ntot=ntrials)
      if wts is None:
            wtc, wte = np.ones((ncond,10))
      else:
            wtc, wte = m.wts.T[:5].T, m.wts.T[5:].T

      y=np.vstack(y)
      cost = np.hstack(np.hstack([y[:, :6] - yhat[:, :6], wtc*y[:, 6:11] - wtc*yhat[:, 6:11], wte*y[:, 11:] - wte*yhat[:, 11:]])).astype(np.float32)

      return cost


def set_bounds(a=(.01, .6), tr=(.001, .5), v=(.01, 4.), z=(.001, .9), ssv=(-4., -.01)):

      """
      set and return boundaries to limit search space
      of parameter optimization in <optimize>
      """

      return {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'z': z}



def simulate_full(a, tr, v, ssv, z, ncond=1, prob=np.array([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=2000, tb=0.650, dt=.0005, si=.01, return_traces=False):
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

      nssd = len(ssd); nss = int(.5*ntot)

      Pg = 0.5*(1 + v*dx/si)
      Ps = 0.5*(1 + ssv*dx/si)

      Tg = np.ceil((tb-tr)/dt).astype(int)
      Ts = np.ceil((tb-ssd)/dt).astype(int)

      # a/tr/v Bias: ALL CONDITIONS, ALL SSD
      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
      init_ss = np.array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)

      if return_traces:
            return [DVg, DVs]

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

      return np.array([np.hstack([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])



def resim_yhat(theta, return_traces=False, bias=['v'], ntrials=2000, wts=None):

      """
      simulate multiple conditions, ssd (as in recost function)

      args:
            theta (dict):                 param dict
            return_traces (bool):         return [DVg, DVs]

      returns:
            output:
                  output is either simulated predictions (i.e. yhat)
                        or Go and Stop traces in list (i.e. [DVg, DVs])

      """


      p = {k:theta[k] for k in theta.keys()}
      ssd=np.arange(.2, .45, .05);
      prob=np.array([.1, .3, .5, .7, .9])

      cond = {pk: p.pop(pk) for pk in p.keys() if pk[-1].isdigit()}
      ncond = len(cond.keys())
      for i in range(ncond):
            p[cond.keys()[i][:-1]] = np.array(cond.values())

      if 'tr' not in bias:
            p['tr'] = np.array([p['tr']]*2)

      output = simulate_full(p['a'], p['tr'], p['v'], -abs(p['ssv']),  p['z'], prob=prob, ncond=ncond, ssd=ssd, ntot=ntrials, return_traces=False)

      return output



def sim_trbias_full(a, tr, v, ssv, z, analyze=True, ncond=1, prob=np.array([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=2000, tb=0.650, dt=.0005, si=.01):

      # Onset-Bias: ALL CONDITIONS, ALL SSD

      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
      init_ss = np.array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)

      # ALL CONDITIONS, ALL SSD
      grt = (tr + (np.where(DVg[:, nss:, :].max(axis=2)>=a, np.argmax(DVg[:, nss:, :]>=a, axis=2)*dt, np.nan).T)).T
      ert = (tr + (np.where(DVg[:, :nss, :].max(axis=2)>=a, np.argmax(DVg[:, :nss, :]>=a, axis=2)*dt, np.nan).T)).T
      ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt, np.nan)

      return grt, ert, ssrt


def sim_vbias_full(a, tr, v, ssv, z, analyze=True, ncond=1, prob=np.array([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=2000, tb=0.650, dt=.0005, si=.01):

      # Drift-Rate Bias: ALL CONDITIONS, ALL SSD
      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg)).T < Pg), dx, -dx).T, axis=2)
      init_ss = np.array([[DVc[:nss, ix] for ix in np.where(Ts<Tg, Tg-Ts, 0)] for DVc in DVg])
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)

      # ALL CONDITIONS, ALL SSD
      grt = np.where(DVg[:, nss:, :].max(axis=2)>=a, tr + np.argmax(DVg[:, nss:, :]>=a, axis=2)*dt, np.nan)
      ert = np.where(DVg[:, :nss, :].max(axis=2)>=a, tr + np.argmax(DVg[:, :nss, :]>=a, axis=2)*dt, np.nan)
      ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt, np.nan)

      return grt, ert, ssrt
