#!/usr/local/bin/env python
from __future__ import division
import time
import numpy as np
from lmfit import Parameters, minimize, fit_report
from numpy.random import random_sample as rs
from scipy.stats.mstats import mquantiles as mq

def optimize_flat(y, inits={}, wts=None, ntrials=5000, maxfev=5000, ftol=1.e-3, xtol=1.e-3, all_params=1, disp=True, fitid=None, log_fits=True, method='nelder'):

      """
      The main function for optimizing parameters of FLAT reactive stop signal model.
      Flat models contain no parameter dependencies, thus a single set of parameters is
      fit by optimizing a weighted cost f(x) comparing 1d (1x16) arrays of
      observed and simulated Go accuracy, Stop accuracy (for each SSD condition), reaction time
      quantiles (.10q, .30q, .50q, .70q, .90q) for correct and error responses calculated over
      collapsing over all conditions in the dataset.

      Reccommended use of this function is by initiating a build.Model object and executing
      the fit_model() method

            Example:

                  model = build.Model(data=pd.DataFrame, inits=param_dict, kind='reactive', prepare=1)
                  model.fit_model(*args, **kwargs)

      args:

            y (np.array [nCondx16):             observed values entered into cost fx
                                                see build.Model for format info

            inits (dict):                       parameter dictionary including
                                                keys: a, tr, v, ssv, z

            wts (np.array [2x5]):               weights to be applied (separately) to
                                                correct and error RT quantiles. Can be estimated
                                                using get_wts() method of build.Model object


      """

      ip = inits.copy()
      lim = set_bounds()
      if 'pGo' in ip.keys(): del ip['pGo']
      if 'ssd' in ip.keys(): del ip['ssd']
      ip['ssv']=-abs(ip['ssv'])

      popt=Parameters()

      if method=='differential_evolution':
            pass
      else:
            aval = ip.pop('a'); zval = ip.pop('z')
            popt.add('a', value=aval, vary=1, min=lim['a'][0], max=lim['a'][1])
            popt.add('zperc', value=zval/aval, vary=1)
            popt.add('z', expr="zperc*a")

      p0 = [popt.add(k, value=v, vary=1, min=lim[k][0], max=lim[k][1]) for k, v in ip.items()]

      f_kws = {'wts':wts, 'ntrials':ntrials}
      opt_kws = {'disp':disp, 'xtol':xtol, 'ftol':ftol}#, 'maxfev':maxfev}
      optmod = minimize(recost_flat, popt, args=(y), method=method, kws=f_kws, options=opt_kws)

      params = popt.valuesdict()
      params['chi'] = optmod.chisqr
      params['rchi'] = optmod.redchi
      try:
            params['AIC']=optmod.aic
            params['BIC']=optmod.bic
      except Exception:
            params['AIC']=1000.0
            params['BIC']=1000.0

      yhat =  y + optmod.residual

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



def recost_flat(theta, y, ntrials=2000, wts=None):

      """
      simulate and cost function for a flat model all params = 1

      simulate data via <simulate_full> and return weighted
      cost between observed (y) and simulated values (yhat).

      returned vector is implicitly used by lmfit minimize
      routine invoked in <optimize> which then submits the
      SSE of the already weighted cost to a Nelder-Mead Simplex
      optimization.


      args:
            theta (dict):           param dict
            y (np.array):           1 x 16 array of observed
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

      yhat = simulate_flat(p['a'], p['tr'], p['v'], -abs(p['ssv']),  p['z'], prob=prob, ssd=ssd, ntot=ntrials)

      if wts is None:
            wtc, wte = np.ones(5), np.ones(5)
      else:
            wtc, wte = wts[0], wts[1]

      cost = np.hstack(np.hstack([y[:6] - yhat[:6], wtc*y[6:11] - wtc*yhat[6:11], wte*y[11:] - wte*yhat[11:]])).astype(np.float32)

      return cost



def set_bounds(a=(.01, .6), tr=(.001, .5), v=(.01, 4.), z=(.001, .9), ssv=(-4., -.01)):

      """
      set and return boundaries to limit search space
      of parameter optimization in <optimize>
      """

      return {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'z': z}




def simulate_flat(a, tr, v, ssv, z, prob=np.array([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=2000, tb=0.650, dt=.0005, si=.01):

      """

      Simulates all SSD, trials, timepoints from a single parameter set (i.e. flat).
      All parameters should be single floating point values.

      args:
            a, tr, v, ssv, z (float):           model parameters
            ssd  (array):                       full set of stop signal delays
            ntot (int):                         number of total trials
            tb (float):                         time boundary

      returns:

            DVg (Go Process):             2d array for all trials, timepoints
                                          (i.e. DVg = [NTrials [NTime]] )

            DVs (Stop Process):           3d array for all SSD, SS trials, timepoints.
                                          i.e. ( DVs = [SSD [nSSTrials [NTime]]] ); All
                                          DVs traces are initiated from DVg(t=SSD | SSD<tr)
      """

      dx=np.sqrt(si*dt)

      nssd = len(ssd); nss = int(.5*ntot)


      Tg = np.ceil((tb-tr)/dt).astype(int)
      Ts = np.ceil((tb-ssd)/dt).astype(int)

      Pg = 0.5*(1 + v*dx/si)
      Ps = 0.5*(1 + ssv*dx/si)

      # Flat Model (Ncond == 1), ALL SSD
      DVg = z + np.cumsum(np.where(rs((ntot, Tg)) < Pg, dx, -dx), axis=1)
      init_ss = np.array([DVg[:nss, ix] for ix in np.where(Ts<Tg, Tg-Ts, 0)])
      DVs = init_ss[:, :, None] + np.cumsum(np.where(rs((nssd, nss, Ts.max()))<Ps, dx, -dx), axis=2)

      grt = np.where(DVg[nss:, :].max(axis=1)>=a, tr + np.argmax(DVg[nss:, :]>=a, axis=1)*dt, np.nan)
      ert = np.where(DVg[:nss, :].max(axis=1)>=a, tr + np.argmax(DVg[:nss, :]>=a, axis=1)*dt, np.nan)
      ssrt = np.where(np.any(DVs<=0, axis=2), ssd[:, None]+np.argmax(DVs<=0, axis=2)*dt, np.nan)

      # compute RT quantiles for correct and error resp.
      gq = mq(grt[grt<tb], prob=prob)
      eq = mq(np.hstack([np.extract(ert<ssi, ert) for ssi in ssrt]), prob=prob)
      # Get response and stop accuracy information
      gac = np.where(grt<tb,1,0).mean()
      sacc = 1 - np.where(ert<ssrt, 1, 0).mean(axis=1)

      return np.hstack([gac, sacc, gq*10, eq*10])
