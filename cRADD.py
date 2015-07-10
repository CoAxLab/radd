#!/usr/local/bin/env python
from __future__ import division
import numpy as np
from numpy.random import random_sample as rs
from scipy.stats.mstats import mquantiles as mq

"""
Working code for fitting reactive data:
This module is essentially an optimized version of the
core fitting functions in fitre.py and the
model simulations in RADD.py

RUNTIME TESTS:
------------------------------------------
50 iterations of simulating 10,000 trials:
------------------------------------------
fitre + RADD:
1 loops, best of 3: 1min 32s per loop

cRADD:
1 loops, best of 3: 12.9 s per loop
-----------------------------------------
"""


def recost(theta, y, ntrials=2000, wts=None, pGo=.5, ssd=np.arange(.2, .45, .05)):

      if not type(theta)==dict:
            theta = theta.valuesdict()

      a, tr, v, ssv, z  = theta['a'], theta['tr'], theta['v'], -abs(theta['ssv']),  theta['z']

      nss = int((1-pGo)*ntrials)
      dvg, dvs = run(a, tr, v, ssv, z, ssd, nss=nss, ntot=ntrials)
      yhat = analyze_reactive(dvg, dvs, a, tr, ssd, nss=nss)
      if wts:
            wtc, wte = wts[0], wts[1]
      else:
            wtc, wte = [np.ones(5)]*2
      cost = np.hstack([y[:6]-yhat[:6], wtc*y[6:11]-wtc*yhat[6:11], wte*y[11:]-wte*yhat[11:]]).astype(np.float32)


      return cost


def run(a, tr, v, ssv, z, ssd=np.arange(.2, .45, .05), nss=1000, ntot=2000, tb=0.650, tau=.0005, si=.01, depends=None):
      """

      Simulates all SSD, trials, timepoints simultaneously
      for a single condition

      args:

            a, tr, v, ssv, z (float):     model parameters
            ssd  (np array):              full set of stop signal delays
            nss  (int):                   number of stop trials
            ntot (int):                   number of total trials
            tb (float):                   time boundary

      returns:

            DVg:  2d array (ntrials x ntimepoints) for all trials

            DVs:  3d array (Num SSD x nSS trials x ntimepoints) for all stop signal trials
                  All nss SS trials for all SSD conditions (DVs). All ss decision traces
                  are initiated from DVg(t=SSD) if SSD<tr

      Output can be passed to  <analyze_reactive()>  for summary measures
      """

      nssd=len(ssd);
      dx=np.sqrt(si*tau)

      Pg = 0.5*(1 + v*dx/si)
      Tg = np.ceil((tb-tr)/tau).astype(int)

      Ps = 0.5*(1 + ssv*dx/si)
      Ts = np.ceil((tb-ssd)/tau).astype(int)

      # SINGLE CONDITION, ALL SSD
      DVg = z + np.cumsum(np.where(rs((ntot, Tg)) < Pg, dx, -dx), axis=1)
      init_ss = np.array([DVg[:, :nss, ix] for ix in np.where(Ts<Tg, Tg-Ts, 0)])
      DVs = init_ss[:, :, None] + np.cumsum(np.where(rs((nssd, nss, Ts.max()))<Ps, dx, -dx), axis=2)

      return DVg, DVs


def run_full(a, tr, ssv, z, v=None, ssd=np.arange(.2, .45, .05), nss=1000, ntot=2000, tb=0.650, tau=.0005, si=.01):
      """

      Simulates all Conditions, SSD, trials, timepoints simultaneously by
      treating drift-rate as a vector, containing a value for each condition.

      args:
            a, tr, ssv, z (float):        model parameters (excl 'v')
            v (array):                    array of drift-rates (1/cond)
            ssd  (array):                 full set of stop signal delays
            nss  (int):                   number of stop trials
            ntot (int):                   number of total trials
            tb (float):                   time boundary

      returns:

            DVg (Go Process):       3d array for all conditions, trials, timepoints 
                                    (i.e. DVg = [COND [NTrials [NTime]]] )
                                    All conditions are simulated simultaneously (i.e., BSL & PNL)

            DVs (Stop Process):     4d array for all conditions, SSD, SS trials, timepoints.
                                    i.e. ( DVs = [COND [SSD [nSSTrials [NTime]]]] )
                                    All ss decision traces are initiated from DVg[Cond](t=SSD | SSD<tr)

      Output can be passed to <analyze_reactive_full()> to extract
      expected values to be entered into the cost f(x)
      """

      nssd=len(ssd);
      dx=np.sqrt(si*tau)
      ncond=len(v)

      Pg = 0.5*(1 + v*dx/si)
      Tg = np.ceil((tb-tr)/tau).astype(int)

      Ps = 0.5*(1 + ssv*dx/si)
      Ts = np.ceil((tb-ssd)/tau).astype(int)

      # ALL CONDITIONS, ALL SSD
      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg)).T < Pg), dx, -dx).T, axis=2)
      init_ss = np.array([np.array([DVc[:nss, ix] for ix in np.where(Ts<Tg, Tg-Ts, 0)]) for DVc in DVg])
      DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((ncond, nssd, nss, Ts.max()))<Ps, dx, -dx), axis=3)

      return DVg, DVs



def analyze_reactive(DVg, DVs, a,  tr, ssd, nss=1000, tb=.650, tau=.0005, p=np.array([.1, .3, .5, .7, .9])):

      """
      Takes Go and Stop process vectors from run() output and
      extracts Go/Stop Accuracy, and Correct and Incorrect RT Quantiles
      for the single condition simulated in run(), including all SSDs

      """

      # SINGLE CONDITION, ALL SSD
      grt = np.where(DVg[nss:, :].max(axis=1)>=a, tr + np.argmax(DVg[nss:, :]>=a, axis=1)*tau, np.nan)
      ert = np.where(DVg[:nss, :].max(axis=1)>=a, tr + np.argmax(DVg[:nss, :]>=a, axis=1)*tau, np.nan)
      ssrt = np.where(np.any(DVs<=0, axis=2), ssd[:, None]+np.argmax(DVs<=0, axis=2)*tau, np.nan)

      # compute RT quantiles for correct and error resp.
      gq = mq(grt[grt<tb], prob=p)
      eq = mq(np.hstack([np.extract(ert<ssi, ert) for ssi in ssrt]), prob=p)

      # Get response and stop accuracy information
      gac = np.where(grt<tb,1,0).mean()
      sacc = 1 - np.where(ert<ssrt, 1, 0).mean(axis=1)

      return np.hstack([gac, sacc, gq*10, eq*10])



def analyze_reactive_full(DVg, DVs, a,  tr, ssd, nss=1000, tb=.650, tau=.0005, p=np.array([.1, .3, .5, .7, .9])):

      """
      Takes Go and Stop process vectors from run_full() output and
      extracts Go/Stop Accuracy, and Correct and Incorrect RT Quantiles
      for all conditions, SSDs simulated in run_full()

      """

      # ALL CONDITIONS, ALL SSD
      grt = np.where(DVg[:, nss:, :].max(axis=2)>=a, tr + np.argmax(DVg[:, nss:, :]>=a, axis=2)*tau, np.nan)
      ert = np.where(DVg[:, :nss, :].max(axis=2)>=a, tr + np.argmax(DVg[:, :nss, :]>=a, axis=2)*tau, np.nan)
      ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*tau, np.nan)

      #collapse across SSD and get average ssrt vector for each condition
      c_ssrt = ssrt.mean(axis=1)

      # compute RT quantiles for correct and error resp.
      gq = np.vstack([mq(rtc[rtc<tb], prob=p) for rtc in grt])
      eq = [mq(np.extract(ert[i]<c_ssrt[i], ert[i]), prob=p) for i in range(ncond)]

      # Get response and stop accuracy information
      gac = np.where(grt<tb, 1, 0).mean(axis=1)
      sacc = np.array([1 - np.where(ert[i]<ssrt[i], 1, 0).mean(axis=1) for i in range(ncond)])




def simulate(theta, ntrials=2000, nss=1000, pGo=.5, ssd=np.arange(.2, .45, .05)):

        if not type(theta)==dict:
              theta = theta.valuesdict()#theta={k:theta[k].value for k in theta.keys()}

        a, tr, v, ssv, z  = theta['a'], theta['tr'], theta['v'], -abs(theta['ssv']),  theta['z']
        nss = int((1-pGo)*ntrials)
        dvg, dvs = run(a, tr, v, ssv, z, ssd, nss=nss, ntot=ntrials)
        return analyze_reactive(dvg, dvs, a, tr, ssd, nss=nss)
