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


def sim_conditions(theta, nsims=50, ssd=np.arange(.2, .45, .05)):

      if not type(theta)==dict:
            theta={k:theta[k].value for k in theta.keys()}

      a, tr, v, ssv, z  = theta['a'], theta['tr'], theta['v'], -abs(theta['ssv']),  theta['z']
      yhat = []

      for i in range(nsims):
            dvg, dvs = simulate_reactive(a, tr, v, ssv, z, ssd, nss=1000, ntot=2000)
            yhat.append(sim_quantile_accuracy(dvg, dvs, a, tr, ssd, nss=1000))

      return yhat


def run(a, tr, v, ssv, z, ssd=np.arange(.2, .45, .05), nss=1000, ntot=2000, tb=0.650, tau=.0005, si=.01):
      """

      This is an optimized version of RADD.run() which is used during the
      fitting routine to vectorize all simulated trials and timepoints.

      In contrast to RADD.run(), which simulates a single SSD condition, this function
      simulates all SSD conditions simultaneously

      args:

            a, tr, v, ssv, z (float):     model parameters
            ssd  (np array):              full set of stop signal delays
            nss  (int):                   number of stop trials
            ntot (int):                   number of total trials
            tb :                          time boundary

      returns:

            DVg:  2d array (ntrials x ntimepoints) for all trials

            DVs:  3d array (Num SSD x nSS trials x ntimepoints) for all stop signal trials
                  All nss SS trials for all SSD conditions (DVs). All ss decision traces
                  are initiated from DVg(t=SSD) if SSD<tr

      Output can be passed to sim quantile accuracy for summary measures
      """

      nssd=len(ssd);
      dx=np.sqrt(si*tau)

      Pg = 0.5*(1 + v*dx/si)
      Tg = np.ceil((tb-tr)/tau).astype(int)

      Ps = 0.5*(1 + ssv*dx/si)
      Ts = np.ceil((tb-ssd)/tau).astype(int)

      DVg = z + np.cumsum(np.where(rs((ntot, Tg))<Pg, dx, -dx), axis=1)

      init_ss = np.array([DVg[:nss, ix] for ix in np.where(Ts<Tg, Tg-Ts, 0)])
      DVs = init_ss[:, :, None] + np.cumsum(np.where(rs((nssd, nss, Ts.max()))<Ps, dx, -dx), axis=2)
      return DVg, DVs


def analyze_reactive(DVg, DVs, a,  tr, ssd, nss=1000, tb=.650, tau=.0005, p=np.array([.1, .3, .5, .7, .9])):

      grt = np.where(DVg[nss:, :].max(axis=1)>=a, tr + np.argmax(DVg[nss:, :]>=a, axis=1)*tau, np.nan)
      ert = np.where(DVg[:nss, :].max(axis=1)>=a, tr + np.argmax(DVg[:nss, :]>=a, axis=1)*tau, np.nan)
      ssrt = np.where(np.any(DVs<=0, axis=2), ssd[:, None]+np.argmax(DVs<=0, axis=2)*tau, np.nan)

      # comute RT quantiles for correct and error respself.
      cg_quant = mq(grt[grt<tb], prob=p)
      eg_quant = mq(np.hstack([np.extract(ert<ssi, ert) for ssi in ssrt]), prob=p)

      # Get response and stop accuracy information
      gac = np.where(grt<tb,1,0).mean()
      sacc = 1 - np.where(ert<ssrt, 1, 0).mean(axis=1)

      return [gac, sacc, cg_quant, eg_quant]
