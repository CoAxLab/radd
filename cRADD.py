#!/usr/local/bin/env python
from __future__ import division
from numpy import cumsum, where, ceil, array, sqrt, nan
from numpy.random import random_sample as rs
from scipy.stats.mstats import mquantiles as mq
from numba import jit, autojit

"""
Working code for fitting reactive data:
all functions are jit compiled to c  using numba.
NOTE: numba does not support list comprehensions

This module is essentially an optimized version of the
effective fitting functions in fitre.py and the
model simulations in RADD.py

"""



@jit
def ssre_minfunc(p, y, nsims=50, nss=1000, ntot=2000):

        """
        still need to write a proper cost function that
        applies the weights and does the summation
        """

        a, tr, v, ssv, z  = p['a'], p['tr'], p['v'], p['ssv'],  p['z']
        yhat = []
        for ssd in np.arange(.2,.45,.05):
                dvg, dvs = simulate_reactive(a, tr, v, ssv, z, ssd, nss=1000, ntot=2000)
                gac, sac, cg_quant, eg_quant = sim_quantile_accuracy(dvg, dvs, a,  ssd, tr, nss=1000)
                yhat.append([gac, sac, cg_quant, eg_quant])

      return yhat


@jit
def simulate_reactive(a, tr, v, ssv, z, ssd, nss, ntot=2000, tb=0.650, tau=.0005, si=.01):

      """
      DVg is instantiated for all trials. DVs contains traces for a subset of
      those trials in which a SS occurs (proportional to pGo provided in theta).
      """
      dx=sqrt(si*tau)

      Pg = 0.5*(1 + v*dx/si)
      Tg = ceil((tb-tr)/tau)

      Ps = 0.5*(1 + ssv*dx/si)
      Ts = ceil((tb-ssd)/tau)

      DVg = z + cumsum(where(rs((ntot, Tg))<Pg, dx, -dx), axis=1)

      if tr<ssd:
            init_ss = DVg[:nss, Tg - Ts]
      else:
            init_ss = array([z]*nss)

      DVs = init_ss[:, None] + cumsum(where(rs((nss, Ts))<Ps, dx, -dx), axis=1)

      return DVg, DVs


@jit
def upper_rt_compiled(tr, ulim, DV, tb=.650, dt=.0005):
      trial_end = []
      for DVi in DV:
            if np.any(DVi>=ulim):
                  trial_end.append(tr + np.argmax(DVi>=ulim)*dt)
            else:
                  trial_end.append(999)

      return np.array(trial_end)


@jit
def lower_rt_compiled(ssd, DV, tb=.650, dt=.0005):

      trial_end = []
      for DVi in DV:
            if np.any(DVi<=0):
                  trial_end.append(ssd + np.argmax(DVi<=0)*dt)
            else:
                  trial_end.append(999)

      return np.array(trial_end)


@jit
def sim_quantile_accuracy(DVg, DVs, a,  ssd, tr, nss, tb=.650, dt=.0005, p=np.array([.1, .3, .5, .7, .9])):

      #check for and record go trial RTs
      #grt = upper_rt(tr, a, DVg[nss:, :])
      grt = upper_rt_compiled(tr, a, DVg[nss:, :])
      #ert = upper_rt(ssd, a, DVg[:nss, :])
      ert = upper_rt_compiled(ssd, a, DVg[:nss, :])
      #ssrt = lower_rt(ssd, DVs)
      ssrt = lower_rt_compiled(ssd, DVs)

      response = np.append(np.where(ert<=ssrt,1,0), np.where(grt<tb,1,0))

      cg_quant = mq(grt[grt<tb], prob=p)
      eg_quant = mq(np.extract(ert<=ssrt, ert), prob=p)
      gac = response[nss:].mean()
      sac = abs(1-response[:nss]).mean()

      return gac, sac, cg_quant, eg_quant
