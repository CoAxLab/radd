#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import hstack as hs
from numpy import newaxis as na
from scipy.stats.mstats import mquantiles as mq

resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=2)*dt
ss_resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=3)*dt
resp_lo = lambda trace: np.argmax((trace.T<=0).T, axis=3)*dt
RT = lambda ontime, rbool: ontime[:,na]+(rbool*np.where(rbool==0, np.nan, 1))
RTQ = lambda zpd: map((lambda x:mq(x[0][x[0]<x[1]], prob)), zpd)


def vectorize_params(p, pc_map, ncond=1):
      pvc = ['a', 'tr', 'vd', 'vi', 'xb']
      for pkey in pvc:
            p[pkey]=p[pkey]*np.ones(ncond)
      for pkey, pkc in pc_map.items():
            if ncond==1:
                  p[pkey]=np.asarray([p[pkey]])
                  break
            elif pkc[0] not in p.keys():
                  p[pkey] = p[pkey]*np.ones(len(pkc))
            else:
                  p[pkey] = array([p[pc] for pc in pkc])
      return p


def update_execution(p):
      """ update Pg (probability of DVg +dx) and Tg (num go process timepoints)
      for go process and get get dynamic bias signal if 'x' model
      """
      Pd = 0.5*(1 + p['vd']*dx/si)
      Pi = 0.5*(1 + p['vi']*dx/si)
      Tg = np.ceil((tb-p['tr'])/dt).astype(int)

      return Pd, Pi, Tg


def simulate_pro(p, pc_map={'v':['v0', 'v100']}, lr=array([1,1]), tb=.555):
      """ Simulate the proactive competition model
      (see simulate_dpm() for I/O details)
      """

      p = vectorize_params(p, pc_map=pc_map, ncond=nc)
      nc = len(p['vd'])

      p = update_execution(p)
      xtb = temporal_dynamics(p, np.cumsum([dt]*Tg.max()))

      for i in xrange(ntot):

            Pd, Pi, Tg = update_execution(p)

            direct = np.where((rs((nc, Tg.max())).T < Pd),dx,-dx).T
            indirect = xtb[:,::-1]*np.where((rs((nc, Tg.max())).T < Pi),dx,-dx).T
            execution = np.cumsum(direct+indirect, axis=1)

            r = np.argmax((execution.T>=p['a']).T, axis=1)*dt
            rt = p['tr']+(r*np.where(r==0, np.nan, 1))

            lr_scale = lr.T*np.where(rt<tb, 1, 0)
            p['vd']=p['vd'] + p['vd']*lr_scale[0]
            p['vi']=p['vi'] + p['vi']*lr_scale[1]

      return p

def feedback_signal():
      pass
