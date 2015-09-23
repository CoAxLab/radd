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

def simulate_learning(p, pc_map={'vd':['vd_e', 'vd_u', 'vd_l'], 'vi':['vi_e', 'vi_u', 'vi_l']}, nc=3, lr=array([.4,.3]), nssd=5, dt=.0005):

      p = vectorize_params(p, pc_map=pc_map, ncond=nc)
      Pd, Pi, Tex = update_execution(p)
      t = np.cumsum([dt]*Tex.max())
      xtb = temporal_dynamics(p, t)
      #Ph, Th = update_brake(p)
      #ss_index = [np.where(Th<Tex[c],Tex[c]-Th,0) for c in range(nc)]
      rts, vd, vi = [], [], []
      for i in xrange(ntot):

            Pd, Pi, Tex = update_execution(p)

            direct = np.where((rs((nc, Tex.max())).T < Pd),dx,-dx).T
            indirect = np.where((rs((nc, Tex.max())).T < Pi),dx,-dx).T
            execution = np.cumsum(direct+indirect, axis=1)

            #if i<=int(.5*ntot):
            #      init_ss = array([[execution[c,ix] for ix in ss_index] for c in range(nc)])
            #      hyper = init_ss[:,:,:,None]+np.cumsum(np.where(rs(Th.max())<Ph, dx, -dx), axis=1)

            
            r = np.argmax((execution.T>=p['a']).T, axis=1)*dt
            rt = p['tr']+(r*np.where(r==0, np.nan, 1))
            resp = np.where(rt<tb, 1, 0)

            # find conditions where response was recorded
            for ci in np.where(~np.isnan(rt))[0]:
                  p['vd'][ci]=p['vd'][ci] + p['vd'][ci]*(lr[0]*(rt[ci]-.500))
                  p['vi'][ci]=p['vi'][ci] - p['vi'][ci]*(lr[1]*(rt[ci]-.500))

            vd.append(deepcopy(p['vd']))
            vi.append(deepcopy(p['vi']))
            rts.append(rt)

      vd = np.asarray(vd)
      vi = np.asarray(vi)
      rts = np.asarray(rts)

def feedback_signal():
      pass
