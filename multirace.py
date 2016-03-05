#!/usr/local/bin/env python
from __future__ import division
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na


temporal_dynamics = lambda p, t: np.cosh(p['xb'][:, na] * t)


def rew_func(rprob):
    if rs()<rprob:
        return 1
    else:
        return 0


def vectorize_params(p, pc_map, ncond=3):
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


def multi_response(p, nresp=4, pc_map={'vd': ['vd_e', 'vd_u', 'vd_l'], 'vi': ['vi_e', 'vi_u', 'vi_l']}, dt=.001, si=.1, tb=1.5):

    nc = len(pc_map.values()[0])
    p = vectorize_params(p, pc_map=pc_map, ncond=nc)

    dx=np.sqrt(si*dt)
    Pd = 0.5*(1 + p['vd']*dx/si)
    Pi = 0.5*(1 + p['vi']*dx/si)

    Tex = np.ceil((tb-p['tr'])/dt).astype(int)
    Pd, Pi, Tex = update_execution(p)
    xtb = temporal_dynamics(p, np.cumsum([dt]*Tex.max()))

    direct = np.where((rs((nc, nresp, Tex.max())).T < Pd),dx,-dx).T
    indirect = np.where((rs((nc, nresp, Tex.max())).T < Pi),dx,-dx).T
    execution = xtb[0] * np.cumsum(direct-indirect, axis=2)
    execution = execution.reshape(nresp, nc, Tex.max())

    winners=np.nan
    while np.isnan(winners):
        winners, rt, p = analyze_multiresponse(execution, p, dt)

    return int(winner), rt


def analyze_multiresponse(execution, p, dt):

    rt = array([p['tr'] + np.argmax((resp_x.T>=p['a']).T, axis=1)*dt for resp_x in execution])
    # set non responses to 999
    rt[rt==p['tr'][0]]=999
    # get accumulator with fastest RT (winner) in each cond
    winners = np.argmin(rt[rt<999], 0)
    # get rt of winner in each cond
    rts = array([rt[winner_ix, i] for i, winner_ix in enumerate(winners)])
    if np.all(rt==999):
        p['a']=p['a']*.99
        return np.nan, p

    return winners, rt, p
