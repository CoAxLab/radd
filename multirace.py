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


def vectorize_params(p, pc_map, ncond=4):
    pvc = ['a', 'tr', 'vd', 'vi', 'xb']
    #remove conditional parameters from pvc
    [pvc.remove(param) for param in pc_map.keys()]
    for pkey in pvc:
        p[pkey]=p[pkey]*np.ones(ncond)
    for pkey, pkc in pc_map.items():
        if ncond==1:
            break
        elif pkc[0] not in p.keys():
            p[pkey] = p[pkey]*np.ones(len(pkc))
        else:
            p[pkey] = array([p[pc] for pc in pkc])
    return p


def multi_response(p, pc_map={'vd': ['vd_a', 'vd_b', 'vd_c', 'vd_d'], 'vi': ['vi_a', 'vi_b', 'vi_c', 'vi_d']}, dt=.001, si=.1, tb=.8):

    nresp = len(pc_map.values()[0])
    p = vectorize_params(p, pc_map=pc_map, ncond=nresp)

    dx=np.sqrt(si*dt)
    Pd = 0.5*(1 + p['vd']*dx/si)
    Pi = 0.5*(1 + p['vi']*dx/si)

    Tex = np.ceil((tb-p['tr'])/dt).astype(int)
    #Pd, Pi, Tex = update_execution(p)
    xtb = temporal_dynamics(p, np.cumsum([dt]*Tex.max()))

    direct = np.where((rs((nresp, Tex.max())).T < Pd),dx,-dx).T
    indirect = np.where((rs((nresp, Tex.max())).T < Pi),dx,-dx).T
    execution = xtb[0] * np.cumsum(direct-indirect, axis=1)
    winners=np.nan
    while np.isnan(winners):
        winners, rt, p = analyze_multiresponse(execution, p, dt)

    return int(winner), rt


def analyze_multiresponse(execution, p, dt):

    nsteps_to_rt = np.argmax((execution.T>=p['a']).T, axis=1)
    rts = p['tr'] + nsteps_to_rt*dt

    # set non responses to 999
    rts[rts==p['tr'][0]]=999
    # get accumulator with fastest RT (winner) in each cond
    winner = np.argmin(rts, 0)
    # get rt of winner in each cond
    winrt = rts[winner]
    # slice all traces at time the winner crossed boundary
    trunc_ex = [execution[i, :nsteps_to_rt[winner]] for i in xrange(len(rts))]

    return winner, winrt, trunc_ex
