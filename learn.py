#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import hstack as hs
from numpy import newaxis as na
from scipy.stats.mstats import mquantiles as mq

resp_up = lambda trace, a: np.argmax((trace.T >= a).T, axis=2) * dt
ss_resp_up = lambda trace, a: np.argmax((trace.T >= a).T, axis=3) * dt
resp_lo = lambda trace: np.argmax((trace.T <= 0).T, axis=3) * dt
RT = lambda ontime, rbool: ontime[:, na] + \
    (rbool * np.where(rbool == 0, np.nan, 1))
RTQ = lambda zpd: map((lambda x: mq(x[0][x[0] < x[1]], prob)), zpd)


def vectorize_params(p, pc_map, ncond=1):
    pvc = ['a', 'tr', 'vd', 'vi', 'xb']
    for pkey in pvc:
        p[pkey] = p[pkey] * np.ones(ncond)
    for pkey, pkc in pc_map.items():
        if ncond == 1:
            p[pkey] = np.asarray([p[pkey]])
            break
        elif pkc[0] not in p.keys():
            p[pkey] = p[pkey] * np.ones(len(pkc))
        else:
            p[pkey] = array([p[pc] for pc in pkc])
    return p


def update_execution(p, si=.1, dt=.001, tb=3):
    """ update Pg (probability of DVg +dx) and Tg (num go process timepoints)
    for go process and get get dynamic bias signal if 'x' model
    """
    dx = np.sqrt(si * dt)
    Pd = 0.5 * (1 + p['vd'] * dx / si)
    Pi = 0.5 * (1 + p['vi'] * dx / si)
    Tex = np.ceil((tb - p['tr']) / dt).astype(int)

    return Pd, Pi, Tex


def simulate_learning(p, pc_map={'vd': ['vd_e', 'vd_u', 'vd_l'], 'vi': ['vi_e', 'vi_u', 'vi_l']}, nc=3, lr=array([.4, .3]), nssd=5, dt=.001, si=.1, ntot=1000, tb=.3):

    dx = np.sqrt(si * dt)
    p = vectorize_params(p, pc_map=pc_map, ncond=nc)

    Pd, Pi, Tex = update_execution(p)
    t = np.cumsum([dt] * Tex.max())
    xtb = temporal_dynamics(p, t)

    #Ph, Th = update_brake(p)
    #ss_index = [np.where(Th<Tex[c],Tex[c]-Th,0) for c in range(nc)]

    rts, vd, vi = [], [], []
    for i in xrange(ntot):

        Pd, Pi, Tex = update_execution(p)
        direct = np.where((rs((nc, Tex.max())).T < Pd), dx, -dx).T
        indirect = np.where((rs((nc, Tex.max())).T < Pi), dx, -dx).T
        execution = np.cumsum(direct + indirect, axis=1)

        # if i<=int(.5*ntot):
        #      init_ss = array([[execution[c,ix] for ix in ss_index] for c in range(nc)])
        #      hyper = init_ss[:,:,:,None]+np.cumsum(np.where(rs(Th.max())<Ph, dx, -dx), axis=1)
        r = np.argmax((execution.T >= p['a']).T, axis=1) * dt
        rt = p['tr'] + (r * np.where(r == 0, np.nan, 1))
        resp = np.where(rt < tb, 1, 0)

        # find conditions where response was recorded
        for ci in np.where(~np.isnan(rt))[0]:
            p['vd'][ci] = p['vd'][ci] + p['vd'][ci] * (lr[0] * (rt[ci] - .500))
            p['vi'][ci] = p['vi'][ci] - p['vi'][ci] * (lr[1] * (rt[ci] - .500))

        vd.append(deepcopy(p['vd']))
        vi.append(deepcopy(p['vi']))
        rts.append(rt)

    vd = np.asarray(vd)
    vi = np.asarray(vi)
    rts = np.asarray(rts)


def qlearn_wdi_exploration(p, stim=[['a', 'b'], ['c', 'd'], ['e', 'f']], preward=[.8, .7, .6], alpha=.01, ntrials=60, tb=1.5, si=.1):

    pc_map = {'vd': ['v_0', 'v_1'], 'vi': ['v_0', 'v_1']}
    p['a'] = .3
    p['vd'] = [.9, .9]
    p['vi'] = [.4, .4]
    plist = [deepcopy(p) for i in xrange(len(stim))]
    plist = [vectorize_params(p, pc_map, 2) for p in plist]

    rand_stim = sum([sorted(range(len(stim)), key=lambda k: random.random())
                     for i in xrange(ntrials)], [])

    for i in rand_stim:

        stim_pair_t = stim[i]
        stim_pair_qval = np.asarray([qdict[spt][-1] for spt in stim_pair_t])

        di_theta = plist[i]
        choice = di_decision(di_theta)
        xchoice = abs(1 - choice)

        if choice == 0:
            r = rew_func(preward[i])
        else:
            r = rew_func(1 - preward[i])

        update_q = stim_pair_qval[choice] + \
            alpha * (r - stim_pair_qval[choice])
        qdict[stim_pair_t[choice]].append(update_q)
        qdict[stim_pair_t[xchoice]].append(stim_pair_qval[xchoice])

        if r:
            di_theta['vd'][choice] = di_theta['vd'][
                choice] + di_theta['vd'][choice] * update_q
            di_theta['vi'][choice] = di_theta['vi'][
                choice] - di_theta['vi'][choice] * update_q
            if di_theta['vd'][choice] < 0:
                di_theta['vd'][choice] = .01
            if di_theta['vd'][choice] > 5.0:
                di_theta['vd'][choice] = 5.0

        else:
            di_theta['vd'][choice] = di_theta['vd'][
                choice] - di_theta['vd'][choice] * update_q
            di_theta['vi'][choice] = di_theta['vi'][
                choice] + di_theta['vi'][choice] * update_q
            if di_theta['vi'][choice] < 0:
                di_theta['vi'][choice] = .01
            if di_theta['vi'][choice] > 5.0:
                di_theta['vi'][choice] = 5.0

        di_qvals[stim_pair_t[choice]]['vd'].append(di_theta['vd'][choice])
        di_qvals[stim_pair_t[choice]]['vi'].append(di_theta['vi'][choice])
        di_qvals[stim_pair_t[xchoice]]['vd'].append(di_theta['vd'][xchoice])
        di_qvals[stim_pair_t[xchoice]]['vi'].append(di_theta['vi'][xchoice])


def rew_func(rprob):
    if rs() < rprob:
        return 1
    else:
        return 0


def di_lca(Id=3.5, Ii=3, dt=.005, si=2.5, tau=.05, ntrials=10, tmax=3.0, w=-.2, k=.93, rmax=70, b=35, g=15):
    """
    tau:       time constant (cf NMDA receptors)
    k:         leak (0<k<1) | rec. excitation (1<k<~2)
    w:         strength of cross-inhibition

    rmax:      max rate of cells
    b:         input needed for 1/2-max firing
    g:         determines steepness of sigmoidal f-I curve
    """

    timepoints = np.arange(0, tmax, dt)
    rd = np.zeros(len(timepoints))
    ri = np.zeros(len(timepoints))
    rd[0] = .01
    ri[0] = 3

    Ed = si * np.sqrt(dt / tau) * rs(len(rd))
    Ei = si * np.sqrt(dt / tau) * rs(len(ri))

    NInput = lambda x, r: rmax / (1 + np.exp(-(x - b) / g)) - r

    for i in timepoints:
        rd[i] = rd[i - 1] + dt / tau * \
            NInput(Id + k * rd[i - 1] + -w * ri[i - 1], rd[i - 1]) + Ed[i]
        ri[i] = ri[i - 1] + dt / tau * \
            NInput(Ii + k * ri[i - 1] + -w * rd[i - 1], ri[i - 1]) + Ei[i]

    return rd, ri


def di_decision(p):

    #p = vectorize_params(p, pc_map=pc_map, ncond=nc)

    Pd = 0.5 * (1 + p['vd'] * dx / si)
    Pi = 0.5 * (1 + p['vi'] * dx / si)

    Tex = np.ceil((tb - p['tr']) / dt).astype(int)
    #state = np.where(rs(ntot)>.5, 'l', 'r')
    # state=np.sort(state)

    Pd, Pi, Tex = update_execution(p)
    direct = np.where((rs((nc, Tex.max())).T < Pd), dx, -dx).T
    indirect = np.where((rs((nc, Tex.max())).T < Pi), dx, -dx).T
    execution = np.cumsum(direct - indirect, axis=1)

    choice = np.nan
    while np.isnan(choice):
        choice, p = analyze_execution(execution, p)

    return int(choice)


def analyze_execution(execution, p):

    rt = p['tr'] + np.argmax((execution.T >= p['a']).T, axis=1) * dt
    rt[rt == p['tr'][0]] = 0

    if np.all(rt == 0):
        # p['a']=p['a']*.99
        return np.nan, p

    return rt.argmin(), p
