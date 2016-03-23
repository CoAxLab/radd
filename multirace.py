#!/usr/local/bin/env python
from __future__ import division
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na


temporal_dynamics = lambda p, t: np.cosh(p['xb'][:, na] * t)
updateQ = lambda q, winner, r, A: q[winner][-1] + A*(r - q[winner][-1])
boltzmann_choiceP = lambda q, winner, losers, B: np.exp(B*q[winner][-1])/np.sum([np.exp(B*q[l][-1]) for l in losers])


def vectorize_params(p, pc_map, nresp=4):
    pvc = ['a', 'tr', 'vd', 'vi', 'xb']
    #remove conditional parameters from pvc
    [pvc.remove(param) for param in pc_map.keys()]
    for pkey in pvc:
        p[pkey]=p[pkey]*np.ones(nresp)
    for pkey, pkc in pc_map.items():
        if nresp==1:
            break
        elif pkc[0] not in p.keys():
            p[pkey] = p[pkey]*np.ones(len(pkc))
        else:
            p[pkey] = array([p[pc] for pc in pkc])
    return p


def rew_func(rprob):
    if rs()<rprob:
        return 1
    else:
        return 0


def run_trials(p, cards, nblocks=1, si=.1, alpha_pos=.06, alpha_neg=.06):
    """simulate series of trials with learning
    Arguments:
        p (dict): parameter dictionary
        cards (DataFrame): pandas DF (ntrials x nalt) with choice outcome vaulues
    Returns:
        choices (list): choice made on each trial
        rts (dict): rt for each trial (winner rt)
        all_traces (list): execution process traces truncated to length of winner
        qdict (dict): sequence of Q-value updates for each alt
    """

    trials_n = np.asarray([cards.index]*nblocks).flatten()
    choices, all_traces = [], []
    names = cards.columns
    rts={k:[] for k in names}
    qdict={k:[0] for k in names}

    for i in trials_n:
        vals = cards.iloc[i, :].values
        winner=np.nan
        while np.isnan(winner):
            execution = simulate_race(p, si=si)
            winner, rt, traces, p, qdict = analyze_multiresponse(execution, p, qdict=qdict, vals=vals, names=names, alpha_pos=alpha_pos, alpha_neg=alpha_neg)
        choice_name = names[winner]
        choices.append(winner); rts[choice_name].append(rt[winner]); all_traces.append(traces);
    return choices, rts, all_traces, qdict


def simulate_race(p, pc_map={'vd': ['vd_a', 'vd_b', 'vd_c', 'vd_d'], 'vi': ['vi_a', 'vi_b', 'vi_c', 'vi_d']}, dt=.001, si=.1, tb=2.5):

    nresp = len(pc_map.values()[0])
    p = vectorize_params(p, pc_map=pc_map, nresp=nresp)

    dx=np.sqrt(si*dt)
    Pd = 0.5*(1 + p['vd']*dx/si)
    Pi = 0.5*(1 + p['vi']*dx/si)

    Tex = np.ceil((tb-p['tr'])/dt).astype(int)
    xtb = temporal_dynamics(p, np.cumsum([dt]*Tex.max()))

    direct = np.where((rs((nresp, Tex.max())).T < Pd),dx,-dx).T
    indirect = np.where((rs((nresp, Tex.max())).T < Pi),dx,-dx).T
    execution = xtb[0] * np.cumsum(direct-indirect, axis=1)

    return execution


def analyze_multiresponse(execution, p, qdict={}, vals=[], names=[], alpha_pos=.06, alpha_neg=.06,  dt=.001):
    """analyze multi-race execution processes"""

    nsteps_to_rt = np.argmax((execution.T>=p['a']).T, axis=1)
    rts = p['tr'] + nsteps_to_rt*dt

    # set non responses to 999
    rts[rts==p['tr'][0]]=999
    if np.all(rts==999):
        # if no response occurs, increase exponential bias
        #p['xb']=p['xb']*1.01
        #p['a']=p['a']*.99
        return np.nan, rts, execution, p, qdict

    # get accumulator with fastest RT (winner) in each cond
    winner = np.argmin(rts)

    # get rt of winner in each cond
    winrt = rts[winner]

    # slice all traces at time the winner crossed boundary
    traces = [execution[i, :nsteps_to_rt[winner]] for i in xrange(len(rts))]

    reward = vals[winner]
    qval = qdict[names[winner]][-1]
    # check valence of RPE
    if reward>=qval:
        rpe=1
        alpha = alpha_pos
    else:
        alpha = alpha_neg
        rpe=-1

    # update action value
    q_update = qval + alpha * (reward - qval)
    qdict[names[winner]].append(q_update)
    # update direct & indirect drift-rates
    p = reweight_drift(p, winner, reward, alpha, rpe)

    return winner, rts, traces, p, qdict


def reweight_drift(p, winner, reward, alpha, rpe):
    """ update direct & indirect drift-rates for multirace winner """

    d0 = p['vd'][winner]
    i0 = p['vi'][winner]
    if rpe>0:
        # update direct drift-rate for choice(t)
        p['vd'][winner] = p['vd'][winner]*(1+alpha) #d0 + (reward-d0)*alpha
    else:
        # update indirect drift-rate for choice(t)
        p['vi'][winner] = p['vi'][winner]*(1+alpha) #i0 + (abs(reward)-i0)*alpha
    return p


def igt_scores(choices):

    A=len(choices[choices==0])
    B=len(choices[choices==1])
    C=len(choices[choices==2])
    D=len(choices[choices==3])
    # payoff (P) score
    P = (C+D) - (A+B)
    # sensitivity (Q) score
    Q = (B+D) - (A+C)

    return P, Q
