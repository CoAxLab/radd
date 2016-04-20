#!/usr/local/bin/env python
from __future__ import division
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na
import pandas as pd

temporal_dynamics = lambda p, t: np.cosh(p['xb'][:, na] * t)
updateQ = lambda q, winner, r, A: q[winner][-1] + A*(r - q[winner][-1])
boltzmann_choiceP = lambda q, name, B: np.exp(B*q[name][-1])/np.sum([np.exp(B*q[k][-1]) for k in q.keys()])


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


def run_trials(p, cards, nblocks=1, si=.1, a_pos=.06, a_neg=.06, beta=5):
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

    trials = cards.append([cards]*(nblocks-1)).reset_index()
    trials.rename(columns={'index':'t'}, inplace=True)
    ntrials=len(trials)
    choices, all_traces = [], []
    names = np.sort(cards.columns.values)
    rts={k:[] for k in names}
    qdict={k:[0] for k in names}
    choice_prob={k:[.25] for k in names}

    vdhist = pd.DataFrame(data=np.zeros((ntrials, len(names))), columns=names, index=np.arange(ntrials))
    vihist = vdhist.copy()

    for i in xrange(ntrials):
        vals = trials.iloc[i, 1:].values
        winner=np.nan
        while np.isnan(winner):
            execution = simulate_race(p, si=si)
            winner, rt, traces, p, qdict, choice_prob = analyze_multiresponse(execution, p, qdict=qdict, vals=vals, names=names, a_pos=a_pos, a_neg=a_neg, beta=beta, choice_prob=choice_prob)

        vdhist.iloc[i, :] = p['vd']
        vihist.iloc[i, :] = p['vi']
        choice_name = names[winner]
        choices.append(winner); rts[choice_name].append(rt[winner]); all_traces.append(traces)

    return choices, rts, all_traces, qdict, choice_prob, vdhist, vihist


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


def analyze_multiresponse(execution, p, qdict={}, vals=[], names=[], a_pos=.06, a_neg=.06,  dt=.001, beta=5, choice_prob={}):
    """analyze multi-race execution processes"""

    nsteps_to_rt = np.argmax((execution.T>=p['a']).T, axis=1)
    rts = p['tr'] + nsteps_to_rt*dt

    # set non responses to 999
    rts[rts==p['tr'][0]]=999
    if np.all(rts==999):
        # if no response occurs, increase exponential bias (up to 3.0)
        if p['xb'] <= 3.0:
            p['xb']=p['xb']*1.005
        return np.nan, rts, execution, p, qdict, choice_prob

    # get accumulator with fastest RT (winner) in each cond
    winner = np.argmin(rts)

    # get rt of winner in each cond
    winrt = rts[winner]

    # slice all traces at time the winner crossed boundary
    traces = [execution[i, :nsteps_to_rt[winner]] for i in xrange(len(rts))]

    reward = vals[winner]
    winner_name = names[winner]
    loser_names = names[names!=winner_name]

    # update action value
    qval = qdict[names[winner]][-1]
    if reward>=qval:
        alpha=a_pos
    else:
        alpha=a_neg

    Qt = updateQ(qdict, winner_name, reward, alpha)
    qdict[winner_name].append(Qt)
    for lname in loser_names:
        qdict[lname].append(qdict[lname][-1])

    for alt_i, name in enumerate(names):
        cp_old = choice_prob[name][-1]
        # update choice probability using boltzmann eq. w/ inv. temp beta
        cp_new = boltzmann_choiceP(qdict, name, beta)
        choice_prob[name].append(cp_new)
        # calc. change in choice probability for alt_i
        cp_delta = cp_new - cp_old
        # update direct & indirect drift-rates with cp_delta
        p = reweight_drift(p, alt_i, cp_delta, a_pos, a_neg)


    return winner, rts, traces, p, qdict, choice_prob


def reweight_drift(p, alt_i, cp_delta, a_pos, a_neg):
    """ update direct & indirect drift-rates for multirace winner """

    p['vi'][alt_i] = p['vd'][alt_i] + (vd_exp*a_pos * cp_delta)
    p['vi'][alt_i] = p['vi'][alt_i] + (vi_exp*a_neg * -cp_delta)

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


def plot_traces_rts(p, all_traces, rts):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from radd import vis

    f, axes = vis.build_multi_axis(p, tb=1000)
    clrs = sns.color_palette('muted', 5)

    for i in xrange(len(all_traces)):
        for ii, ax in enumerate(axes.flatten()):
            x=np.arange(len(all_traces[i][ii]))
            ax.plot(x, all_traces[i][ii], color=clrs[ii], alpha=.51)

    for i, ax in enumerate(axes.flatten()):
        divider = make_axes_locatable(ax)
        axx = divider.append_axes("top", size=.7, pad=0.01, sharex=ax)
        for spine in ['top', 'left', 'bottom', 'right']:
            axx.spines[spine].set_visible(False)
        axx.set_xticklabels([])
        axx.set_yticklabels([])

        k = np.sort(rts.keys())[i]
        rt_dist_k = np.asarray(rts[k])*1000-200
        sns.distplot(rt_dist_k, ax=axx, label=k, color=clrs[i])
