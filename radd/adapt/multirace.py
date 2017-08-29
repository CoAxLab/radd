#!/usr/local/bin/env python
from __future__ import division
import os
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na
import pandas as pd
from radd.IGT import visr, analyzer
from copy import deepcopy
from radd import theta
from scipy.stats.mstats import mquantiles as mq

temporal_dynamics = lambda p, t: np.cosh(p['xb'][:, na] * t)
updateQ = lambda q, winner, r, A: q[winner][-1] + A*(r - q[winner][-1])
softmax_update = lambda q, name, B: np.exp(B*q[name][-1])/np.sum([np.exp(B*q[k][-1]) for k in q.keys()])
go_resp = lambda trace, upper, dt: np.argmax((trace.T >= upper).T, axis=0) * dt
ss_resp = lambda trace, x, dt: np.argmax((trace.T <= 0).T, axis=0) * dt
go_RT = lambda ontime, rbool: ontime[:, na] + (rbool*np.where(rbool==0., np.nan, 1))
ss_RT = lambda ontime, rbool: ontime[:, :, na] + (rbool*np.where(rbool==0., np.nan, 1))
# RTQ = lambda zpd, prob: map((lambda x: mq(x[0][x[0] < x[1]], prob)), zpd)
RTQ = lambda zpd: [mquantiles(rt[rt < deadline], prob) for rt, deadline in zpd]

def run_full_sims(p, env=pd.DataFrame, alphas_go=[], alphas_no=None, betas=[], a_Q=.1, nblocks=2, nagents=30, si=.01, agent_list=[]):

    n_a = len(alphas_go)
    n_b = len(betas)
    trials = np.arange(1, (len(env)*nblocks)+1)

    if alphas_no is None:
        alphas_no=deepcopy(alphas_go)

    agroups = np.hstack([np.hstack([np.arange(n_a)]*n_b)]*nagents)
    bgroups = np.sort(np.hstack([np.sort(np.hstack([np.arange(n_b)]*n_a))]*nagents))
    agents = np.sort(np.hstack([np.arange(nagents)]*n_a*n_b))
    group = 0
    for agroup, bgroup, agent_i in zip(agroups, bgroups, agents):

        beta = betas[bgroup]
        a_go = alphas_go[agroup]
        a_no = alphas_no[agroup]

        pcopy=deepcopy(p)
        #p_rand = theta.random_inits(pcopy.keys())
        sim_out = run_trials(pcopy, env, nblocks=nblocks, si=si, a_go=a_go, a_no=a_no, beta=beta, a_Q=a_Q)
        choices, rts, all_traces, qdict, choicep, vd_all, vi_all = sim_out

        format_dict = {'agent': agent_i+1, 'trial': trials, 'a_go': a_go, 'a_no': a_no,
        'adiff': a_go-a_no, 'choices':choices, 'rts': rts, 'group': group, 'agroup': agroup+1,
        'bgroup': bgroup+1, 'qdict': qdict, 'choicep':choicep, 'vd_all':vd_all, 'vi_all':vi_all,
        'beta': beta}

        format_dict_updated = analyzer.analyze_learning_dynamics(format_dict)
        igtdf, agdf = analyzer.format_dataframes(format_dict_updated)
        agent_list.append([agdf, igtdf])
        group += 1

    trial_df = pd.concat([ag[0] for ag in agent_list]) #.groupby(['group', 'trial']).mean().reset_index()
    igt_df = pd.concat([ag[1] for ag in agent_list], axis=1).T #.groupby('group').mean().reset_index()

    return [trial_df, igt_df]

def vectorize_params(p, pcmap={'vd': ['vd_a', 'vd_b', 'vd_c', 'vd_d'], 'vi': ['vi_a', 'vi_b', 'vi_c', 'vi_d']}, nresp=4, sstrial=False):
    constants = ['a', 'tr', 'vd', 'vi', 'xb']
    if sstrial:
        constants.append('ssv')
    #remove conditional parameters from constants
    [constants.remove(param) for param in pcmap.keys()]
    for pkey in constants:
        p[pkey]=p[pkey]*np.ones(nresp)
    for pkey, pkc in pcmap.items():
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


def run_trials(p=None, cards=None, nblocks=2, si=.1, a_go=.2, a_no=.2, beta=5, aX=.001, dt=.001, plot=False, tb=1.1):
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
    if p is None:
        p={'vd':.09, 'vi':.05}; p['a']=.006; p['tr']=.3; p['xb']=.001
    trials = cards.copy()
    trials = trials.append([trials]*(nblocks-1)).reset_index()
    trials.rename(columns={'index':'t'}, inplace=True)
    ntrials=len(trials)
    choices, all_traces = [], []
    names = np.sort(cards.columns.values)
    rts={k:[] for k in names}
    qdict={k:[0] for k in names}
    choice_prob={k:[.25] for k in names}

    vdhist = pd.DataFrame(data=np.zeros((ntrials, len(names))), columns=names, index=np.arange(ntrials))
    vihist = vdhist.copy()
    needsfixed = 0
    prevWinner=np.nan
    for i in range(ntrials):
        vals = trials.iloc[i, 1:].values
        iquit=0
        winner=np.nan
        while np.isnan(winner) and iquit<35:
            execution = simulate_multirace(p, si=si, dt=dt)
            winner, rt, traces, p, qdict, choice_prob, choices = analyze_multiresponse(execution, p, qdict=qdict, vals=vals, names=names, a_go=a_go, a_no=a_no, aX=aX, beta=beta, choice_prob=choice_prob, dt=dt, choices=choices, prevWinner=prevWinner)
            iquit+=1
            if np.isnan(np.mean(p['xb'])):
                iquit=36
        prevWinner=winner
        if winner>=len(names) or np.isnan(winner):
            needsfixed+=1
            winner = int(np.random.choice(np.arange(len(names))))
        choice_name = names[choices[i]]
        oldval = trials.loc[i, choice_name]
        new_col = trials[choice_name].shift(-1)
        new_col.set_value(new_col.index[-1], oldval)
        trials=trials.copy()
        trials.loc[:, choice_name] = new_col

        vdhist.iloc[i, :] = p['vd']
        vihist.iloc[i, :] = p['vi']
        choice_name = names[winner]
        rts[choice_name].append(rt[winner]); all_traces.append(traces)
    percent_random_choice = (needsfixed*100.)/ntrials
    if percent_random_choice>=10.:
        print("trials with no winner {:.2f}%".format(percent_random_choice))
    if plot:
        visr.plot_traces_rts(p, all_traces, rts)
        return all_traces, rts
    return choices, rts, all_traces, qdict, choice_prob, vdhist, vihist


def simulate_multirace(p, pcmap={'vd': ['vd_a', 'vd_b', 'vd_c', 'vd_d'], 'vi': ['vi_a', 'vi_b', 'vi_c', 'vi_d']}, dt=.001, si=.01, tb=.9, single_process=0, return_di=False):

    nresp = len(pcmap.values()[0])
    dx = si * np.sqrt(dt)
    p = vectorize_params(p, pcmap=pcmap, nresp=nresp)

    Tex = np.ceil((tb-p['tr'])/dt).astype(int)
    xtb = temporal_dynamics(p, np.cumsum([dt]*Tex.max()))

    if single_process:
        Pe = 0.5*(1 + (p['vd']-p['vi'])*dx/si)
        execution = xtb * np.cumsum(np.where((rs((nresp, Tex.max())).T < Pe), dx, -dx).T, axis=1)
    else:
        Pd = 0.5 * (1 + p['vd'] * dx / si)
        Pi = 0.5 * (1 + p['vi'] * dx / si)
        direct = xtb * np.where((rs((nresp, Tex.max())).T < Pd),dx,-dx).T
        indirect = np.where((rs((nresp, Tex.max())).T < Pi),dx,-dx).T
        execution = np.cumsum(direct-indirect, axis=1)
        if return_di:
            return np.cumsum(direct, axis=1), np.cumsum(indirect, axis=1), execution
    return execution

def analyze_multiresponse(execution, p, qdict={}, vals=[], names=[], a_go=.2, a_no=.2, dt=.001, beta=5, choice_prob={}, aX=.001, choices=[], prevWinner=np.nan):
    """analyze multi-race execution processes"""
    nsteps_to_rt = np.argmax((execution.T>=p['a']).T, axis=1)
    rts = p['tr'] + nsteps_to_rt*dt
    # set non responses to 999
    rts[rts==p['tr'][0]]=999
    if np.all(rts==999):
        # if no response occurs, increase exponential bias (up to 3.0)
        if np.mean(p['xb']) <= 4.0:
            p['xb']=p['xb']*1.005
        return np.nan, rts, execution, p, qdict, choice_prob, choices
    # get accumulator with fastest RT (winner) in each cond
    winner = np.argmin(rts)
    choices.append(winner)
    # get rt of winner in each cond
    winrt = rts[winner]
    # slice all traces at time the winner crossed boundary
    traces = [execution[i, :nsteps_to_rt[winner]] for i in range(len(rts))]
    reward = vals[winner]
    winner_name = names[winner]
    if prevWinner==winner and reward>0:
        #p['xb'][winner] = p['xb'][winner] + aX * reward
        p['a'][winner] = p['a'][winner]*(1. - aX)
    elif prevWinner==winner and reward<0:
        p['a'][winner] = p['a'][winner]*(1. + aX)
        #p['xb'][winner] = p['xb'][winner] + aX * reward
    loser_names = names[names!=winner_name]
    # update action value
    qval = qdict[names[winner]][-1]
    if reward>=qval:
        alpha=a_go
    else:
        alpha=a_no
    if reward<0:
        reward
    Qt = updateQ(qdict, winner_name, reward, alpha)#, a_Q)
    qdict[winner_name].append(Qt)
    for lname in loser_names:
        qdict[lname].append(qdict[lname][-1])

    for alt_i, name in enumerate(names):
        cp_old = choice_prob[name][-1]
        # update choice probability using boltzmann eq. w/ inv. temp beta
        cp_new = softmax_update(qdict, name, beta)
        choice_prob[name].append(cp_new)
        # calc. change in choice probability for alt_i
        delta_prob = cp_new - cp_old
        # update direct & indirect drift-rates with cp_delta
        p = reweight_drift(p, alt_i, delta_prob, a_go, a_no)
        #p = weight_drift(p, alt_i, deltaQ, a_go, a_no)
    #p['a'] = array([a_no*(bound_expected-np.sum(p['vi']))]*p['a'].size)
    return winner, rts, traces, p, qdict, choice_prob, choices

def weight_drift(p, alt_i, deltaQ, a_go, a_no):
    vd_exp = p['vd'][alt_i]
    vi_exp = p['vi'][alt_i]
    p['vd'][alt_i] = vd_exp + a_go * (vd_exp - vd_exp * np.exp(1-(deltaQ)))
    p['vi'][alt_i] = vi_exp + a_no * (vi_exp - vi_exp * np.exp(1-(deltaQ)))
    return p

def reweight_drift(p, alt_i, delta_prob, a_go, a_no):
    """ update direct & indirect drift-rates for multirace winner
    """

    #p['vd'][alt_i] = vd_exp * np.exp(delta_prob)
    vd_exp = p['vd'][alt_i]
    #p['vd'][alt_i] = vd_exp + a_go * (vd_exp - (vd_exp * (1+delta_prob*2)))
    # p['vd'][alt_i] = vd_exp + a_go * (vd_exp - (vd_exp * np.exp(-delta_prob)))
    vi_exp = p['vi'][alt_i]
    #p['vi'][alt_i] = vd_exp + a_go * (vd_exp - (vd_exp * (1-delta_prob*2)))
    # p['vi'][alt_i] = vi_exp + a_no * (vi_exp - (vi_exp * np.exp(delta_prob)))
    #p['vi'][alt_i] = vi_exp * np.exp(-delta_prob)
    # if delta_prob > 0:
    #p['vd'][alt_i] = vd_exp + (a_go*delta_prob)
    #p['vi'][alt_i] = vi_exp + (a_no*-delta_prob)
    # else:
    p['vd'][alt_i] = vd_exp + (a_go * delta_prob)
    p['vi'][alt_i] = vi_exp + (a_no * -delta_prob)
    #vTrial = vi_exp + (a_go * (vd_exp - vd_exp * np.exp((delta_prob))))
    #vTrial = vi_exp + (a_no * (vi_exp - vi_exp * np.exp((delta_prob))))
    # p['vd'][alt_i] = vd_exp + (vd_exp*a_go)*delta_prob
    # p['vi'][alt_i] = vi_exp + (vi_exp*a_no)*-delta_prob
    return p
