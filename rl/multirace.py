#!/usr/local/bin/env python
from __future__ import division
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na
import pandas as pd
from radd import vis
from copy import deepcopy

temporal_dynamics = lambda p, t: np.cosh(p['xb'][:, na] * t)
updateQ = lambda q, winner, r, A: q[winner][-1] + A*(r - q[winner][-1])
softmax_update = lambda q, name, B: np.exp(B*q[name][-1])/np.sum([np.exp(B*q[k][-1]) for k in q.keys()])

def run_full_sims(p, env=pd.DataFrame, alphas_go=[], alphas_no='same', betas=[], nblocks=2, nagents=100, si=.01):

    agents=np.arange(nagents)
    trials = np.arange(1, (len(env)*nblocks)+1)
    agent_list = []
    i=0
    if alphas_no=='same':
        alphas_no=deepcopy(alphas_go)
    alphas = zip(alphas_go, alphas_no)

    for bgroup in xrange(betas.size):
        beta = betas[bgroup]
        for agroup in xrange(len(alphas)):
            a_go, a_no = alphas[agroup]
            for agent_i in agents:
                pcopy=deepcopy(p)
                sim_out = run_trials(pcopy, env, nblocks=nblocks, si=si, a_go=a_go, a_no=a_no, beta=beta)
                choices, rts, all_traces, qdict_go, qdict_no, choicep, vd_all, vi_all = sim_out

                format_dict = {'agent':agent_i, 'trial':trials, 'a_go':a_go, 'a_no':a_no,
                'choices':choices, 'rts':rts, 'group': i, 'agroup': agroup, 'bgroup': bgroup,
                'qdict_go':qdict_go, 'qdict_no':qdict_no, 'qdict': qdict, 'choicep':choicep,
                'vd_all':vd_all, 'vi_all':vi_all, 'beta':beta}

                format_dict_updated = analyzr.analyze_learning_dynamics(format_dict)
                igtdf, agdf = analyzr.format_dataframes(format_dict_updated)
                agent_list.append([agdf, igtdf])

            i+=1

    trial_df = pd.concat([ag[0] for ag in agent_list]).groupby(['group', 'trial']).mean().reset_index()
    igt_df = pd.concat([ag[1] for ag in agent_list], axis=1).T.groupby('group').mean().reset_index()

    return [trial_df, igt_df]

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


def run_trials(p, cards, nblocks=1, si=.01, a_go=.06, a_no=.06, beta=5):
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
    qdict_go={k:[100] for k in names}
    qdict_no={k:[100] for k in names}
    choice_prob={k:[.25] for k in names}

    vdhist = pd.DataFrame(data=np.zeros((ntrials, len(names))), columns=names, index=np.arange(ntrials))
    vihist = vdhist.copy()

    for i in xrange(ntrials):
        vals = trials.iloc[i, 1:].values
        winner=np.nan
        while np.isnan(winner):
            execution = simulate_race(p, si=si)
            winner, rt, traces, p, qdict, qdict_go, qdict_no, choice_prob = analyze_multiresponse(execution, p, qdict=qdict, qdict_go=qdict_go, qdict_no=qdict_no, vals=vals, names=names, a_go=a_go, a_no=a_no, beta=beta, choice_prob=choice_prob)

        vdhist.iloc[i, :] = p['vd']
        vihist.iloc[i, :] = p['vi']
        choice_name = names[winner]
        choices.append(winner); rts[choice_name].append(rt[winner]); all_traces.append(traces)

    return choices, rts, all_traces, qdict, qdict_go, qdict_no, choice_prob, vdhist, vihist


def simulate_race(p, pc_map={'vd': ['vd_a', 'vd_b', 'vd_c', 'vd_d'], 'vi': ['vi_a', 'vi_b', 'vi_c', 'vi_d']}, dt=.001, si=.01, tb=2.5):

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


def analyze_multiresponse(execution, p, qdict={}, qdict_go={}, qdict_no={}, vals=[], names=[], a_go=.06, a_no=.06,  dt=.001, beta=5, choice_prob={}):
    """analyze multi-race execution processes"""

    nsteps_to_rt = np.argmax((execution.T>=p['a']).T, axis=1)
    rts = p['tr'] + nsteps_to_rt*dt

    # set non responses to 999
    rts[rts==p['tr'][0]]=999
    if np.all(rts==999):
        # if no response occurs, increase exponential bias (up to 3.0)
        if np.mean(p['xb']) <= 3.0:
            p['xb']=p['xb']*1.005
        return np.nan, rts, execution, p, qdict, qdict_go, qdict_no, choice_prob

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
        alpha=a_go
    else:
        alpha=a_no

    Qt = updateQ(qdict, winner_name, reward, alpha)
    #Q_Go_t = qdict_go[winner_name][-1] + a_go * (r - qdict_go[winner_name][-1])
    #Q_No_t = qdict_no[winner_name][-1] + a_no * -(r - qdict_no[winner_name][-1])

    qdict[winner_name].append(Qt)
    #qdict_go[winner_name].append(Q_Go_t)
    #qdict_no[winner_name].append(Q_No_t)

    for lname in loser_names:
        qdict[lname].append(qdict[lname][-1])
        #qdict_go[lname].append(qdict_go[lname][-1])
        #qdict_no[lname].append(qdict_no[lname][-1])

    #bound_expected = deepcopy(np.sum(p['vi']))
    for alt_i, name in enumerate(names):
        cp_old = choice_prob[name][-1]
        # update choice probability using boltzmann eq. w/ inv. temp beta
        cp_new = softmax_update(qdict, name, beta)
        choice_prob[name].append(cp_new)
        # calc. change in choice probability for alt_i
        delta_prob = cp_new - cp_old
        # update direct & indirect drift-rates with cp_delta
        p = reweight_drift(p, alt_i, delta_prob, a_go, a_no)
    #p['a'] = array([a_no*(bound_expected-np.sum(p['vi']))]*p['a'].size)

    return winner, rts, traces, p, qdict, qdict_go, qdict_no, choice_prob


def reweight_drift(p, alt_i, delta_prob, a_go, a_no):
    """ update direct & indirect drift-rates for multirace winner """

    vd_exp = p['vd'][alt_i]
    vi_exp = p['vi'][alt_i]

    p['vd'][alt_i] = vd_exp + (delta_prob*a_go)
    p['vi'][alt_i] = vi_exp + (-delta_prob*a_no)

    return p
