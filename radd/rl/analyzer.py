#!/usr/local/bin/env python
from __future__ import division
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na
import pandas as pd
from radd import vis
from copy import deepcopy


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


def analyze_learning_dynamics(fd, broadcast=True):

    choices = fd['choices']
    vd_all = fd['vd_all']
    vi_all = fd['vi_all']
    vdiff_all = vd_all - vi_all
    qdict = fd['qdict']
    nrows = len(choices)

    choice_vec = [np.sort(qdict.keys())[i] for i in choices]
    fd['choice']=choice_vec
    rts_copy = deepcopy(fd['rts'])
    fd['rt'] = [rts_copy[choice].pop(0) for i, choice in enumerate(choice_vec)]
    qcopy = deepcopy(qdict)
    fd['qval'] = [qcopy[choice].pop(0) for choice in choice_vec]

    fd['vd'] = [vd_all.loc[i, choice] for choice in choice_vec]
    fd['vi'] = [vi_all.loc[i, choice] for choice in choice_vec]
    fd['vdiff'] = [vdiff_all.loc[i, choice] for choice in choice_vec]
    fd['vdiff_all'] = [vdiff_all.loc[i, choice] for choice in choice_vec]
    vopt = vdiff_all['c'].values + vdiff_all['d'].values
    vsub = vdiff_all['a'].values + vdiff_all['b'].values

    vimp = vdiff_all['b'].values + vdiff_all['d'].values
    vnon = vdiff_all['a'].values + vdiff_all['c'].values
    fd['v_opt_diff'] = vopt - vsub
    fd['v_imp_diff'] = vimp - vnon
    #fd['agent'] = [fd['agent']*nrows]

    #q_go = fd['qdict_go']
    #q_no = fd['qdict_no']
    #qgo_copy = deepcopy(q_go)
    #qno_copy = deepcopy(q_no)
    #fd['q_go'] = [qgo_copy[choice].pop(0) for choice in choice_vec]
    #fd['q_no'] = [qno_copy[choice].pop(0) for choice in choice_vec]

    return fd

def format_dataframes(fd):
    from copy import deepcopy
    from collections import OrderedDict
    agdf_cols = ['agent', 'trial', 'agroup', 'qval', 'vd', 'vi', 'vdiff',
                 'v_opt_diff', 'v_imp_diff', 'choice', 'rt', 'a_go', 'a_no', 'adiff', 'beta']
    nrows = len(fd['choices'])
    fdbroad = dict(deepcopy(fd))
    #for key in ['agroup', 'agent', 'a_go', 'a_no', 'adiff', 'beta']:
    #    fdbroad[key] = [fd[key]]*nrows
    # agdf_cols = ['agent', 'trial', 'agroup', 'bgroup', 'group', 'qval', 'vd', 'vi', 'vdiff',
    #              'v_opt_diff', 'v_imp_diff', 'choice', 'rt', 'a_go', 'a_no', 'adiff', 'beta']
    agdf = pd.DataFrame(OrderedDict((col, fdbroad[col]) for col in agdf_cols))
    igtdf_cols=['agent', 'agroup', 'a_go', 'a_no', 'beta', 'P', 'Q']
    fd['P'], fd['Q'] = igt_scores(np.asarray(fd['choices']))

    igtdf = pd.Series(OrderedDict((col, fd[col]) for col in igtdf_cols))

    return igtdf, agdf
