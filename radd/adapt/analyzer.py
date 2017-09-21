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


def analyze_learning_dynamics(fd, broadcast=True, targets=['A', 'B', 'C', 'D']):

    choices = fd['choices']
    vd_all = fd['vd_all']
    vi_all = fd['vi_all']
    vdiff_all = vd_all - vi_all
    qdict = fd['qdict']
    nrows = len(choices)

    choice_vec = [targets[i] for i in choices]
    fd['choice']=choice_vec
    rts_copy = deepcopy(fd['rts'])
    fd['rt'] = [rts_copy[choice].pop(0) for i, choice in enumerate(choice_vec)]
    qcopy = deepcopy(qdict)
    fd['qval'] = [qcopy[choice].pop(0) for choice in choice_vec]

    fd['vd'] = [vd_all.loc[i, choice] for i, choice in enumerate(choice_vec)]
    fd['vi'] = [vi_all.loc[i, choice] for i, choice in enumerate(choice_vec)]
    fd['vdiff'] = [vdiff_all.loc[i, choice] for i, choice in enumerate(choice_vec)]
    fd['vdiff_all'] = [vdiff_all.loc[i, choice] for i, choice in enumerate(choice_vec)]
    vopt = vdiff_all['C'].values + vdiff_all['D'].values
    vsub = vdiff_all['A'].values + vdiff_all['B'].values

    vimp = vdiff_all['B'].values + vdiff_all['D'].values
    vnon = vdiff_all['A'].values + vdiff_all['C'].values
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


def calcPostErrAdjust(df, ntrials=200):
    if type(df.ttype.values[0]) is str:
        ssErrDF = df[(df.ttype=='stop')&(df.response==1)&(df.rt<.68)]
    else:
        ssErrDF = df[(df.ttype==0.)&(df.response==1)&(df.rt<.68)]
    ssErrRT = ssErrDF.rt.mean()
    adjustIX = ssErrDF.index.values + 1
    adjustIX[-1] = ssErrDF.index.values[-1]
    adjustDF = df.loc[adjustIX, :]
    PostErrRT = adjustDF[(adjustDF.response==1)&(adjustDF.rt<.68)].rt.mean()
    PostErrAdjust = PostErrRT - ssErrRT
    return PostErrAdjust

def calcTargetAdjust(df, ntrials=200):
    if type(df.ttype.values[0]) is str:
        GoDF = df[(df.ttype=='go')&(df.response==1)&(df.rt<.68)]
    else:
        GoDF = df[(df.ttype==1.)&(df.response==1)&(df.rt<.68)]
    GoRT = GoDF.rt.mean()
    adjustIX = GoDF.index.values + 1
    adjustIX[-1] = GoDF.index.values[-1]
    adjustDF = df.loc[adjustIX]
    GoAdjustRT = adjustDF[(adjustDF.response==1)&(adjustDF.rt<.68)].rt.mean()
    TargetAdjust =  GoAdjustRT - GoRT
    return TargetAdjust
