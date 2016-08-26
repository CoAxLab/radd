#!/usr/local/bin/env python
# Authors:
#   Kyle Dunovan (ked64@pitt.edu) & 
#   Jeremy Huang (jeremyhuang@cmu.edu)

from __future__ import division
import numpy as np
import numba as nb
from numba.decorators import jit
from numba import float64, int64
from radd.numbaradd.helperfx import *


@jit(nopython=True)
def slice_theta_array(theta_array, index_arrays, n_vals):
    i, ix = 0, 0
    param_array = np.empty(index_arrays.T.shape)
    for nlvls in n_vals:
        arr_ix = index_arrays[i]
        param_array[:, i] = theta_array[ix:ix+nlvls][arr_ix]
        i += 1
        ix += nlvls
    return param_array.T

@jit(nopython=True)
def sim_many_dpm(rvector, rvector_ss, dvg, xtb, v_prob, vs_prob, bound, onset, ssd, dx, dt, rts, ssrts):
    ncond, ntrials, ntime = rvector.shape
    ncond, nssd, nss_per, ntime = rvector_ss.shape
    for i in xrange(ncond):
        tr_ix = onset[i]
        for j in xrange(ntrials):
            ix = sim_dpm_trace_upper(rvector[i,j], dvg[i,j,tr_ix:], xtb[i], v_prob[i], bound[i], dx[i])
            if ix<0:
                rt_ix = 1000.
                rts[i,j] = rt_ix
            else:
                rt_ix = tr_ix + ix
                rts[i,j] = rt_ix * dt
            # Simulate Stop Process
            if j<nss_per:
                ssbase = dvg[i,j][ssd[i]]
                for k in xrange(nssd):
                    if rt_ix<ssd[i,k] or ix<0:
                        ssrts[i,k,j] = 999.
                        continue
                    ssrts[i,k,j] = sim_dpm_trace_lower(rvector_ss[i,k,j], ssbase[k], vs_prob[i], ssd[i,k], dx[i], dt)
    return rts, ssrts


@jit(int64(float64[:], float64[:], float64[:], float64, float64, float64), nopython=True)
def sim_dpm_trace_upper(rvector, trace, xtb, v_prob, bound, dx):
    evidence = 0
    trace[0] = evidence
    for ix in xrange(1, len(trace)):
        if rvector[ix] < v_prob:
            evidence += dx
        else:
            evidence -= dx
        weightedEvidence = evidence * xtb[ix]
        trace[ix] = weightedEvidence
        if weightedEvidence >= bound:
            return ix
    return -1

@jit(float64(float64[:], float64, float64, int64, float64, float64), nopython=True)
def sim_dpm_trace_lower(rvector, dvs, v_prob, onset, dx, dt):
    ix = onset
    timelimit = rvector.size + 1
    while dvs>0 and ix<timelimit:
        ix += 1
        if rvector[ix] < v_prob:
            dvs += dx
            continue
        dvs -= dx
    return ix*dt

@jit(float64(float64[:], float64), nopython=True)
def calc_dpm_rt(dv, dt):
    return dv.size * dt

@jit(float64[:](float64[:], float64[:], float64), nopython=True)
def norm_drift(v, si, dx):
    vprob = 0.5 * (1 + v * dx / si)
    return vprob

@jit(int64(float64[:], float64, float64), nopython=True)
def get_timesteps(onset, tb, dt):
    ntime = np.int((tb - onset.min()) / dt)
    return ntime

@jit(float64[:](float64[:], float64), nopython=True)
def get_onset_index(onset, dt):
    onsets = onset / dt
    return onsets

@jit(nopython=True)
def sim_many_go_trace(rvector, xtb, v_prob, bound, onset, dx, dt, rts):
    ncond, ntrials, nx = rvector.shape
    for i in xrange(ncond):
        for j in xrange(ntrials):
            rts[i,j] = sim_trace_upper(rvector[i,j], xtb[i], v_prob[i], bound[i], onset[i], dx, dt)
            #rts[i,j] = calc_rt(trace, onset[i], dt)
    #return dvg, rts
    return rts


@jit(nopython=True)
def sim_many_ss_trace(rvector, xtb, vs_prob, bound, ssd, dx, dt, rts):
    ncond, nssd, ntrials, ntime = rvector.shape
    for i in xrange(ncond):
        for j in xrange(nssd):
            for k in xrange(ntrials):
                rts[i,j,k] = sim_trace_upper(rvector[i,j,k], xtb, vs_prob[i], bound[i], ssd[i,j], dx, dt)
                #rts[i, j, k] = calc_rt(trace, ssd[i, j], dt)
    #return dvs, rts
    return rts

@jit(float64(float64[:], float64, float64), nopython=True)
def calc_rt(dv, onset, dt):
    return onset + dv[dv>0].size * dt

@jit(nopython=True)
def upper_cross_jit_ufunc(trace, onset, upper, dt, outarray):
    ni, nj, nk = trace.shape
    for i in xrange(ni):
        for j in xrange(nj):
            outarray[i][j] = numba_argmax_func(trace[i][j], onset[i], upper[i], dt)
    return outarray

@jit(nopython=True)
def upper_cross_jit(trace, onset, upper, dt, outarray):
    ni, nj, nk = trace.shape
    for i in xrange(ni):
        for j in xrange(nj):
            outarray[i][j] = onset[i] + np.argmax(trace[i][j]>=upper[i])* dt
    return outarray


@jit(float64(float64[:], float64, float64, float64), nopython=True)
def lower_cross_jit_single(trace, onset, lower, dt):
    return onset + (np.argmax(trace <= 0) * dt)

@jit(float64(float64[:], float64, float64, float64), nopython=True)
def upper_cross_jit_single(trace, onset, upper, dt):
    return onset + (np.argmax(trace >= upper) * dt)
