#!/usr/local/bin/env python
# Authors:
#   Kyle Dunovan (ked64@pitt.edu) &
#   Jeremy Huang (jeremyhuang@cmu.edu)
from __future__ import division
import numpy as np
import numba as nb
from scipy.stats import norm
from numba.decorators import jit
from numba import float64, int64, vectorize, boolean
from numpy.random import random_sample as randsample


def sample_posterior_belief(data, m0=300, k0=1, s2=1, v0=1, nsamples=800):

    ntrials = data.shape[0]
    muPosterior = np.zeros((ntrials, nsamples))
    varPosterior = np.zeros((ntrials, nsamples))

    for i in range(1, ntrials+1):
        mu_t, var_t = draw_mus_and_sigmas(data[:i], m0, k0, s2, v0, nsamples)
        muPosterior[i-1, :] = mu_t
        varPosterior[i-1, :] = var_t

    return muPosterior, varPosterior


def rolling_variance(old_avg, new, old, N):
    newavg = oldavg + (new - old)/N
    average = newavg
    variance += (new-old)*(new-newavg+old-oldavg)/(N-1)
    stddev = sqrt(variance)


def get_norm(mu, sd, sample):
    return norm(mu, sd).logpdf(sample)


@vectorize([float64(boolean, float64, float64)])
def ufunc_where(condition, x, y):
    if condition:
        return x
    else:
        return y

@vectorize([int64(float64, float64)], nopython=True)
def get_onset_index(onset, dt):
    onsets = int(onset / dt)
    return onsets


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



@jit(nb.typeof((1, 1))(float64[:], float64[:], float64, float64, float64, float64), nopython=True)
def sim_ddm_trace(rProb, trace, vProb, bound, gbase, dx):
    evidence = gbase
    trace[0] = evidence
    timebound = trace.size
    for ix in range(1, timebound):
        if rProb[ix] < vProb:
            evidence += dx
        else:
            evidence -= dx
        trace[ix] = evidence
        if evidence >= bound:
            return ix, 1
        elif evidence <= 0:
            return ix, 0
    return -1, -1


@jit((float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], float64[:], float64[:], float64[:], int64[:], float64[:], float64), nopython=True)
def sim_many_ddm_traces(rProb, dvg, rts, choices, vProb, bound, gbase, gOnset, dx, dt):
    ncond, ntrials, ntime = rProb.shape
    for i in range(ncond):
        tr = gOnset[i]
        dvg[i,:,:tr] = gbase[i]
        for j in range(ntrials):
            ix, choice = sim_ddm_trace(rProb[i,j,tr:], dvg[i,j,tr:], vProb[i], bound[i], gbase[i], dx[i])
            if ix<0:
                rt_ix = ntime + 1
                rts[i,j] = 1000.
            else:
                rt_ix = tr + ix
                rts[i,j] = rt_ix * dt
            choices[i, j] = choice



@jit(int64(float64[:], float64[:], float64[:], float64, float64, float64, float64), nopython=True)
def sim_dpm_trace_upper(rProb, trace, xtb, vProb, bound, gbase, dx):
    evidence = gbase
    trace[0] = evidence
    timebound = trace.size
    for ix in range(1, timebound):
        if rProb[ix] < vProb:
            evidence += dx
        else:
            evidence -= dx
        weightedEvidence = evidence * xtb[ix]
        trace[ix] = weightedEvidence
        if weightedEvidence >= bound:
            return ix
    return -1


@jit(float64(float64[:], float64, float64, int64, float64, float64), nopython=True)
def sim_dpm_trace_lower(rProbSS, ssbase, vsProb, onset, dx, dt):
    ix = onset
    evidence = ssbase
    timebound = rProbSS.size
    while evidence>0 and ix<timebound:
        ix += 1
        if rProbSS[ix] < vsProb:
            evidence += dx
            continue
        evidence -= dx
    return ix * dt


@jit((float64[:,:,:], float64[:,:,:,:], float64[:,:,:], float64[:,:], float64[:,:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:], int64[:], int64[:,:], float64[:], float64[:], float64), nopython=True)
def sim_many_dpm(rProb, rProbSS, dvg, rts, ssrts, xtb, drift, ssdrift, bound, gbase, gOnset, ssOnset, dx, si, dt):
    ncond, ntrials, ntime = rProb.shape
    ncond, nssd, nss_per, ntime = rProbSS.shape
    vProb = 0.5 * (1 + (drift * np.sqrt(dt))/si)
    vsProb = 0.5 * (1 + (ssdrift * np.sqrt(dt))/si)
    for i in range(ncond):
        tr = gOnset[i]
        ssOn = ssOnset[i]
        dvg[i,:,:tr] = gbase[i]
        for j in range(ntrials):
            ix = sim_dpm_trace_upper(rProb[i,j,tr:], dvg[i,j,tr:], xtb[i], vProb[i], bound[i], gbase[i], dx[i])
            if ix<0:
                rt_ix = ntime + 1
                rts[i,j] = rt_ix * dt # 1000.
            else:
                rt_ix = tr + ix
                rts[i,j] = rt_ix * dt
            # Simulate Stop Process
            if j<nss_per:
                ssbase = dvg[i,j][ssOn]
                for k in range(nssd):
                    if rt_ix < ssOn[k] or ix<0:
                        ssrts[i,k,j] = ntime * dt # 1000.
                        continue
                    ssrts[i,k,j] = sim_dpm_trace_lower(rProbSS[i,k,j], ssbase[k], vsProb[i], ssOn[k], dx[i], dt)



@jit(float64(float64[:], float64[:], float64, float64, int64, float64, float64), nopython=True)
def sim_dpm_trace_lower_trace(rProbSS, dvs, ssbase, vsProb, onset, dx, dt):
    ix = onset
    evidence = ssbase
    timebound = rProbSS.size
    while evidence>0 and ix<timebound:
        if rProbSS[ix] < vsProb:
            evidence += dx
        else:
            evidence -= dx
        dvs[ix] = evidence
        ix += 1
    return ix * dt


@jit((float64[:,:,:], float64[:,:,:,:], float64[:,:,:], float64[:,:,:,:], float64[:,:], float64[:,:,:], float64[:,:], float64[:], float64[:], float64[:], float64[:], int64[:], int64[:,:], float64[:], float64[:], float64), nopython=True)
def sim_many_dpm_traces(rProb, rProbSS, dvg, dvs, rts, ssrts, xtb, drift, ssdrift, bound, gbase, gOnset, ssOnset, dx, si, dt):
    ncond, ntrials, ntime = rProb.shape
    ncond, nssd, nss_per, ntime = rProbSS.shape
    vProb = 0.5 * (1 + (drift * np.sqrt(dt))/si)
    vsProb = 0.5 * (1 + (ssdrift * np.sqrt(dt))/si)
    for i in range(ncond):
        tr = gOnset[i]
        ssOn = ssOnset[i]
        dvg[i,:,:tr] = gbase[i]
        for j in range(ntrials):
            ix = sim_dpm_trace_upper(rProb[i,j,tr:], dvg[i,j,tr:], xtb[i], vProb[i], bound[i], gbase[i], dx[i])
            if ix<0:
                rt_ix = ntime + 1
                rts[i,j] = 1000.
            else:
                rt_ix = tr + ix
                rts[i,j] = rt_ix * dt
            # Simulate Stop Process
            if j<nss_per:
                ssbase = dvg[i,j][ssOn]
                for k in range(nssd):
                    if rt_ix < ssOn[k] or ix<0:
                        ssrts[i,k,j] = 1000.
                        continue
                    ssrts[i,k,j] = sim_dpm_trace_lower_trace(rProbSS[i,k,j], dvs[i,k,j], ssbase[k], vsProb[i], ssOn[k], dx[i], dt)




@jit(nb.typeof((1.0, 1.0))(float64[:], float64[:], float64, float64, int64, float64[:], float64, int64, float64, float64, float64, float64), nopython=True)
def sim_dpm_go_stop(rProb, xtb, vProb, bound, onsetIX, rProbSS, vsProb, ssdIX, dx, dt, evidence, ssbound):
    ssEvidence = 0
    ssStarted = 0
    weightedEvidence = 0
    timebound = rProb.shape[0]
    start = onsetIX
    if ssdIX <= onsetIX:
        ssStarted = 1
        start = ssdIX
    for ix in range(start, timebound):
        if rProb[ix] < vProb:
            evidence += dx
        else:
            evidence -= dx
        weightedEvidence = evidence * xtb[ix]
        if weightedEvidence >= bound:
            return 1., ix*dt
        if ix == ssdIX:
            ssStarted = 1
            ssEvidence = weightedEvidence
        if ssStarted:
            if rProbSS[ix] < vsProb:
                ssEvidence += dx
            else:
                ssEvidence -= dx
            if ssEvidence <= ssbound:
                return 0., ix*dt
    return 0., ix*dt


@jit(nb.typeof((1.0, 1.0))(float64[:], float64[:], float64, float64, int64, float64, float64, float64), nopython=True)
def sim_dpm_go(rProb, xtb, vProb, bound, onsetIX, dx, dt, evidence):
    timebound = rProb.shape[0]
    for ix in range(onsetIX, timebound):
        if rProb[ix] < vProb:
            evidence += dx
        else:
            evidence -= dx
        weightedEvidence = evidence * xtb[ix]
        if weightedEvidence >= bound:
            return 1., ix*dt
    return 0., ix*dt



@jit((float64[:,:], float64[:,:], float64[:,:], float64[:], int64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64), nopython=True)
def sim_dpm_learning(results, rProb, rProbSS, xtb, idxArray, drift, ssdrift, bound, onset, B, C, R, dx, dt, tb, maxTrials):

    onsetIX = np.int(onset)
    TargetRT = .52
    goStartTrial = 0.
    ssbound = 0.
    si = .1
    tb = tb + .02

    vsProb = 0.5 * (1 + (ssdrift * np.sqrt(dt))/si)

    for idx in idxArray:
        idxResults = results[results[:, 0]==idx]
        ntrials = idxResults.shape[0]
        vTrial = drift
        aTrial = bound
        errT = 10000

        for t in range(ntrials):
            ttype, ssOnset = idxResults[t, 1:3]
            sensitivity = np.exp2(-t*R) #(t/10.)**-R
            B_t = B * sensitivity
            C_t = C * sensitivity

            vProbTrial = 0.5 * (1 + (vTrial * np.sqrt(dt))/si)

            if ttype==0.:
                ssdIX = np.int(ssOnset)
                response, rt = sim_dpm_go_stop(rProb[t], xtb, vProbTrial, aTrial, onsetIX, rProbSS[t], vsProb, ssdIX, dx, dt, goStartTrial, ssbound)

                if response:
                    correct = 0.; score = -1.; errT = 0.
                    #vTrial = vTrial * np.exp((rt-tb) * B_t)
                    #vTrial = vTrial + B_t * ((aTrial/TargetRT) - (aTrial/rt))
                    vTrial = vTrial + B_t * ((aTrial/tb) - (aTrial/rt))
                else:
                    correct = 1.; score = 0.; errT += 1.
                aTrial = bound + C_t * np.exp(-errT)

            else:
                errT += 1
                response, rt = sim_dpm_go(rProb[t], xtb, vProbTrial, aTrial, onsetIX, dx, dt, goStartTrial)
                if response:
                    correct = 1.
                    score = np.exp(-abs(TargetRT-rt)*10)
                    #vTrial = vTrial * np.exp((rt-TargetRT) * B)
                    vTrial = vTrial + B * ((aTrial/TargetRT) - (aTrial/rt))
                else:
                    correct = 0.
                    score = -1.

            if vTrial > 2.0:
                vTrial = 1.8
            elif vTrial < .25:
                vTrial = .28

            idxResults[t] = np.array([idx, ttype, ssOnset, response, correct, rt, sensitivity, vTrial, aTrial])

        results[results[:, 0]==idx] = idxResults


#
# @jit((float64[:,:], float64[:,:], float64[:,:], float64[:], int64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64), nopython=True)
# def sim_dpm_learning(results, rProb, rProbSS, xtb, idxArray, vProb, vsProb, bound, onset, B, C, R, dx, dt, tb, maxTrials):
#
#     # rProb = randsample((maxTrials, 680))
#     # rProbSS = randsample((maxTrials, 680))
#     onsetIX = np.int(onset)
#     TargetRT = .52
#     goStartTrial = 0.
#     ssbound = 0.
#     # nblocks = rtMatrix.shape[1]
#     # resultsTemp = results
#     # if makeTemp:
#     # resultsTemp = np.zeros_like(results)
#     # resultsTemp[:, np.array([0, 1, 2])] = results[:, np.array([0, 1, 2])]
#
#     # Taim = lambda tr, a, v: tr + (a / v)
#     # zBrake = lambda v, muSSD, tr: v * (muSSD - tr)
#     # SSRT = lambda z, muSSD, ssv: muSSD + (z / abs(ssv))
#
#     for idx in idxArray:
#         idxResults = results[results[:, 0]==idx]
#         ntrials = idxResults.shape[0]
#         vTrial = vProb
#         aTrial = bound
#         errT = 10000
#
#         sigTrial = idxResults[:, 2].std()
#         tau = sigTrial
#         alphaTrial = sigTrial / (sigTrial + tau)
#
#         for t in range(ntrials):
#             ttype, ssOnset = idxResults[t, 1:3]
#             sensitivity = np.exp2(-t*R) #(t/10.)**-R
#             B_t = B * sensitivity
#             C_t = C * sensitivity
#             if ttype==0.:
#                 ssdIX = np.int(ssOnset)
#                 response, rt = sim_dpm_go_stop(rProb[t], xtb, vTrial, aTrial, onsetIX, rProbSS[t], vsProb, ssdIX, dx, dt, goStartTrial, ssbound)
#
#                 if response:
#                     correct = 0.; score = -1.; errT = 0.
#                     #vTrial = vTrial * np.exp((rt-tb) * B_t)
#                     vTrial = vTrial + C_t * ((aTrial/tb) - (aTrial/rt))
#                 else:
#                     correct = 1.; score = 0.; errT += 1.
#
#                 aTrial = bound + C_t * np.exp(-errT)
#                 sigTrial = sigTrial - (alphaTrial * sigTrial)
#                 alphaTrial = sigTrial / (sigTrial + tau)
#
#                 # aTrial =
#             else:
#                 errT += 1
#                 response, rt = sim_dpm_go(rProb[t], xtb, vTrial, aTrial, onsetIX, dx, dt, goStartTrial)
#                 if response:
#                     correct = 1.
#                     score = np.exp(-abs(TargetRT-rt)*10)
#                     # vTrial = vTrial * np.exp((rt-TargetRT) * B)
#
#                     #############################
#                     ### NEW DRIFT UPDATE RULE ###
#                     #############################
#                     vTrial = vTrial + B_t * ((aTrial/TargetRT) - (aTrial/rt))
#                     #vTrial = vTrial + alphaTrial * ((aTrial/rt) - (aTrial/TargetRT))
#                 else:
#                     correct = 0.
#                     score = -1.
#             if vTrial > 1.0:
#                 vTrial = 1.
#             elif vTrial < .05:
#                 vTrial = .05
#             idxResults[t] = np.array([idx, ttype, ssOnset, response, correct, rt, alphaTrial, vTrial, aTrial, sigTrial, score])
#         results[results[:, 0]==idx] = idxResults

    # return results
        # rtVals = idxResults[idxResults[:,3]==1.][:, 5]
        # saccVals = idxResults[idxResults[:,1]==0.][:, 4]
        # ixRT = 0; ixAcc = 0
        # wRT = np.int(np.floor(rtVals.size / nblocks))
        # wAcc = np.int(np.floor(saccVals.size / nblocks))
        # for i in range(nblocks):
        #     rtMatrix[idxCount, i] = np.nanmean(rtVals[ixRT:ixRT+wRT])
        #     saccMatrix[idxCount, i] = np.nanmean(saccVals[ixAcc:ixAcc+wAcc])
        #     ixRT += wRT
        #     ixAcc += wAcc
        # idxCount += 1


# @jit(float64[:,:](float64[:,:], float64[:,:], float64[:,:], float64[:], int64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64, int64), nopython=True)
# def sim_dpm_learning_nruns(nresults, rProb, rProbSS, xtb, idxArray, vProb, vsProb, bound, onset, B, C, R, dx, dt, tb, ntrials, n):
#     nidx = idxArray.size
#     nrowsTotal = nidx * ntrials
#     for i in range(n):
#         start = nrowsTotal*i
#         end = nrowsTotal*(i+1)
#         res = nresults[start:end, :]
#         nresults[start:end, :] = sim_dpm_learning(res, rProb, rProbSS, xtb, idxArray, vProb, vsProb, bound, onset, B, C, R, dx, dt, tb, ntrials)
#     return nresults


@jit((float64[:,:,:], float64[:,:], float64[:,:], float64[:], float64[:], int64[:], float64[:], float64), nopython=True)
def sim_many_single(rProb, rts, xtb, vProb, bound, gOnset, dx, dt):
    ncond, ntrials, ntime = rProb.shape
    tb = ntime * dt
    for i in range(ncond):
        tr = gOnset[i]
        threshold = bound[i]
        timebound = ntime-tr
        for j in range(ntrials):
            ix = tr
            urg_ix = 0
            evidence = 0
            weightedEvidence = 0
            while weightedEvidence<threshold and ix<timebound:
                ix += 1
                urg_ix += 1
                if rProb[i, j, ix] < vProb[i]:
                    evidence += dx[i]
                    continue
                evidence -= dx[i]
                weightedEvidence = evidence * xtb[i, urg_ix]
            rts[i,j] = ix * dt


def sim_multi_params(rProb, dvg, rts, xtb, vProb, bound, gOnset, dx, dt):
    ncond, ntrials, ntime = rProb.shape
    tb = ntime * dt
    for i in range(ncond):
        tr = gOnset[i]
        for j in range(ntrials):
            ix = sim_dpm_trace_upper(rProb[i,j,tr:], dvg[i,j,tr:], xtb[i], vProb[i], bound[i], dx[i])
            if ix<0:
                rt_ix = ntime + 1
                rts[i,j] = 1000.
            else:
                rt_ix = tr + ix
                rts[i,j] = rt_ix * dt
    #
    # gacc = np.mean(ufunc_where(rts < tb, 1, 0), axis=1)
    # rts[rts>=tb] = 1000.
    # cq = self.RTQ(zip(rts, [self.tb] * nl))
    # return hs([hs([i[ii] for i in [gacc, cq]]) for ii in range(nl)])



# @jit((float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:], float64[:], float64[:], int64[:], float64[:], float64), nopython=True)
# def sim_many_single(rProb, dvg, rts, xtb, vProb, bound, gOnset, dx, dt):
    # ix = sim_dpm_trace_upper(rProb[i,j,tr:], dvg[i,j,tr:], xtb[i], vProb[i], bound[i], dx[i])
    # if ix<0:
    #     rt_ix = ntime + 1
    #     rts[i,j] = 1000.
    # else:
    #     rt_ix = tr + ix
    #     rts[i,j] = rt_ix * dt

# @jit(nopython=True)
# def sim_multi_radd(results, xtb, idxArray, vProb, bound, onset, B, dx, dt, tb, maxTrials):
#     TargetRT = .52
#     onsetIX = np.int(onset)
#     rProb = randsample((maxTrials, nAlt, 680))
#     for idx in idxArray:
#         idxResults = results[results[:, 0]==idx]
#         ntrials = idxResults.shape[0]
#         vTrial = vProb
#         for t in range(ntrials):
#             vPrevious = vTrial
#             response, rt = sim_multi_go(rProb[t], xtb, vTrial, aTrial, onsetIX, dx, dt)
#             if response:
#                 correct = 1.
#                 score = np.exp(-abs(TargetRT-rt)*9)
#             else:
#                 correct = 0.
#                 score = -1.
#             vTrial = vTrial + (B * (vTrial - vTrial * np.exp((TargetRT-rt))))
#         if vTrial > 1.0:
#             vTrial = .95
#         elif vTrial < .05:
#             vTrial = .1
#         idxResults[t] = np.array([idx, response, correct, rt, score, vTrial])
#     results[results[:, 0]==idx] = idxResults



# if TargetRT <= .52:
#     TargetRT = .52
# elif TargetRT >= tb:
#     TargetRT = tb


# X = xBase * t**-.1
# idxSSD = np.zeros(idxResults[idxResults[:, 2]<1.].shape[0])
# idxSSD = results[results[:, 2]<1.]
# decay = np.exp(np.arange(ntrials)*-xBase)
# ssExpected = .5
# xBase = .0005
# ssdMS = ssdIX*.001
# delta = ssdMS - ssExpected
# ssExpected = ssExpected + C * delta
# lrDecay = (t/10)**-xBase #np.exp(t*-xBase)
# C = lrDecay * C

# TargetRT = TargetRT + ((C * t**-.1) * TargetRT)
# TargetRT = TargetRT + ((C * t**-.1) * TargetRT)
# TargetRT = TargetRT + ((R * t**-.1) * TargetRT)
# TargetRT = TargetRT + .001
# TargetRT = TargetRT + .001 * decay[t]
# TargetRT = .52 + xBase * np.exp(-errT) #(lrDecay * C) * TargetRT
# TargetRT = .52 + C * np.exp(-errT)

# TargetRT = TargetRT - ((R * t**-.1) * TargetRT)
# TargetRT = TargetRT - .001 #* sensitivity#(sensitivity * .01 * TargetRT)
# TargetRT = TargetRT - .001 * decay[t]#(lrDecay * C) * TargetRT
# TargetRT = TargetRT - ((C * t**-.1) * TargetRT)

# TargetRT = .52 + xBase * np.exp(-errT)


# TargetRT = .52 + C * np.exp(-errT)
# * np.exp((stopTargetRT-rt)))
#
# if nthStop<15:
#     lrBound = .1 * C
# else:
#     # lrBound = lrBound + C * (np.std(idxSSD[nthStop-15 : nthStop]) - lrBound)
#     lrBound = C * np.std(idxSSD[nthStop-15 : nthStop])
# aTrial = bound + lrBound * np.exp(-errT)
# aTrial = bound * np.exp(C * delta)
# ssbound = C * np.exp(-errT)
# nthStop += 1

# TargetRT = TargetRT - .001 * sensitivity
# TargetRT = TargetRT - ((R * t**-.1) * TargetRT)
# TargetRT = TargetRT - ((R * t**-.1) * TargetRT)
# if TargetRT <= .52:
#     TargetRT = .52
# elif TargetRT >= tb:
#     TargetRT = tb








# stopTargetRT = stopTargetRT + .002 * stopTargetRT
# if stopTargetRT > tb:
#     stopTargetRT = tb
# TargetRT = TargetRT + C * TargetRT


# stopTargetRT = stopTargetRT - C * stopTargetRT
# if stopTargetRT < TargetRT:
#     stopTargetRT = TargetRT



# ssdVector = idxResults[:,2].std()
# ssdTrue = ssdVector[ssdVector>0].mean()
# lnlike=1.
# lnlike = get_norm(ssExpected, .05, ssOnset*.001)
# ssExpected = ssExpected + C * (ssOnset*.001 - ssExpected)

# vTrial = vTrial * np.exp(-(TargetRT-rt)* B * Learn)
# vTrial = vTrial * np.exp((tb-rt) * B * Learn *
# vTrial = vTrial + (B * Learn * (vTrial - vTrial * np.exp(TargetRT-rt)))
# vTrial = vTrial + (B * Learn * (vTrial - vTrial * np.exp((TargetRT-rt))))
# vTrial = vTrial + (B * Learn * (vTrial - vTrial * np.exp((ssExpected-ssExPrevious))))
# vTrial  np.exp(-(ssOnset*.001 - ssExpected))
# vTrial = vTrial + (C * Learn * (vTrial - vTrial * np.exp((tb-rt))))
# vTrial = vTrial + (B * Learn * (vTrial - vTrial * np.exp((stopTargetRT-rt))))
# vTrial = vTrial + (B * Learn * (vTrial - vTrial * np.exp((stopTargetRT-rt))))

# vsTrial = vsTrial + (C * Learn * (vsTrial - vsTrial * np.exp((stopTargetRT-rt))))

# aTrial = aTrial + C * startLearn * 1/np.exp((ssExpected-ssExPrevious))
# aTrial = aTrial - (C * Learn * (aTrial - aTrial * np.exp((stopTargetRT-rt))))
# aTrial = aTrial + C * Learn * bound
# aTrial = aTrial - (C * Learn * (aTrial - aTrial * np.exp((stopTargetRT-rt))))
# aTrial = aTrial - (C * Learn * (aTrial - aTrial * np.exp((stopTargetRT-rt))))
# aTrial = aTrial + Learn * C * (ssExpected - ssExPrevious)
# aTrial = aTrial + Learn * C * aTrial #np.exp(ssExPrevious - ssExpected)
# aTrial = bound * (t/10)**-C
# aTrial = aTrial * Learn * np.exp(-(stopTargetRT-rt))
# aTrial = aTrial + C * (score - np.mean(idxResults[:t, 6]))
# aTrial = aTrial + C *
# aTrial = aTrial + C * (aTrial - (aTrial * startLearn * np.exp(-(stopTargetRT-rt))))
# aTrial = aTrial + aTrial * Learn * C * np.exp(stopTargetRT-rt) * np.exp(ssOnset*.001 - ssExpected)

# goStartTrial = goStartTrial - (B * startLearn * np.exp((stopTargetRT-rt)))
# goStartTrial = goStartTrial + C * Learn * np.exp((rt-stopTargetRT))
# goStartTrial = goStartTrial - C * startLearn * 1/np.exp((ssExpected-ssExPrevious))

# if goStartTrial > 0:
#     goStartTrial = 0
# if aTrial < .05:
#     aTrial = .06
# elif aTrial > 1.:
#     aTrial = 1.
