#!/usr/local/bin/env python
# Authors:
#   Kyle Dunovan (ked64@pitt.edu) &
#   Jeremy Huang (jeremyhuang@cmu.edu)
from __future__ import division
from future.utils import listvalues
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy.random import random_sample as randsample
from scipy.stats.mstats import mquantiles
from itertools import product
from radd import build, theta
from radd.tools.utils import pandaify_results
from numpy import hstack as hs
from radd.compiled.jitfx import *


class Simulator(object):

    def __init__(self, inits, fitparams=None, ssdMethod='all', **kwargs):
        self.ssdMethod = ssdMethod
        self.update(fitparams=fitparams, inits=inits)


    def update(self, force=False, **kwargs):
        kw_keys = list(kwargs)
        if 'fitparams' in kw_keys:
            self.fitparams = kwargs['fitparams']
        if 'inits' in kw_keys:
            self.inits = deepcopy(kwargs['inits'])
            self.theta = pd.Series(self.inits)
        if 'constants' in kw_keys:
            self.constants = deepcopy(kwargs['constants'])
        if 'lmParamsNames' in kw_keys:
            lmParamsNames = kwargs['lmParamsNames']
        else:
            lmParamsNames = None

        self.kind = self.fitparams.kind
        self.clmap = self.fitparams.clmap
        self.depends_on = self.fitparams.depends_on
        self.conds = list(self.clmap)
        self.nconds = len(self.conds)
        self.quantiles = self.fitparams.quantiles
        self.tb = self.fitparams['tb']
        self.dt = self.fitparams['dt']
        self.si = self.fitparams['si']
        self.nlevels = self.fitparams['nlevels']
        self.ntrials = self.fitparams['ntrials']
        self.y = self.fitparams.y.flatten()
        self.wts = self.fitparams.wts.flatten()
        self.ssd, nssd, nss, nss_per, ssd_ix = self.fitparams.ssd_info
        self.format_cond_params(lmParamsNames=lmParamsNames)
        self.make_io_vectors()
        self.__init_analyze_functions__()


    def __init_analyze_functions__(self):
        """ initiates the analysis function used in
        optimization routine to produce the yhat vector
        """
        prob = self.quantiles
        # self.RTQ = lambda zpd: map((lambda x: mquantiles(x[0][x[0] < x[1]], prob)), zpd)
        self.RTQ = lambda zpd: [mquantiles(rt[rt < deadline], prob) for rt, deadline in zpd]


    def cost_fx(self, theta_array):
        yhat = self.simulate_model(theta_array)
        return np.sum((self.wts * (yhat - self.y))**2)


    def cost_fx_lmfit(self, lmParams, sse=False):
        thetaSeries = pd.Series(lmParams.valuesdict())[self.lmParamsNames]
        yhat = self.simulate_model(thetaSeries.values)
        residuals = self.wts * (yhat - self.y)
        if sse:
            return np.sum(residuals**2)
        return residuals


    def simulate_model(self, params, analyze=True):
        xtb, vProb, vsProb, bound, gOnset, ssOnset, dx = self.params_to_array(params, preprocess=True)
        dvg, goRT, ssRT = self.get_io_copies()
        sim_many_dpm(self.rProb, self.rProbSS, dvg, goRT, ssRT, xtb, vProb, vsProb, bound, gOnset, ssOnset, dx, self.dt)
        if analyze:
            return self.analyze(goRT, ssRT)
        return pandaify_results(goRT, ssRT, ssd=self.ssd, bootstrap=False, clmap=self.clmap)


    def analyze(self, rts, ssrts):
        """ get rt and accuracy of go and stop process for simulated
        conditions generated from simulate_dpm
        """
        nl, nssd, nssPer = ssrts.shape
        nss = nssd * nssPer
        erts = rts[:, :nss].reshape(ssrts.shape)
        gacc = np.mean(ufunc_where(rts < self.tb, 1, 0), axis=1)
        sacc = np.mean(ufunc_where(erts <= ssrts, 0, 1), axis=2)
        rts[rts>=self.tb] = 1000.
        ssrts[ssrts>=self.tb] = 1000.
        cq = self.RTQ(zip(rts, [self.tb] * nl))
        eq = self.RTQ(zip(erts, ssrts))
        return hs([hs([i[ii] for i in [gacc, sacc, cq, eq]]) for ii in range(nl)])


    def params_to_array(self, params, preprocess=False):
        if type(params)==dict:
            params = self.pdict_to_array(params)
        pmat = self.pmatrix_vals.copy()
        pvary_vals = slice_theta_array(params, self.pvary_ix, self.nvary)
        pmat.loc[:, self.pvary] = pvary_vals.T
        theta_array = pmat.T.values
        if preprocess:
            return self.preproc_params(theta_array)
        return theta_array

    def pdict_to_array(self, pdict):
        plist = []
        init_ix_arrs = [np.ones(n) for n in self.nvary]
        for i, p in enumerate(self.pvary):
            plist.append(pdict[p] * init_ix_arrs[i])
        parray = np.hstack(plist)
        return parray

    def lmparams_to_array(self, lmparams):
        plist = []
        for i, pname in enumerate(self.pvary):
            p_array = np.array([p.value for p in lmparams if p.vary and pname in p.name])
            plist.append(p_array)
        parray = np.hstack(plist)
        return parray

    def preproc_params(self, theta_array):
        a, si, sso, ssv, tr, v, xb = theta_array
        xtb = np.cosh(xb[:,None] * self.xtime)
        ssd = sso[:, None] + self.ssd
        # dx = np.sqrt(si * self.dt)
        dx = si * np.sqrt(self.dt)
        vProb = 0.5 * (1 + v * dx / si)
        vsProb = 0.5 * (1 + ssv * dx / si)
        gOnset = get_onset_index(tr, self.dt)
        ssOnset = get_onset_index(ssd, self.dt)
        self.vProb = vProb
        self.vsProb = vsProb
        self.gOnset = gOnset
        self.ssOnset = ssOnset
        self.dx = dx
        self. xtb = xtb
        self.bound = a
        self.si = si
        return [xtb] + [vProb, vsProb, a, gOnset, ssOnset, dx]


    def format_cond_params(self, lmParamsNames=None):
        self.ntime = np.int(np.floor(self.tb / self.dt))
        self.xtime = np.cumsum([self.dt] * self.ntime)
        self.allparams = ['a', 'si', 'sso', 'ssv', 'tr', 'v', 'xb']
        pcmap = self.fitparams['pcmap']
        apriori = pd.Series({'z': 0., 'sso': 0., 'xb': 0., 'si': self.si})
        # number of cells in condition matrix (df index)
        self.modelparams = [p for p in self.allparams if p in self.theta.keys()]
        if 'all' in list(self.depends_on) or self.fitparams.nlevels==1:
            self.pvary = np.array(self.modelparams)
            self.nvary = np.ones(len(self.pvary)).astype(np.int64)
            self.lmParamsNames = self.pvary
        else:
            self.pvary = np.array([p for p in self.allparams if p in list(self.depends_on)])
            self.nvary = np.array([len(pcmap[p]) for p in self.pvary])
            if lmParamsNames is None:
                pclist = listvalues(self.fitparams['pcmap'])
                self.lmParamsNames = np.sort(np.hstack(pclist))
        ixx = np.append(0, self.nvary.cumsum())
        self.pvary_start_ix = [(ixx[i], ixx[i+1]) for i in range(self.pvary.size)]
        self.pflat = [p for p in self.modelparams if not p in self.pvary]
        remove_def = list(set(apriori.keys()).intersection(self.pflat))
        self.apriori = apriori.drop(remove_def)
        self.make_params_matrix()
        self.set_pconstant_values_matrix(self.theta)

    def make_io_vectors(self):
        self.ssd, nssd, nss, nss_per, ssd_ix = self.fitparams.ssd_info
        self.rProb = randsample((self.nlevels, self.ntrials, self.ntime))
        self.rProbSS = randsample((self.nlevels, nssd, nss_per, self.ntime))
        self.rProbSS3d = randsample((self.nlevels, nssd * nss_per, self.ntime))
        dvg = np.zeros_like(self.rProb)
        self.goRT = np.zeros((self.nlevels, self.ntrials))
        self.ssRT = np.zeros((self.nlevels, nssd, nss_per))
        # self.ssRT2d = np.zeros((self.nlevels, nssd * nss_per))
        self.vectors = [dvg, self.goRT, self.ssRT]
        ssdSteps = get_onset_index(self.ssd, self.dt)
        self.ssdTrials = np.sort(np.tile(ssdSteps, nss_per))

    def get_io_copies(self):
        return [v.copy() for v in self.vectors]

    def complete_allparams_pdict(self, pdict):
        theta = pd.Series(pdict)
        for p in self.allparams:
            if p not in self.modelparams:
                theta[p] = self.apriori[p]
        # order params like allparams
        theta = theta[self.allparams]
        return theta

    def set_pconstant_values_matrix(self, pdict):
        theta = self.complete_allparams_pdict(pdict)
        pmatrix_ix = self.pmatrix.copy()
        pmatrix_vals = self.pmatrix.copy()
        for p in self.allparams:
            if p in self.pvary:
                continue
            pmatrix_vals.loc[:, p] = pmatrix_ix.loc[:, p] + theta[p]
        self.pmatrix_vals = pmatrix_vals

    def pvary_broadcast_arrays(self, pmatrix):
        index = pmatrix.index.values
        c_levels = [lvl for lvl in listvalues(self.clmap)]
        level_data = list(product(*c_levels))
        condsdf = pd.DataFrame(level_data, columns=self.conds, index=index)

        # concat (along axis 1) conditional level names and depends_on param columns
        pmatrix = pd.concat([condsdf, pmatrix], axis=1)
        for pvary in self.pvary:
            pv_cond = self.depends_on[pvary]
            if isinstance(pv_cond, list):
                p_levels = self.fitparams['pcmap'][pvary]
                pvary_ix = np.arange(len(p_levels))
                pmatrix[pvary] = pvary_ix
            else:
                levels = self.clmap[pv_cond]
                for i, level_name in enumerate(levels):
                    pmatrix.ix[pmatrix[pv_cond]==level_name, pvary]=i
        pmatrix = pmatrix.iloc[:, self.nconds:].apply(pd.to_numeric)
        return pmatrix

    def make_params_matrix(self):
        cols = np.array(self.allparams)
        index = np.arange(self.nlevels)
        zerodata = np.zeros((index.size, cols.size)).astype(np.int)
        pmatrix = pd.DataFrame(zerodata, columns=cols, index=index)
        if self.nlevels>1:
            pmatrix = self.pvary_broadcast_arrays(pmatrix)
        self.pvary_ix = pmatrix[self.pvary].T.values
        self.n_vals = np.array([pmatrix[p].unique().size for p in cols])
        self.index_arrays = pmatrix.T.values
        self.pmatrix = pmatrix
