from __future__ import division
from future.utils import listvalues
import pandas as pd
import numpy as np
from numpy.random import random_sample as randsample
from scipy.stats.mstats import mquantiles
from itertools import product
from radd import build, theta
from radd.numbaradd.jitfx import *
from radd.numbaradd.helperfx import *


class Simulator(object):

    def __init__(self, inits, fitparams=None, learn=False, dt=.005, si=.01):
        self.fitparams = fitparams
        self.y = fitparams.y.flatten()
        self.wts = fitparams.wts.flatten()
        self.theta = pd.Series(inits)
        self.learn=learn
        self.clmap = fitparams.clmap
        self.depends_on = fitparams.depends_on
        self.conds = list(self.clmap)
        self.nconds = len(self.conds)
        self.nlevels = self.fitparams.nlevels
        self.kind = self.fitparams.kind
        self.tb = self.fitparams.tb
        self.ntrials = self.fitparams.ntrials
        self.quantiles = self.fitparams.quantiles
        self.ssd, nssd, nss, nss_per, ssd_ix = self.fitparams.ssd_info

        self.format_cond_params(dt=dt, si=si)
        # generate vectors of random floats [0-1)
        self.rvector = randsample((self.nlevels, self.ntrials, self.ntime))
        self.rvector_ss = randsample((self.nlevels, nssd, nss_per, self.ntime))
        self.DVg = np.zeros_like(self.rvector)
        self.rts = np.zeros((self.nlevels, self.ntrials))
        self.ssrts = np.zeros((self.nlevels, nssd, nss_per))

    def cost_fx(self, theta_array):
        yhat = self.simulate_model(theta_array)
        return np.sum((self.wts * (yhat - self.y))**2)

    def slice_theta_arr_wrapper(self, theta_array):
        out = slice_theta_array(theta_array, self.index_arrays, self.n_vals)
        out = [np.asarray(out[i]) for i in range(len(out))]
        return out

    def simulate_model(self, theta_array, analyze=True):
        a, tr, v, xb, ssv, sso, si = self.slice_theta_arr_wrapper(theta_array)
        xtb = np.cosh(xb[:,None] * self.xtime)
        ssd = sso[:, None] + self.ssd
        dx = np.sqrt(si * self.dt)
        vg_prob = 0.5 * (1 + v * dx / si)
        vs_prob = 0.5 * (1 + ssv * dx / si)
        g_on = get_onset_index(tr, self.dt).astype(np.int64)
        ss_on = get_onset_index(np.hstack(ssd), self.dt).reshape(ssd.shape)
        ss_on = ss_on.astype(np.int64)
        go_rts, stop_rts = sim_many_dpm(self.rvector, self.rvector_ss, self.DVg, xtb, vg_prob, vs_prob, a, g_on, ss_on, dx, self.dt, self.rts, self.ssrts)
        if analyze:
            return self.analyze(go_rts, stop_rts)
        return go_rts, stop_rts

    def analyze(self, rts, ssrts):
        nl = self.nlevels
        nss = self.fitparams['ssd_info'][2]
        def correct_quant(rts):
            return mquantiles(rts[rts < self.tb], self.quantiles)
        def error_quant(rt_list):
            erts, ssrts = rt_list
            return mquantiles(erts[erts < ssrts], self.quantiles)
        erts = rts[:, :nss].reshape(ssrts.shape)
        cq = map(correct_quant, rts)
        eq = map(error_quant, zip(erts, ssrts))
        gacc = np.mean(ufunc_where(rts < self.tb, 1, 0), axis=1)
        sacc = np.mean(ufunc_where(erts <= ssrts, 0, 1), axis=2)
        yhat = np.hstack([np.hstack([i[ii] for i in [gacc, sacc, cq, eq]]) for ii in xrange(nl)])
        return yhat

    def pdict_to_array(self, pdict):
        plist = []
        init_ix_arrs = [np.ones(n) for n in self.n_vals]
        for i, p in enumerate(self.allparams):
            if p in self.modelparams:
                plist.append(pdict[p] * init_ix_arrs[i])
                continue
            plist.append(self.apriori[p] * init_ix_arrs[i])
        parray = np.hstack(plist)
        return parray

    def format_cond_params(self, si=.01, dt=.005):
        self.dt = dt
        self.ntime = int(self.tb / self.dt)
        self.xtime = np.cumsum([self.dt] * self.ntime)
        self.allparams = ['a', 'tr', 'v', 'xb', 'ssv', 'sso', 'si']
        apriori = pd.Series({'z': 0., 'sso': 0., 'xb': 0., 'si': si})
        # number of cells in condition matrix (df index)
        self.modelparams = [p for p in self.allparams if p in self.theta.keys()]
        self.pvary = np.array([p for p in self.allparams if p in list(self.depends_on)])
        self.nvary = [self.clmap[self.depends_on[p]].size for p in self.pvary]
        self.pflat = [p for p in self.modelparams if not p in self.pvary]
        remove_def = list(set(apriori.keys()).intersection(self.pflat))
        self.apriori = apriori.drop(remove_def)
        self.make_params_matrix()

    def init_theta_array(self, inits):
        pseries = []
        n = self.n_vals
        cast_value = lambda val, ix: val * np.ones(n[ix])
        for i, p in enumerate(self.allparams):
            if p in list(self.apriori):
                pseries.append(cast_value(self.apriori[p] * i))
                continue
            pseries.append(cast_value(inits[p], i))
        theta_array = np.hstack(np.hstack([pseries]))
        return theta_array

    def pvary_broadcast_arrays(self, pmatrix):
        index = pmatrix.index.values
        c_levels = [lvl for lvl in listvalues(self.clmap)]
        level_data = list(product(*c_levels))
        condsdf = pd.DataFrame(level_data, columns=self.conds, index=index)
        # concat (along axis 1) conditional level names and depends_on param columns
        pmatrix = pd.concat([condsdf, pmatrix], axis=1)
        for pvary in self.pvary:
            pv_cond = self.depends_on[pvary]
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
