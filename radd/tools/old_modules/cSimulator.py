from __future__ import division
from future.utils import listvalues
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy.random import random_sample as randsample
from scipy.stats.mstats import mquantiles
from itertools import product
from radd import build, theta
from radd.tools.analyze import blockify_trials, pandaify_results
from radd.compiled import jitfx
import itertools

class Simulator(object):
    def __init__(self, inits, data=None, fitparams=None, constants=[], ssdMethod='central', nblocks=25):
        self.analyzeProbes = False
        self.ssdMethod = ssdMethod
        self.kind = fitparams.kind
        self.tb = fitparams.tb
        self.quantiles = fitparams.quantiles
        self.fitparams = fitparams
        self.allparams = ['a', 'tr', 'v', 'xb', 'ssv', 'B', 'C']
        self.update(fitparams=self.fitparams, inits=inits, data=data, constants=constants, nblocks=nblocks)

    def update(self, **kwargs):
        kw_keys = list(kwargs)
        self.tb = self.fitparams['tb']
        self.dt = self.fitparams['dt']
        self.si = self.fitparams['si']
        if 'fitparams' in kw_keys:
            self.fitparams = kwargs['fitparams']
        if 'inits' in kw_keys:
            self.inits = deepcopy(kwargs['inits'])
            if 'B' not in list(self.inits):
                self.inits['B'] = .02
                self.inits['C'] = .02
            self.theta = pd.Series(self.inits)
        if 'constants' in kw_keys:
            const = kwargs['constants']
            self.constants = [p for p in self.allparams if p in const]
        if 'nblocks' in kw_keys:
            self.nblocks = kwargs['nblocks']
            self.blocksCol = 'block{}'.format(self.nblocks)
            if hasattr(self, 'data'):
                self.data = blockify_trials(self.data, nblocks=self.nblocks)
                self.rtBlocks, self.saccBlocks, self.scoreBlocks = self.blockify_data(self.data, measures=['rt', 'acc', 'score'])
        if 'data' in kw_keys:
            self.get_trials_data(kwargs['data'])
        self.ssd = self.fitparams.ssd_info[-1]
        self.nlevels = self.fitparams['nlevels']
        self.y = self.fitparams.y.flatten()
        self.wts = self.fitparams.wts.flatten()
        self.format_params()
        self.make_results_matrix()
        self.make_io_vectors()

    def cost_fx_rl(self, p):
        resultsDF = self.simulate_model(p, analyze=False)
        return self.analyze_trials(resultsDF)

    def cost_fx(self, p):
        yhat = self.simulate_model(p)
        return np.sum((self.wts * (yhat - self.y))**2)

    def cost_fx_lmfit(self, lmParams, sse=False):
        yhat = self.simulate_model(lmParams.valuesdict())
        residuals = self.wts * (yhat - self.y)
        if sse:
            return np.sum(residuals**2)
        return residuals

    def simulate_learning_model(self, p, analyze=True):
        results = np.copy(self.resultsDF.values)
        self.preproc_params(p)
        jitfx.sim_dpm_learning(results, self.xtb, self.idxArray, self.vProb, self.vsProb, self.bound, self.gOnset, self.Beta, self.Alpha, self.dx, self.dt, self.tb, self.ntrials)
        if analyze:
            return self.analyze(results)
        resultsDF = pd.DataFrame(results, columns=self.resultsHeader)
        return pd.concat([resultsDF, self.data[['cond', 'sstrial', 'probe', 'trial', self.blocksCol]]], axis=1)

    def simulate_model(self, params, analyze=True):
        xtb, vProb, vsProb, bound, gOnset, ssOnset, dx = self.params_to_array(params, preprocess=True)
        dvg, goRT, ssRT = self.get_io_copies()
        jitfx.sim_many_dpm(self.rProb, self.rProbSS, dvg, goRT, ssRT, xtb, vProb, vsProb, bound, gOnset, ssOnset, dx, self.dt)
        if analyze:
            return self.analyze_static(goRT, ssRT)
        return pandaify_results(goRT, ssRT, ssd=self.ssd)

    def analyze(self, res):
        idxIX, ttypeIX, ssdIX, respIX, accIX, rtIX, scoreIX = self.dataColsIX
        # calc go and stop accuracy
        goTrials = res[res[:, ttypeIX]==1.]
        ssTrials = res[res[:, ttypeIX]==0.]
        gAcc = goTrials[:, accIX].mean()
        if self.analyzeProbes:
            if self.ssdMethod=='all':
                ssdAccuracy = [res[res[:, ssdIX]==ssd][:, accIX] for ssd in self.probes]
                sAcc = np.array([np.nanmean(ssdAcc) for ssdAcc in ssdAccuracy])
            else:
                sAcc = np.nanmean(np.hstack(ssdAccuracy))
        else:
            sAcc = ssTrials[:, accIX].mean()
        # calc correct and error rt quantiles
        respTrials = res[res[:, respIX]==1.]
        corRT = respTrials[respTrials[:, accIX]==1.][:, rtIX]
        errRT = respTrials[respTrials[:, accIX]==0.][:, rtIX]
        cq = mquantiles(corRT[corRT < self.tb], self.quantiles)
        eq = mquantiles(errRT[errRT < self.tb], self.quantiles)
        # concatenate into single cost vector
        return np.hstack([gAcc, sAcc, cq, eq])

    def analyze_static(self, rts, ssrts):
        """ get rt and accuracy of go and stop process for simulated
        conditions generated from simulate_dpm
        """
        nl, nssd, nssPer = ssrts.shape
        nss = nssd * nssPer
        erts = rts[:, :nss].reshape(ssrts.shape)
        gacc = np.mean(jitfx.ufunc_where(rts < self.tb, 1, 0), axis=1)
        sacc = np.mean(jitfx.ufunc_where(erts <= ssrts, 0, 1), axis=2)
        rts[rts>=self.tb] = np.nan
        ssrts[ssrts>=self.tb] = np.nan
        cq = self.RTQ(zip(rts, [self.tb] * nl))
        eq = self.RTQ(zip(erts, ssrts))
        return hs([hs([i[ii] for i in [gacc, sacc, cq, eq]]) for ii in range(nl)])

    def preproc_params(self, p, asarray=False):
        params = deepcopy(self.fixedParams)
        if isinstance(p, np.ndarray):
            p = dict(zip(self.pvary, p))
        params.update(p)
        self.bound = params['a']
        self.xtb = np.cosh(params['xb'] * self.xtime).squeeze()
        self.vProb = 0.5 * (1 + params['v'] * self.dx / self.si)
        self.vsProb = 0.5 * (1 + params['ssv'] * self.dx / self.si)
        self.gOnset = params['tr'] / self.dt
        try:
            self.Beta = params['B']
            self.Alpha = params['C']
        except Exception:
            pass
        if asarray:
            pSeries = pd.Series(params)[self.pvary]
            return pSeries.values

    def preproc_params_static(self, theta_array):
        a, tr, v, xb, ssv, sso, si = theta_array
        xtb = np.cosh(xb[:,None] * self.xtime)
        ssd = sso[:, None] + self.ssd
        dx = np.sqrt(si * self.dt)
        vProb = 0.5 * (1 + v * dx / si)
        vsProb = 0.5 * (1 + ssv * dx / si)
        gOnset = jitfx.get_onset_index(tr, self.dt)
        ssOnset = jitfx.get_onset_index(ssd, self.dt)
        return [xtb] + [vProb, vsProb, a, gOnset, ssOnset, dx]


    def make_io_vectors(self):
        self.ntrials = self.data.groupby('idx').count()['trial'].max()
        # generate vectors of random floats [0-1)
        self.rProb = randsample((self.ntrials, self.ntime))
        self.rProbSS = randsample((self.ntrials, self.ntime))

    def make_results_matrix(self):
        dataCols = ['idx', 'ttype', 'ssd', 'response', 'acc', 'rt']
        index = self.data.index.values
        resCols = np.array(dataCols + ['score', 'vTrial', 'ssTarget'])
        resData = np.zeros((index.size, resCols.size))
        resultsDF = pd.DataFrame(resData, columns=resCols, index=index)
        self.resultsDF = resultsDF.copy()
        self.resultsDF.loc[:, ['idx', 'ttype', 'ssd']] = self.data.loc[:, ['idx', 'ttype', 'ssd']]
        self.resultsHeader = self.resultsDF.columns.tolist()
        self.dataColsIX = [self.resultsHeader.index(col) for col in dataCols + ['score']]

    def get_trials_data(self, data):
        if 'probe' in data.columns:
            probeDF = data[data.probe==1]
            self.probes = np.sort(.001*probeDF.ssd.unique() / self.dt)
            self.analyzeProbes = True
        data = data.copy()
        data.loc[:, 'ssd'] = .001*data.ssd.values / self.dt
        data = data.replace({'ttype': {'go':1., 'stop':0.}})
        data = data.reset_index(drop=True)
        self.data = blockify_trials(data, nblocks=self.nblocks)
        self.rtBlocks, self.saccBlocks, self.scoreBlocks = self.blockify_data(self.data, measures=['rt', 'acc', 'score'])
        self.idxArray = self.data.idx.unique()

    def format_params(self):
        self.dx = np.sqrt(self.si * self.dt)
        self.ntime = int(self.tb / self.dt)
        self.xtime = np.cumsum([self.dt] * self.ntime)
        self.modelparams = [p for p in self.allparams if p in list(self.inits)]
        self.pflat = [p for p in self.allparams if p in self.constants]
        self.pvary  = [p for p in self.allparams if p not in self.pflat]
        self.fixedParams = self.theta[self.pflat].to_dict()
        if 'xb' not in self.modelparams:
            self.fixedParams['xb'] = 0.
        # number of cells in condition matrix (df index)
        self.nvary = np.ones(len(self.pvary))
        self.preproc_params(self.inits)

    def make_empircal_data_matrix(self):
        ttypeIX, ssdIX, respIX, accIX, rtIX = self.dataColsIX
        res = self.data
        goTrials = res[res[:, ttypeIX]==1.]
        ssTrials = res[res[:, ttypeIX]==0.]
        self.GoAccEmp = goTrials[:, accIX]
        self.ssAccEmp = ssTrials[:, accIX]
        self.CorRTEmp = goTrials[goTrials[:, accIX]==1.][:, rtIX]
        self.ErrRTEmp = goTrials[goTrials[:, accIX]==0.][:, rtIX]
        respTrials = res[res[:, respIX]==1.]
        self.CorRTEmp = respTrials[respTrials[:, accIX]==1.][:, rtIX]
        self.ErrRTEmp = respTrials[respTrials[:, accIX]==0.][:, rtIX]

    def blockify_data(self, data, get_var=False, measures=['rt', 'acc']):
        data = data.copy()
        if self.blocksCol not in data.columns:
            data = blockify_trials(data, nblocks=self.nblocks)
        goDF = data[data.response==1.]
        ssDF = data[data.ttype==0.]
        tableList = []
        if 'rt' in measures:
            rtTable = pd.pivot_table(goDF, values='rt', columns=self.blocksCol, index='idx')
            tableList.append(rtTable)
        if 'acc' in measures:
            ssTable = pd.pivot_table(ssDF, values='acc', columns=self.blocksCol, index='idx')
            tableList.append(ssTable)
        if 'score' in measures:
            scoreTable = pd.pivot_table(data, values='score', columns=self.blocksCol, index='idx')
            tableList.append(scoreTable)
        if 'vTrial' in measures:
            vTable = pd.pivot_table(data, values='vTrial', columns=self.blocksCol, index='idx')
            tableList.append(vTable)
        if get_var:
            blockedMeasures = [[table.mean().values, table.sem().values*1.96] for table in tableList]
            blockedMeasures = list(itertools.chain.from_iterable(blockedMeasures))
        else:
            blockedMeasures = [table.mean().values for table in tableList]
        return blockedMeasures

    def analyze_trials(self, resultsDF):
        goDF = resultsDF[resultsDF.response==1.]
        ssDF = resultsDF[resultsDF.ttype==0.]
        rtBlocks = pd.pivot_table(goDF, values='rt', columns=self.blocksCol, index='idx').mean().values
        saccBlocks = pd.pivot_table(ssDF, values='acc', columns=self.blocksCol, index='idx').mean().values
        rtErr = np.sum((rtBlocks*10 - self.rtBlocks*10)**2)
        saccErr = np.sum((saccBlocks - self.saccBlocks)**2)
        return rtErr + saccErr
