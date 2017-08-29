from __future__ import division
from future.utils import listvalues
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy.random import random_sample as randsample
from scipy.stats.mstats import mquantiles
from itertools import product
from radd import build, theta
from radd.tools.analyze import blockify_trials
from radd.compiled import jitfx
import itertools

class Simulator(object):
    def __init__(self, inits, data=None, fitparams=None, constants=[], ssdMethod='central', nblocks=25, nruns=10):
        self.analyzeProbes = False
        self.ssdMethod = ssdMethod
        self.kind = fitparams.kind
        self.quantiles = fitparams.quantiles
        self.fitparams = fitparams
        self.nruns = nruns
        self.allparams = ['B', 'C', 'R', 'a', 'ssv', 'tr', 'v', 'xb']
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
                self.inits['B'] = .1
                self.inits['C'] = .002
                self.inits['R'] = .1
            self.theta = pd.Series(self.inits)
        if 'nruns' in kw_keys:
            self.nruns = kwargs['nruns']
        if 'constants' in kw_keys:
            const = kwargs['constants']
            self.constants = [p for p in self.allparams if p in const]
        if 'nblocks' in kw_keys:
            self.nblocks = kwargs['nblocks']
            self.blocksCol = 'block{}'.format(self.nblocks)
            if hasattr(self, 'data'):
                self.data = blockify_trials(self.data, nblocks=self.nblocks)
                self.rtBlocks, rtErr, self.saccBlocks, saccErr = self.blockify_data(self.data, measures=['rt', 'acc'], get_var=True)
                self.rt_weights = rtErr.mean() / rtErr
                self.sacc_weights = saccErr.mean() / saccErr
        if 'data' in kw_keys:
            self.get_trials_data(kwargs['data'])
        self.nlevels = self.fitparams['nlevels']
        self.y = self.fitparams.y.flatten()
        self.wts = self.fitparams.wts.flatten()
        self.format_params()
        self.make_results_matrix()
        self.make_io_vectors()


    def cost_fx_rl(self, p):
        resultsDF = self.simulate_model(p, analyze=False)
        return self.analyze_trials(resultsDF)


    def cost_fx(self, p, analyze=True):
        resultsDF = self.simulate_model(p, analyze=False)
        return self.analyze_trials(resultsDF)


    def cost_fx_lmfit(self, lmParams, sse=False):
        # resultsDF = self.simulate_model_nruns(lmParams.valuesdict(), analyze=False)
        resultsDF = self.simulate_model(lmParams.valuesdict(), analyze=False)
        saccBlocks = resultsDF[resultsDF.ttype==0].groupby(self.blocksCol).mean().acc.values
        rtBlocks = resultsDF[resultsDF.response==1].groupby(self.blocksCol).mean().rt.values
        rtErr = self.rt_weights * (rtBlocks*15 - self.rtBlocks*15)
        saccErr = self.sacc_weights * (saccBlocks - self.saccBlocks)
        return np.hstack([rtErr, saccErr])

    # def simulate_model_nruns(self, p, analyze=True):
        #     self.preproc_params(p)
        #     res = jitfx.sim_dpm_learning_nruns(self.nresults, self.rProb, self.rProbSS, self.xtb, self.idxArray, self.vProb, self.vsProb, self.bound, self.gOnset, self.Beta, self.Alpha, self.Rho, self.dx, self.dt, self.tb, self.ntrials, self.nruns)
        #     if analyze:
        #         return self.analyze(res)
        #     self.nresultsDF.loc[:, self.resultsHeader] = res
        #     return self.nresultsDF

    def simulate_model(self, p, analyze=True):
        self.preproc_params(p)
        # results = np.copy(self.results)
        jitfx.sim_dpm_learning(self.results, self.rProb, self.rProbSS, self.xtb, self.idxArray, self.vProb, self.vsProb, self.bound, self.gOnset, self.Beta, self.Alpha, self.Rho, self.dx, self.dt, self.tb, self.ntrials)
        if analyze:
            return self.analyze(results)
        self.resultsDF.loc[:, self.resultsHeader] = self.results
        return self.resultsDF
        # resultsDF = pd.DataFrame(self.results, columns=self.resultsHeader)
        #pd.concat([resultsDF, self.data[['cond', 'sstrial', 'probe', 'trial', self.blocksCol]]], axis=1)
        # return self.resultsDF

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

    def preproc_params(self, p, asarray=False):

        params = deepcopy(self.fixedParams)
        if isinstance(p, np.ndarray):
            p = dict(zip(self.pvary, p))
        params.update(p)

        try:
            self.Beta = params['B']
            self.Alpha = params['C']
            self.Rho = params['R']
        except Exception:
            pass

        self.bound = params['a']
        self.xtb = np.cosh(params['xb'] * self.xtime).squeeze()
        self.vProb = 0.5 * (1 + params['v'] * self.dx / self.si)
        self.vsProb = 0.5 * (1 + params['ssv'] * self.dx / self.si)
        self.gOnset = params['tr'] / self.dt

        if asarray:
            pSeries = pd.Series(params)[self.pvary]
            return pSeries.values

    def make_io_vectors(self):
        self.ntrials = self.data.groupby('idx').count()['trial'].max()
        # generate vectors of random floats [0-1)
        self.rProb = randsample((self.ntrials, self.ntime))
        self.rProbSS = randsample((self.ntrials, self.ntime))

    def make_results_matrix(self):
        dataCols = ['idx', 'ttype', 'ssd', 'response', 'acc', 'rt']
        index = self.data.index.values
        resCols = np.array(dataCols + ['score', 'drift', 'bound'])
        resData = np.zeros((index.size, resCols.size))
        resultsDF = pd.DataFrame(resData, columns=resCols, index=index)
        nidx = self.data.idx.unique().size

        resultsDF.loc[:, ['idx', 'ttype', 'ssd']] = self.data.loc[:, ['idx', 'ttype', 'ssd']]

        self.resultsHeader = resultsDF.columns.tolist()
        self.results = resultsDF.values
        self.resultsDF = pd.concat([resultsDF, self.data[['cond', 'sstrial', 'probe', 'trial', self.blocksCol]]], axis=1)

        self.dataColsIX = [self.resultsHeader.index(col) for col in dataCols + ['score']]
        self.rtMatrix = np.zeros((nidx, self.nblocks))
        self.saccMatrix = np.zeros((nidx, self.nblocks))
        self.nresultsDF = pd.concat([self.resultsDF]*self.nruns)
        self.nresultsDF.reset_index(inplace=True, drop=True)
        self.nresults = np.vstack([self.results]*self.nruns)

    def get_trials_data(self, data):
        if 'probe' in data.columns:
            probeDF = data[data.probe==1]
            self.probes = np.sort(.001*probeDF.ssd.unique() / self.dt)
            self.analyzeProbes = True
        data = data.copy()
        data.loc[:, 'ssd'] = .001*data.ssd.values / self.dt
        data.replace({'ttype': {'go':1., 'stop':0.}}, inplace=True)
        data = data.reset_index(drop=True)
        self.data = blockify_trials(data, nblocks=self.nblocks)
        self.rtBlocks, rtErr, self.saccBlocks, saccErr = self.blockify_data(self.data, measures=['rt', 'acc'], get_var=True)

        self.rt_weights = rtErr.mean() / rtErr
        self.sacc_weights = saccErr.mean() / saccErr
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
        self.nvary = np.ones(len(self.pvary)).astype(np.int64)
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
        if 'bound' in measures:
            scoreTable = pd.pivot_table(data, values='bound', columns=self.blocksCol, index='idx')
            tableList.append(scoreTable)
        if 'score' in measures:
            scoreTable = pd.pivot_table(data, values='score', columns=self.blocksCol, index='idx')
            tableList.append(scoreTable)
        if 'vTrial' in measures:
            vTable = pd.pivot_table(data, values='drift', columns=self.blocksCol, index='idx')
            tableList.append(vTable)
        if get_var:
            blockedMeasures = [[table.mean().values, table.sem().values*1.96] for table in tableList]
            blockedMeasures = list(itertools.chain.from_iterable(blockedMeasures))
        else:
            blockedMeasures = [table.mean().values for table in tableList]
        return blockedMeasures


    def analyze_trials(self, resultsDF):
        saccBlocks = resultsDF[resultsDF.ttype==0].groupby(self.blocksCol).mean().acc.values
        rtBlocks = resultsDF[resultsDF.response==1].groupby(self.blocksCol).mean().rt.values
        rtErr = np.sum((self.rt_weights * (rtBlocks*15 - self.rtBlocks*15))**2)
        saccErr = np.sum((self.sacc_weights * (saccBlocks - self.saccBlocks))**2)
        return rtErr + saccErr
