#!usr/bin/env python
from __future__ import division
import os
from future.utils import listvalues
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy import array
from radd.tools import analyze
from radd import theta
from itertools import product


class DataHandler(object):

    def __init__(self, model, max_wt=3., verbose=False):
        self.model = model
        self.data = model.data
        self.inits = model.inits
        self.model_id = model.model_id
        self.idx = model.idx
        self.nidx = model.nidx
        self.weighted = model.weighted
        self.max_wt = max_wt
        self.ssd_method = model.ssd_method
        self.kind = model.kind
        self.fit_on = model.fit_on
        self.quantiles = model.quantiles
        self.groups = model.groups
        self.depends_on = model.depends_on
        self.pcmap = model.pcmap
        self.clmap = model.clmap
        self.conds = model.conds
        self.nconds = model.nconds
        self.nlevels = model.nlevels
        self.cond_matrix = model.cond_matrix
        self.bwfactors = model.bwfactors
        self.nrows = self.nidx * model.nlevels
        self.grpData = self.data.groupby(self.groups)
        self.verbose = verbose


    def make_dataframes(self):
        """ Generates the following dataframes and arrays:
        observed (list of ndarrays):
            list containing ndarrays (ncond x n_observed_data_points) entered into thec
            costfx during fitting. If self.fit_on=='average', self.observed will
            contain a single ndarray (ncond x average_data). If self.fit_on=='subjects',
            self.observed will contain 1 ndarray for each subject and fits will be performed
            iteratively
        observedDF (DF):
              Contains Prob and RT quant. for each subject
              used to calc. cost fx weights
        fits (DF):
              empty DF shaped like observed DF, used to store simulated
              predictions of the optimized model
        fitinfo (DF):
              stores all opt. parameter values and model fit statistics
        """
        # make observed y and wts dataframes (all subjects)
        self.make_observed_groupDFs()
        # make yhatdf to fill w/ model-predicted data arrays
        self.yhatdf = self.make_yhat_df()
        # make fitdf for storing w/ goodness-of-fit stats and popt
        self.fitdf = self.make_fit_df()
        # make poptdf for storing popt of conditional models as matrix
        self.poptdf = self.make_popt_df()
        odf = self.observedDF.copy()
        wdf = self.wtsDF.copy()
        condvalues = lambda df: df.loc[:, 'acc':].dropna(axis=1).values.squeeze()
        flatvalues = lambda df: df.loc['acc':].values.squeeze()
        if self.fit_on=='subjects':
            self.observed = [condvalues(odf[odf['idx']==idx]) for idx in self.idx]
            self.cond_wts = [condvalues(wdf[wdf['idx']==idx]) for idx in self.idx]
            self.observed_flat = [flatvalues(odf[odf['idx']==idx].mean()) for idx in self.idx]
            self.flat_wts = [flatvalues(wdf[wdf['idx']==idx].mean()) for idx in self.idx]
        elif self.fit_on=='average':
            self.observed = [condvalues(odf.groupby(self.conds).mean())]
            self.cond_wts = [condvalues(wdf.groupby(self.conds).mean())]
            self.observed_flat = [flatvalues(odf.mean())]
            self.flat_wts = [flatvalues(wdf.mean())]


    def make_observed_groupDFs(self):
        """ concatenate all idx data vectors into a dataframe
        """
        odf_header = self.make_headers()
        data = self.data.copy()
        ssdmethod = self.ssd_method
        self.grpData = data.groupby(np.hstack(['idx', self.conds]).tolist())
        datdf = self.grpData.apply(analyze.rangl_data, ssdmethod, self.quantiles).sortlevel(0)
        # self.datdf = datdf
        groupvalues = datdf.reset_index()[self.groups].values
        nan_data = np.zeros((groupvalues.shape[0], len(odf_header)), dtype=np.int64)
        self.observedDF = pd.DataFrame(nan_data, columns=odf_header, index=range(nan_data.shape[0]))
        self.observedDF.loc[:, self.groups] = groupvalues
        self.wtsDF = self.make_wts_df()
        for rowi in self.observedDF.index.values:
            # fill observedDF one row at a time, using idx_rows
            self.observedDF.loc[rowi, self.idx_cols[rowi]] = datdf.values[rowi]

        if self.bwfactors is not None and self.model.fit_on=='subjects':
            bwix = self.observedDF[self.groups].columns.size
            bwunique = [self.data[self.data.idx==idx][self.bwfactors].unique() for idx in self.idx]
            #wfactors = list(self.clmap)
            #wfactors.remove(self.bwfactors)
            # nwithin = np.sum([len(self.clmap[wfactor]) for wfactor in wfactors])
            bwcast = np.hstack([np.tile(bw, self.nlevels) for bw in bwunique])

            print(len(bwcast))
            print(self.observedDF.shape[0])
            print(self.bwfactors)
            #print(self.observedDF)
            self.observedDF.insert(bwix, self.bwfactors, bwcast)
            self.wtsDF.insert(bwix, self.bwfactors, bwcast)
            errdf = self.observedDF.groupby(self.conds+[self.bwfactors]).sem()*2.
            self.observedErr = errdf.reset_index()[self.observedDF.columns[1:]]
        else:
            errdf = self.observedDF.groupby(self.conds).sem()*2
            self.observedErr = errdf.reset_index()[self.observedDF.columns[1:]]


    def make_freq_df(self):
        data = self.data.copy()
        self.grpData = data.groupby(self.groups)
        freqdf = self.grpData.apply(analyze.rangl_freq, self.quantiles)
        countdf =  self.grpData.apply(analyze.rangl_counts)

        metadf = freqdf.reset_index()[self.groups]
        bins = np.arange(self.quantiles.size+1)
        freqcols = sum([['{}{}'.format(rtype, i) for i in bins+1] for rtype in ['o', 'e']], [])
        freqvals = pd.DataFrame(np.vstack(freqdf.values), columns=freqcols)
        countvals = pd.DataFrame(np.vstack(countdf.values), columns=['Ncor', 'Nerr'])
        count_freq_vals = pd.concat([countvals, freqvals], axis=1)
        freqDF = pd.concat([metadf, count_freq_vals], axis=1)
        return freqDF


    def make_wts_df(self):
        """ calculate and store cost_function weights
        for all subjects/conditions in data
        """
        wtsDF = self.observedDF.copy()
        if self.weighted:
            try:
                # calc & fill wtsDF with idx quantile and accuracy weights (ratios)
                quant_wts, acc_wts = self.calc_empirical_weights()
                qwts = np.vstack(quant_wts).reshape(wtsDF.shape[0], -1)
                awts = np.vstack(acc_wts).reshape(wtsDF.shape[0], -1)
                wtsDF.loc[:, self.q_cols] = qwts
                wtsDF.loc[:, self.p_cols] = awts
                wts_numeric = wtsDF.loc[:, 'acc':]
                wtsDF.loc[:, 'acc':] = wts_numeric.apply(analyze.fill_nan_vals, axis=1)
            except Exception:
                if self.verbose:
                    print("Unable to calculate cost f(x) weights, setting all w=1.")
                wtsDF.loc[:, self.p_cols+self.q_cols] = 1.
        else:
            wtsDF.loc[:, self.p_cols+self.q_cols] = 1.
        return wtsDF.copy()


    def calc_empirical_weights(self):
        """ calculates weight vectors for observed correct & err RT quantiles and
        go and stop accuracy for each subject (see funcs in radd.tools.analyze)
        """
        data = self.data.copy()
        # quant_wts = [analyze.idx_quant_weights(df, conds=self.conds, max_wt=self.max_wt, quantiles=self.quantiles, bwfactors=self.bwfactors) for i, df in data.groupby('idx')]
        quant_wts = analyze.idx_quant_weights_OLD(data, prob=self.quantiles, groups=self.groups, nsplits=np.cumprod(self.cond_matrix)[-1], max_wt=self.max_wt)
        acc_wts = [analyze.idx_acc_weights(df, conds=self.conds, ssd_method=self.ssd_method) for i, df in data.groupby('idx')]
        return quant_wts, acc_wts


    def get_cond_combos(self):
        clevels = [list(self.clmap[c]) for c in np.sort(list(self.clmap))]
        level_data = list(product(*clevels))
        return pd.DataFrame(level_data, columns=self.groups[1:])


    def make_yhat_df(self):
        """ make empty dataframe for storing model predictions (yhat)
        """
        yhatcols = self.observedDF.columns
        indx = np.arange(self.nlevels)
        yhatdf = pd.DataFrame(np.nan, index=indx, columns=yhatcols)
        minfo = self.get_cond_combos()
        yhatdf.loc[:, minfo.columns] = minfo
        yhatdf = yhatdf.copy()
        yhatdf.insert(len(self.groups), 'pvary', np.nan)
        self.empty_yhatdf = yhatdf.copy()
        return yhatdf


    def make_popt_df(self):
        """ make empty dataframe for storing popt after each fit
        """
        indx = np.arange(self.nlevels)
        poptcols = self.groups + self.poptdf_cols
        poptdf = pd.DataFrame(np.nan, index=indx, columns=poptcols)
        minfo = self.get_cond_combos()
        poptdf[minfo.columns] = minfo
        poptdf.insert(len(self.groups), 'pvary', np.nan)
        self.empty_poptdf = poptdf.copy()
        return poptdf


    def make_fit_df(self):
        """ make empty dataframe for storing fit info
        """
        fitdf = pd.DataFrame(np.nan, index=[0], columns=self.f_cols)
        self.empty_fitdf = fitdf.copy()
        return fitdf


    def fill_poptdf(self, popt, fitparams=None):
        """ fill fitdf with fit statistics
        ::Arguments::
            popt (dict):
                fitinfo Series containing model statistics and
                optimized parameters (see Model.assess_fit() method)
            fitparams (Series):
                model.fitparams dict w/ meta info for last fit
        """

        if fitparams is None:
            fitparams = self.model.fitparams

        poptdf = self.empty_poptdf.copy()
        poptdf['idx'] = str(fitparams.idx)
        poptdf['pvary'] = '_'.join(list(self.model.depends_on))

        p = pd.Series(deepcopy(popt))[self.poptdf_cols].to_dict()
        poptdf.loc[:, self.poptdf_cols] = pd.DataFrame(p, index=poptdf.index)
        # popt = self.model.simulator.vectorize_params(popt)
        # popt_vals = np.array([popt[pkey] for pkey in self.poptdf_cols]).T
        # poptdf.loc[:, self.poptdf_cols] = popt_vals
        if np.any(self.poptdf.isnull()):
            poptdf = poptdf
        else:
            poptdf = pd.concat([self.poptdf, poptdf], axis=0)

        self.poptdf = poptdf.reset_index(drop=True)
        return self.poptdf


    def fill_fitdf(self, finfo, fitparams=None):
        """ fill fitdf with fit statistics
        ::Arguments::
            finfo (Series):
                fitinfo Series containing model statistics and
                optimized parameters (see Model.assess_fit() method)
            fitparams (Series):
                model.fitparams dict w/ meta info for last fit
        """
        if fitparams is None:
            fitparams = self.model.fitparams
        fitdf = self.empty_fitdf.copy()
        pvary = list(self.model.depends_on)

        for fcol in finfo.keys():
            if fcol in pvary:
                pkey = '_'.join([fcol, 'avg'])
                fitdf.loc[0, pkey] = np.mean(finfo[fcol])
                continue
            fitdf.loc[0, fcol] = finfo[fcol]
        if np.any(self.fitdf.isnull()):
            fitdf = fitdf
        else:
            fitdf = pd.concat([self.fitdf, fitdf], axis=0)
        self.fitdf = fitdf.reset_index(drop=True)
        return self.fitdf


    def fill_yhatdf(self, yhat, fitparams=None):
        """ fill yhatdf with model predictions
        ::Arguments::
            yhat (ndarray):
                array containing model predictions (nlevels x ncols)
                where ncols is number of data columns in observedDF
            fitparams (Series):
                model.fitparams dict w/ meta info for last fit
        """
        if fitparams is None:
            fitparams = self.model.fitparams
        nl = fitparams['nlevels']
        yhat = yhat.reshape(nl, -1)
        yhatdf = self.empty_yhatdf.copy()
        yhatdf.loc[:, 'acc':] = yhat
        yhatdf['idx'] = str(fitparams.idx)
        yhatdf['pvary'] = '_'.join(list(self.model.depends_on))
        if np.any(self.yhatdf.isnull()):
            yhatdf = yhatdf
        else:
            yhatdf = pd.concat([self.yhatdf, yhatdf], axis=0)
        self.yhatdf = yhatdf.reset_index(drop=True)
        return self.yhatdf


    def set_model_ssds(self, stopdf, index=['idx']):
        """ set model attr "ssd" as list of np.arrays
        ssds to use when simulating data during optimization
        """
        if self.ssd_method is None:
            self.ssd_method = analyze.determine_ssd_method(stopdf)
            self.model.ssd_method = self.ssd_method

        bwfactors = self.bwfactors
        if bwfactors is not None:
            stop_dfs = stopdf.groupby(bwfactors)
        else:
            stop_dfs = [[None, stopdf]]

        ssdList = []
        for lvl, df in stop_dfs:
            sdf = analyze.get_model_ssds(df, conds=self.conds, ssd_method=self.ssd_method, bwfactors=bwfactors)
            if lvl is not None:
                sdf[bwfactors] = lvl
            ssdList.append(sdf)

        self.ssdDF = pd.concat(ssdList, ignore_index=True)


    def make_headers(self, ssd_list=None):
        g_cols = self.groups
        if 'ssd' in self.data.columns:
            # get ssd's for fits if in datacols
            stopdf = self.data[self.data.ttype=='stop']
            if 'probe' in stopdf.columns and self.ssd_method=='all':
                stopdf = stopdf[stopdf.probe==1]
            self.set_model_ssds(stopdf)
            if self.ssd_method=='all':
                get_df_ssds = lambda df: np.round(df.ssd.unique(), 1).astype(int)
                ssds = [get_df_ssds(df) for _, df in stopdf.groupby(g_cols)]
                ssd_list = [np.sort(issd).tolist() for issd in ssds]
            else:
                ssd_list = [['sacc'] for i in range(self.nrows)]
        self.make_idx_cols(ssd_list)
        masterDF_header = g_cols + self.p_cols + self.q_cols
        return masterDF_header


    def make_idx_cols(self, ssd_list=None):
        """ make idx-specific headers in event of missing data
        make all other headers if not done yet
        """
        if not hasattr(self, 'q_cols'):
            self.make_q_cols()
        if not hasattr(self, 'p_cols'):
            self.make_p_cols(ssd_list)
        if not hasattr(self, 'f_cols'):
            self.make_f_cols()
        if ssd_list:
            acc = [self.p_cols[0]]
            self.idx_cols = [acc + issd + self.q_cols for issd in ssd_list]
        else:
            self.idx_cols = [self.p_cols + self.q_cols]*self.nrows


    def make_q_cols(self):
        """ make header names for correct/error RT quants
        in observedDF, yhatdf, and wtsDF
        """
        cq = ['c' + str(int(n * 100)) for n in self.model.quantiles]
        eq = ['e' + str(int(n * 100)) for n in self.model.quantiles]
        self.q_cols = cq + eq


    def make_p_cols(self, ssd_list=None):
        """ make header names for response accuracy in observedDF,
        yhatdf, and wtsDF (including SSDs if stop model)
        """
        self.p_cols = ['acc']
        if ssd_list:
            ssd_unique = np.unique(np.hstack(ssd_list)).tolist()
            self.p_cols = self.p_cols + ssd_unique


    def make_f_cols(self):
        """ make header names for various fit statistics in fitdf
        (model parameters, goodness-of-fit measures, etc)
        """
        self.poptdf_cols = np.sort(list(self.inits)).tolist()
        self.f_cols = ['idx', 'pvary', 'nvary', 'AIC', 'BIC', 'nfev', 'df', 'ndata', 'chi', 'rchi', 'logp', 'cnvrg']


    def save_results(self, saveobserved=False):
        """ Saves yhatdf and fitdf results to model output dir
        ::Arguments::
            saveobserved (bool):
                if True will write observedDF & wtsDF to
                model output dir
        """
        fname = self.model.model_id
        if self.model.is_nested:
            fname='nested_models'
        make_fname = lambda savestr: os.path.join(self.resultsdir, '_'.join([fname, savestr+'.csv']))
        yName, fName, pName = [make_fname(dfType) for dfType in ['yhat', 'finfo', 'popt']]
        self.model.yhatdf.to_csv(yName, index=False)
        self.model.fitdf.to_csv(fName, index=False)
        self.model.poptdf.to_csv(pName, index=False)
        if saveobserved:
            self.observedDF.to_csv(os.path.join(self.resultsdir, make_fname('observed_data')))
            self.wtsDF.to_csv(os.path.join(self.resultsdir, make_fname('cost_weights')))


    def make_results_dir(self, custompath=None, get_path=False):
        """ make directory for writing model output and figures
        dir is named according to model_id, navigate to dir
        after ensuring it exists
        """
        self.resultsdir = os.path.abspath(os.path.expanduser('~'))
        if custompath is not None:
            self.resultsdir = os.path.join(self.resultsdir, custompath)
        elif self.model.is_nested:
            self.resultsdir = os.path.join(self.resultsdir, "nested_models")
        else:
            self.resultsdir = os.path.join(self.resultsdir, self.model.model_id)
        if not os.path.isdir(self.resultsdir):
            os.makedirs(self.resultsdir)
        if get_path:
            return self.resultsdir
