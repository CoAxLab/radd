#!usr/bin/env python
from __future__ import division
import os
from future.utils import listvalues
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy import array
from radd.tools import analyze
from scipy.stats.mstats import mquantiles as mq
from scipy.stats.mstats_extras import mjci

class DataHandler(object):

    def __init__(self, model, max_wt=2.5):
        self.model = model
        self.data = model.data
        self.inits = model.inits
        self.model_id = model.model_id
        self.idx = model.idx
        self.nidx = model.nidx
        self.max_wt = max_wt
        self.ssd_method = model.ssd_method
        self.kind = model.kind
        self.fit_on = model.fit_on
        self.quantiles = model.quantiles
        self.groups = model.groups
        self.depends_on = model.depends_on
        self.conds = model.conds
        self.nconds = model.nconds
        self.nlevels = model.nlevels
        self.nrows = self.nidx * self.nlevels * self.nconds
        self.grpData = self.data.groupby(self.groups)

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
        self.make_observed_groupDFs()
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
        masterDF_header = self.__make_headers__()
        data = self.data
        ncols = len(masterDF_header)
        nrows = self.nrows
        index = range(nrows)
        nan_data = np.zeros((nrows, ncols))*np.nan

        self.datdf = self.grpData.apply(self.rangl_data).sortlevel(0)
        self.dfvals = [self.datdf.values[i].astype(float) for i in index]
        self.observedDF = pd.DataFrame(nan_data, columns=masterDF_header, index=index)
        self.observedDF.loc[:, self.groups] = self.datdf.reset_index()[self.groups].values
        # make yhatDF to fill w/ model-predicted data arrays
        self.yhatDF = self.observedDF.copy()
        # make wtsDF for handling cost-fx weights
        self.wtsDF = self.observedDF.copy()
        # make fitDF for storing w/ goodness-of-fit stats and popt
        self.fitDF = pd.DataFrame(columns=self.f_cols, index=range(self.nidx))
        self.fitDF['idx'] = self.idx

        for rowi in range(nrows):
            # fill observedDF one row at a time, using idx_rows
            self.observedDF.loc[rowi, self.idx_cols[rowi]] = self.dfvals[rowi]

        if self.model.weighted:
            # Calculate p(resp) and rt quantile costfx weights
            idx_qwts, idx_pwts = self.estimate_cost_weights()
            # fill wtsDF with idx quantile wts (ratios)
            self.wtsDF.loc[:, self.q_cols] = idx_qwts
            # fill wtsDF with resp. probability wts (ratios)
            self.wtsDF.loc[:, self.p_cols] = idx_pwts
        else:
            self.wtsDF.loc[:, self.p_cols+self.q_cols] = 1
        if self.fit_on=='average':
            observed_err = self.observedDF.groupby(self.conds).std()
            self.observed_err = observed_err.loc[:, 'acc':].values.squeeze()
        else:
            self.varDF=None

    def rangl_data(self, data):
        """ called by __make_dataframes__ to generate
        observed data arrays
        """
        gac = data.query('ttype=="go"').acc.mean()
        grt = data.query('response==1 & acc==1').rt.values
        ert = data.query('response==1 & acc==0').rt.values
        gq = mq(grt, prob=self.quantiles)
        eq = mq(ert, prob=self.quantiles)
        data_vector = [gac, gq, eq]
        if 'ssd' in self.data.columns:
            stopdf = data.query('ttype=="stop"')
            if self.model.ssd_method=='all':
                sacc=stopdf.groupby('ssd').mean()['acc'].values
            elif self.model.ssd_method=='central':
                sacc = np.array([stopdf.mean()['acc']])
            data_vector.insert(1, sacc)
        return np.hstack(data_vector)

    def fill_fitDF(self, data, fitparams=None):
        """ fill fitDF with fit statistics
        ::Arguments::
            data (Series):
                fitinfo Series containing model statistics and
                optimized parameters (see Model.assess_fit() method)
            fitparams (dict):
                model.fitparams dict w/ meta info for last fit
        """
        if fitparams is None:
            fitparams = self.model.fitparams
        fitDF = self.fitDF.copy()
        data['idx'] = self.idx[fitparams['idx']]
        next_row = np.argmax(fitDF.isnull().any(axis=1))
        keys = self.f_cols
        fitDF.loc[next_row, keys] = data
        if self.fit_on=='average':
            fit_df = fitDF.dropna().copy()
            fit_df.idx='average'
            fitDF = fit_df.set_index('idx').T
        self.fitDF = fitDF.copy()

    def fill_yhatDF(self, data, fitparams=None):
        """ fill yhatDF with model predictions
        ::Arguments::
            data (ndarray):
                array containing model predictions (nlevels x ncols)
                where ncols is number of data columns in observedDF
            fitparams (dict):
                model.fitparams dict w/ meta info for last fit
        """
        if fitparams is None:
            fitparams = self.model.fitparams
        yhatDF = self.yhatDF.copy()
        nl = fitparams['nlevels']
        data = data.reshape(nl, int(data.size/nl))
        next_row = np.argmax(yhatDF.isnull().any(axis=1))
        keys = self.idx_cols[next_row]
        for i in range(nl):
            data_series = pd.Series(data[i], index=keys)
            yhatDF.loc[next_row+i, keys] = data_series
        if self.fit_on=='average':
            yhat_df = yhatDF.dropna()
            yhat_df.idx = 'average'
            yhatDF = yhat_df.copy()
        self.yhatDF = yhatDF.copy()

    def determine_ssd_method(self, stopdf):
        ssd_n = [df.size for _, df in stopdf.groupby('ssd')]
        # test if equal # of trials per ssd & return ssd_n
        all_equal_counts = ssd_n[1:] == ssd_n[:-1]
        if all_equal_counts:
            self.model.ssd_method = 'all'
        else:
            self.model.ssd_method = 'central'
        return self.model.ssd_method

    def set_model_ssds(self, stopdf, index=['idx'], scale=.001):
        """ set model attr "ssd" as list of np.arrays
        ssds to use when simulating data during optimization
        """
        if self.ssd_method is None:
            self.ssd_method = self.determine_ssd_method(stopdf)
        if self.ssd_method == 'all':
            get_df_ssds = lambda df: df.groupby(self.conds).ssd.unique().values
            cond_ssds =  [get_df_ssds(df) for _,df in stopdf.groupby('idx')]
        elif self.ssd_method == 'central':
            mean_cond_ssd_df = stopdf.pivot_table('ssd', index='idx', columns=self.conds)
            cond_ssds = list(mean_cond_ssd_df.values)
        self.model.ssd = [np.sort(np.vstack(ssds))*scale for ssds in cond_ssds]

    def estimate_cost_weights(self):
        """ calculate weights using observed variability
        across subjects (model.observedDF)
        """
        data = self.data
        nsplits = self.nlevels * self.nconds
        percents = self.quantiles
        nquant = percents.size
        # estimate quantile weights
        idx_qwts = self.idx_quant_weights()
        # estimate resp. probability weights
        idx_pwts = self.idx_acc_weights()
        return idx_qwts, idx_pwts

    def idx_quant_weights(self):
        """ calculates weight vectors for reactive RT quantiles by
        first estimating the SEM of RT quantiles for corr. and err. responses.
        (using Maritz-Jarrett estimatation: scipy.stats.mstats_extras.mjci).
        Then representing these variances as ratios.
        e.g.
              QSEM = mjci(rtvectors)
              wts = median(QSEM)/QSEM
        """
        idx_mjci = lambda x: mjci(x.rt, prob=self.quantiles)
        nidx = self.nidx
        nquant = self.quantiles.size
        groups = self.groups
        nsplits = self.nlevels * self.nconds
        # get all trials with response recorded
        godf = self.data.query('response==1')
        # sort by ttype first so go(acc==1) occurs before stop(acc==0)
        ttype_ordered_groups = np.hstack([groups, 'ttype', 'acc']).tolist()
        godf_grouped = godf.groupby(ttype_ordered_groups)
        # apply self.idx_mjci() to estimate quantile CI's
        quant_err = np.vstack(godf_grouped.apply(idx_mjci).values)
        # reshape [nidx   x   ncond * nquant * nacc]
        idx_qerr = quant_err.reshape(nidx, nsplits * nquant * 2)
        # calculate subject median across all conditions quantiles and accuracy
        # this implicitly accounts for n_obs as the mjci estimate of sem will be lower
        # for conditions with greater number of observations (i.e., more acc==1 in godf)
        idx_medians = np.nanmedian(idx_qerr, axis=1)
        idx_qratio = idx_medians[:, None] / idx_qerr
        # set extreme values to max_wt arg val
        idx_qratio[idx_qratio >= self.max_wt] = self.max_wt
        # reshape to fit in wtsDF[:, q_cols]
        return idx_qratio.reshape(nidx * nsplits, nquant * 2)

    def idx_acc_weights(self, index=['idx']):
        """ count number of observed responses across levels, transform into ratios
        (counts_at_each_level / np.median(counts_at_each_level)) for weight
        subject-level p(response) values in cost function.
        ::Arguments::
            df (DataFrame): group-level dataframe
            var (str): column header for variable to count responses
            conds (list): depends_on.values()
        """
        df = self.data.copy()
        if not self.model.is_flat:
            index = index + self.conds
        if 'ssd' in df.columns:
            if self.ssd_method=='all':
                df = df[df.ttype=='stop'].copy()
                split_by = 'ssd'
            else:
                split_by = 'ttype'
        else:
            split_by = self.conds
            _ = index.remove(split_by)
        df['n'] = 1
        countdf = df.pivot_table('n', index=index, columns=split_by, aggfunc=np.sum)
        idx_pwts = countdf.values / countdf.median(axis=1).values[:, None]
        if self.ssd_method=='all':
            go_wts = np.ones(countdf.shape[0])
            idx_pwts = np.concatenate((go_wts[:,None], idx_pwts), axis=1)
        return idx_pwts

    def __make_headers__(self, ssd_list=None):
        g_cols = self.groups
        if 'ssd' in self.data.columns:
            # get ssd's for fits if in datacols
            if 'ssd' in self.data.columns:
                stopdf = self.data[self.data.ttype=='stop']
                self.set_model_ssds(stopdf)
            if self.ssd_method=='all':
                get_df_ssds = lambda df: df.ssd.unique()
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
        in observedDF, yhatDF, and wtsDF
        """
        cq = ['c' + str(int(n * 100)) for n in self.quantiles]
        eq = ['e' + str(int(n * 100)) for n in self.quantiles]
        self.q_cols = cq + eq

    def make_p_cols(self, ssd_list=None):
        """ make header names for response accuracy in observedDF,
        yhatDF, and wtsDF (including SSDs if stop model)
        """
        self.p_cols = ['acc']
        if ssd_list:
            ssd_unique = np.unique(np.hstack(ssd_list))
            self.p_cols = self.p_cols + ssd_unique.tolist()

    def make_f_cols(self):
        """ make header names for various fit statistics in fitDF
        (model parameters, goodness-of-fit measures, etc)
        """
        params = np.sort(list(self.inits))
        if not self.model.is_flat:
            dep_keys = list(self.model.pc_map)
            cond_param_names = listvalues(self.model.pc_map)
            params = np.hstack([params, np.squeeze(cond_param_names)]).tolist()
            _ = [params.remove(pname) for pname in dep_keys]
        fit_cols = ['nfev', 'nvary', 'df', 'chi', 'rchi', 'logp', 'AIC', 'BIC', 'cnvrg']
        self.f_cols = np.hstack([['idx'], params, fit_cols]).tolist()

    def write_results(self, save_observed=False):
        """ Saves yhatDF and fitDF results to model output dir
        ::Arguments::
            save_observed (bool):
                if True will write observedDF & wtsDF to
                model output dir
        """
        make_fname = lambda savestr: '_'.join([self.model_id, savestr+'.csv'])
        self.yhatDF.to_csv(make_fname('yhat_df'))
        self.fitDF.to_csv(make_fname('finfo_df'))
        if save_observed:
            self.observedDF.to_csv(make_fname('observed_data'))
            self.wtsDF.to_csv(make_fname('cost_weights'))

    def make_results_dir(self, custompath=None, get_path=True):
        """ make directory for writing model output and figures
        dir is named according to model_id, navigate to dir
        after ensuring it exists
        """
        savepath = os.path.expanduser('~')
        if custompath is not None:
            savepath = os.path.join(savepath, custompath)
        abspath = os.path.abspath(savepath)
        abs_savepath = os.path.join(abspath, self.model_id)
        if not os.path.isdir(abs_savepath):
            os.makedirs(abs_savepath)
        os.chdir(abs_savepath)
        if get_path:
            return abs_savepath
