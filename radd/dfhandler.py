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
        self.pc_map = model.pc_map
        self.conds = model.conds
        self.nconds = model.nconds
        self.nlevels = model.nlevels
        self.cond_matrix = model.cond_matrix
        self.nrows = self.nidx * model.nlevels
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
        nrows = self.nrows
        nan_data = np.zeros((nrows, len(odf_header)))*np.nan
        qprob = self.quantiles
        ssdmethod = self.ssd_method
        datdf = self.grpData.apply(analyze.rangl_data, qprob, ssdmethod).sortlevel(0)
        groupvalues = datdf.reset_index()[self.groups].values
        self.observedDF = pd.DataFrame(nan_data, columns=odf_header, index=range(nrows))
        self.observedDF.loc[:, self.groups] = groupvalues
        self.wtsDF = self.make_wts_df()
        for rowi in range(nrows):
            # fill observedDF one row at a time, using idx_rows
            self.observedDF.loc[rowi, self.idx_cols[rowi]] = datdf.values[rowi]
        if self.fit_on=='average':
            observed_err = self.observedDF.groupby(self.conds).sem()*2
            self.observed_err = observed_err.loc[:, 'acc':].values.squeeze()
        else:
            self.varDF=None

    def make_wts_df(self, weighted=True):
        """ calculate and store cost_function weights
        for all subjects/conditions in data
        """
        wtsdf = self.observedDF.copy()
        if weighted:
            # calc & fill wtsDF with idx quantile and accuracy weights (ratios)
            quant_wts, acc_wts = self.calc_empirical_weights()
            wtsdf.loc[:, self.q_cols] = quant_wts
            wtsdf.loc[:, self.p_cols] = acc_wts
            wts_numeric = wtsdf.loc[:, 'acc':]
            wtsdf.loc[:, 'acc':] = wts_numeric.apply(analyze.fill_nan_vals, axis=1)
        else:
            wtsdf.loc[:, self.p_cols+self.q_cols] = 1.
        return wtsdf.copy()

    def calc_empirical_weights(self):
        """ calculates weight vectors for observed correct & err RT quantiles and
        go and stop accuracy for each subject (see funcs in radd.tools.analyze)
        """
        data = self.data.copy()
        # calculate weights for RT quantiles (correct & error trials seperately)
        quant_wts = analyze.idx_quant_weights(data=data, groups=self.groups, nlevels=self.nlevels, quantiles=self.quantiles, max_wt=self.max_wt)
        # calculate weights for accuracy (go and stop trials separately)
        acc_wts = analyze.idx_acc_weights(data, conds=self.conds, ssd_method=self.ssd_method)
        return quant_wts, acc_wts

    def make_yhat_df(self):
        """ make empty dataframe for storing model predictions (yhat)
        """
        yhatcols = self.observedDF.columns
        indx = np.arange(self.nlevels)
        yhatdf = pd.DataFrame(np.nan, index=indx, columns=yhatcols)
        minfo = self.observedDF[self.groups[1:]].iloc[:self.nlevels, :]
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
        minfo = self.observedDF[self.groups[1:]].iloc[:self.nlevels, :]
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
        popt = self.model.simulator.vectorize_params(popt)
        popt_vals = np.array([popt[pkey] for pkey in self.poptdf_cols]).T
        poptdf.loc[:, self.poptdf_cols] = popt_vals
        poptdf['idx'] = str(fitparams.idx)
        poptdf['pvary'] = '_'.join(list(self.model.depends_on))
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
        yhat = yhat.reshape(nl, int(yhat.size/nl))
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
        self.model.ssd = analyze.get_model_ssds(stopdf, self.conds, self.ssd_method)

    def make_headers(self, ssd_list=None):
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
            ssd_unique = np.unique(np.hstack(ssd_list))
            self.p_cols = self.p_cols + ssd_unique.tolist()

    def make_f_cols(self):
        """ make header names for various fit statistics in fitdf 
        (model parameters, goodness-of-fit measures, etc)
        """
        self.poptdf_cols = np.sort(list(self.inits)).tolist()
        self.f_cols = ['idx', 'pvary', 'nvary', 'AIC', 'BIC', 'nfev', 'df', 'ndata', 'chi', 'rchi', 'logp', 'cnvrg']

    def save_results(self, save_observed=False):
        """ Saves yhatdf and fitdf results to model output dir
        ::Arguments::
            save_observed (bool):
                if True will write observedDF & wtsDF to
                model output dir
        """
        fname = self.model.model_id
        if self.model.is_nested:
            fname='nested_models'
        make_fname = lambda savestr: '_'.join([fname, savestr+'.csv'])
        self.model.yhatdf.to_csv(make_fname('yhat'), index=False)
        self.model.fitdf.to_csv(make_fname('finfo'), index=False)
        self.model.poptdf.to_csv(make_fname('popt'), index=False)
        if save_observed:
            self.observedDF.to_csv(make_fname('observed_data'))
            self.wtsDF.to_csv(make_fname('cost_weights'))

    def read_results(self, ftype='finfo', fname=None, dropcols='Unnamed: 0'):
        """ read fits/yhat csv files into pandas DF
        ::Arguments::
            ftype (str):
                data type: if 'finfo' reads fitdf, if 'yhat' reads yhatdf
            path (str):
                custom path if not reading from self.resultsdir
            fname (str):
                custom file name if not nested_models or model_id
        ::Returns::
            df (DataFrame): pandas df with requested data
        """
        path = self.resultsdir
        if fname is None:
            if self.model.is_nested:
                fname='nested_models'
            else:
                fname = self.model.model_id
        fname = '_'.join([fname, ftype])
        full_fpath = os.path.join(path, ''.join([fname, '.csv']))
        df = pd.read_csv(full_fpath)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        return df.dropna()

    def make_results_dir(self, custompath=None, get_path=False):
        """ make directory for writing model output and figures
        dir is named according to model_id, navigate to dir
        after ensuring it exists
        """
        parentdir = os.path.expanduser('~')
        if custompath is not None:
            parentdir = os.path.join(parentdir, custompath)
        abspath = os.path.abspath(parentdir)
        if self.model.is_nested:
            self.resultsdir = os.path.join(abspath, "nested_models")
        else:
            self.resultsdir = os.path.join(abspath, self.model.model_id)
        if not os.path.isdir(self.resultsdir):
            os.makedirs(self.resultsdir)
        os.chdir(self.resultsdir)
        if get_path:
            return self.resultsdir

    def params_io(p={}, io='w', path=None, iostr='popt'):
        """ read // write parameters dictionaries
        """
        if path is None:
            path = self.resultsdir
        fname = os.path.join(path, ''.join([iostr, '.csv']))
        if io == 'w':
            pd.Series(p).to_csv(fname)
        elif io == 'r':
            p = pd.read_csv(fname, index_col=0)#, header=None)
            #p = dict(zip(ps[0], ps[1]))
            return p
