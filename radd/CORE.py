#!/usr/local/bin/env python
from __future__ import division
from future.utils import listvalues
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from scipy.stats.mstats import mquantiles as mq
from radd.tools import theta, analyze
from radd import dfhandler


class RADDCore(object):
    """ Parent class for constructing attributes and methods used by
    of Model objects. Not meant to be used directly.

    Contains methods for building dataframes, generating observed data vectors
    that are entered into cost function during fitting as well as calculating
    summary measures and weight matrix for weighting residuals during optimization.
    """
    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', depends_on=None, dynamic='hyp', percentiles=np.array([.1, .3, .5, .7, .9]), ssd_method=None, weighted=True, verbose=False):
        self.verbose = verbose
        self.kind = kind
        self.fit_on = fit_on
        self.ssd_method = ssd_method
        self.dynamic = dynamic
        self.weighted = weighted
        self.percentiles = percentiles
        self.tb = data[data.response == 1].rt.max()
        self.idx = list(data.idx.unique())
        self.nidx = len(self.idx)
        if depends_on is None:
            data = data.copy()
            data['cond'] = 'flat'
            self.conds = ['cond']
            self.is_flat = True
            self.depends_on = {}
        else:
            self.depends_on = depends_on
            self.conds = listvalues(depends_on)
            self.is_flat = False
        # PARAMETER INITIALIZATION
        if inits is None:
            self.__get_default_inits__()
        else:
            self.inits = inits
        self.nconds = len(self.conds)
        self.levels = [np.sort(data[cond].unique()) for cond in self.conds]
        self.nlevels = np.sum([len(lvls) for lvls in self.levels])
        self.groups = np.hstack([['idx'], self.conds]).tolist()
        self.data = data
        self.__prepare_fit__()

    def __prepare_fit__(self):
        """ model setup and initiates dataframes. Automatically run when Model object is initialized
        *pc_map is a dict containing parameter names as keys with values
                corresponding to the names given to that parameter in Parameters object
                (see optmize.Optimizer).
        *Parameters (p[pkey]=pval) that are constant across conditions are broadcast as [pval]*n.
                Conditional parameters are treated as arrays with distinct values [V1, V2...Vn], one for
                each condition.
        pc_map (dict): keys: conditional parameter names (i.e. 'v')
                    values: keys + condition names ('v_bsl, v_pnl')
                    |<--- PARAMETERS OBJECT [LMFIT] <------- [IN]
                    |---> p = {'v_bsl': V1, 'v_pnl': V2...} --->|
                    |<--- pc_map = {'v':['v_bsl', 'v_pnl']} <---|
                    |---> p['v'] = array([V1, V2]) -------> [OUT]
        """
        from radd.fit import Optimizer
        from radd.models import Simulator
        self.pc_map = {}
        if not self.is_flat:
            self.__set_pcmap__()
        if 'ssd' in self.data.columns:
            self.set_ssd()
        # initialize dataframe handler
        self.handler = dfhandler.DataHandler(self)
        # generate dataframes for observed data, popt, fitinfo, etc
        self.__make_dataframes__()
        # calculate costfx weights
        self.__set_wts__()
        # set fit parameters with default values
        self.set_fitparams()
        # set basinhopping parameters with default values
        self.set_basinparams()
        # initialize optimizer object for controlling fit routines
        # (updated with fitparams/basinparams whenever params are set)
        self.opt = Optimizer(fitparams=self.fitparams, basinparams=self.basinparams, kind=self.kind, inits=self.inits, depends_on=self.depends_on, pc_map=self.pc_map)
        # initialize model simulator, mainly accessed by the model optimizer object
        self.opt.simulator = Simulator(fitparams=self.fitparams, kind=self.kind, pc_map=self.pc_map)
        if self.verbose:
            self.is_prepared = messages.saygo(depends_on=self.depends_on, labels=self.levels,
            kind=self.kind, fit_on=self.fit_on, dynamic=self.dynamic)
        else:
            self.prepared = True

    def __make_dataframes__(self):
        """ wrapper for dfhandler.DataHandler.make_dataframes
        """
        self.handler.make_dataframes()
        # Group dataframe (nsubjects*nconds*nlevels x ndatapoints)
        self.observedDF = self.handler.observedDF.copy()
        # list (nsubjects long) of data arrays (nconds*nlevels x ndatapoints) to fit
        self.observed = self.handler.observed
        # list of flattened data arrays (averaged across conditions)
        self.observed_flat = self.handler.observed_flat
        # dataframe with same dim as observeddf for storing model predictions
        self.fits = self.handler.fits
        # dataframe with same dim as observeddf for storing fit info
        self.fitinfo = self.handler.fitinfo

    def __set_wts__(self):
        """ wrapper for analyze functions used to calculate
        weights used in cost function
        """
        # dataframe containing cost_function wts (see dfhandler docs)
        self.wtsDF = self.handler.wtsDF
        # list of arrays containing conditional costfx weights
        self.cond_wts = self.handler.cond_wts
        # list of arrays containing flat costfx weights
        self.flat_wts = self.handler.flat_wts

    def set_fitparams(self, get_params=False, **kwargs):
        """ dictionary of fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'fitparams'):
            y = self.observed[0]; wts = self.cond_wts[0]
            # initialize with default values and first arrays in observed_flat, flat_wts
            self.fitparams = {'idx':0, 'y':y, 'wts':wts, 'ntrials': 10000, 'maxfev': 5000,
            'maxiter': 500, 'disp': True, 'tol': 1.e-4, 'method': 'nelder', 'niter': 500,
            'tb': self.tb, 'fit_on': self.fit_on, 'percentiles': self.percentiles,
            'dynamic': self.dynamic, 'depends_on': self.depends_on}
        else:
            # fill with kwargs (i.e. y, wts, idx, etc) for the upcoming fit
            for kw_arg, kw_val in kwargs.items():
                self.fitparams[kw_arg] = kw_val
            if np.any([mk in self.kind for mk in ['dpm', 'irace', 'iact']]):
                self.fitparams['ssd'] = self.ssd[self.fitparams['idx']]
                self.fitparams['nssd'] = self.fitparams['ssd'].size
            self.opt.fitparams = self.fitparams
            self.opt.simulator.__update__(fitparams=self.opt.fitparams)

        if get_params:
            return self.fitparams

    def set_basinparams(self, get_params=False, **kwargs):
        """ dictionary of global fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'basinparams'):
            self.basinparams = {'nrand_inits': 20, 'nrand_samples': 500, 'interval': 5, 'niter': 40, 'stepsize': .06, 'niter_success': 30, 'method': 'TNC', 'tol': 1.e-4, 'maxiter': 100, 'disp': True}
        else:
            # fill with kwargs for the upcoming fit
            for kw_arg, kw_val in kwargs.items():
                self.basinparams[kw_arg] = kw_val
            self.opt.basinparams = self.basinparams
        if get_params:
            return self.basinparams

    def set_ssd(self, index=['idx'], scale=.001, get=False):
        """ set model attr "ssd" as list of np.arrays
        ssds to use when simulating data during optimization
        """
        sdf = self.data[self.data.ttype=='stop'].copy()
        if self.ssd_method is None:
            self.__determine_ssd_method__()
        if get and hasattr(self, "ssd"):
            get_df_ssds = lambda df: df.ssd.unique()
            ssds = [get_df_ssds(df) for _, df in sdf.groupby(self.groups)]
            return [np.sort(issd).tolist() for issd in ssds]
        if self.ssd_method == 'all':
            get_df_ssds = lambda df: df.groupby(self.conds).ssd.unique().values
            cond_ssds =  [get_df_ssds(df) for _,df in sdf.groupby('idx')]
        elif self.ssd_method == 'central':
            mean_cond_ssd_df = sdf.pivot_table('ssd', index='idx', columns=self.conds)
            cond_ssds = list(mean_cond_ssd_df.values)
        self.ssd = [np.sort(np.vstack(ssds))*scale for ssds in cond_ssds]

    def __determine_ssd_method__(self):
        """ Determine how to include ssd in cost function
        only called if not supplied by user.
        if all ssd have same ntrials, ssd_method = 'all':
            costfx includes stopaccuracy at each ssd
        if ssd is set by adaptive tracking procedure
        (and thus different ntrials per ssd) ssd_method = 'central':
            costfx includes stopaccuracy at mean ssd
        """
        stopdf = self.data[self.data.ttype=='stop'].copy()
        ssd_n = [df.size for _, df in stopdf.groupby('ssd')]
        # test if equal # of trials per ssd
        #return ssd_n
        if ssd_n[1:] == ssd_n[:-1]:
            self.ssd_method = 'all'
        else:
            self.ssd_method = 'central'

    def __set_pcmap__(self):
        """ format pc_map, dictionary used by Simulator to extract conditional
        parameter values by name from lmfit Parameters object and store in
        standard p dictionary as an array """
        for p, cond in self.depends_on.items():
            levels = np.sort(self.data[cond].unique())
            self.pc_map[p] = ['_'.join([p, lvl]) for lvl in levels]

    def __extract_popt_fitinfo__(self, finfo=None):
        """ wrapper for DataHandler.extract_popt_fitinfo()
        """
        popt = self.handler.extract_popt_fitinfo(finfo=finfo)
        return popt

    def rangl_data(self, data):
        """ wrapper for analyze.rangl_data
        """
        rangled = analyze.rangl_data(data, percentiles=self.percentiles, tb=self.tb)
        return rangled

    def resample_data(self, data):
        """ wrapper for analyze.resample_data
        """
        resampled = analyze.resample_data(data, n=100, data_style=self.data_style, tb=self.tb, kind=self.kind)
        return resampled

    def assess_fit(self, finfo=None):
        """ wrapper for analyze.assess_fit calculates and stores
        rchi, AIC, BIC and other fit statistics
        """
        return analyze.assess_fit(finfo)

    def params_io(self, p={}, io='w', iostr='popt'):
        """ read // write parameters dictionaries
        """
        if io == 'w':
            pd.Series(p).to_csv(''.join([iostr, '.csv']))
        elif io == 'r':
            ps = pd.read_csv(''.join([iostr, '.csv']), header=None)
            p = dict(zip(ps[0], ps[1]))
            return p

    def fits_io(self, fitparams, fits=[], io='w', iostr='fits'):
        """ read // write y, wts, yhat arrays
        """
        y = fitparams['y'].flatten()
        wts = fitparams['wts'].flatten()
        fits = fits.flatten()
        if io == 'w':
            index = np.arange(len(fits))
            df = pd.DataFrame({'y': y, 'wts': wts, 'yhat': fits}, index=index)
            df.to_csv(''.join([iostr, '.csv']))
        elif io == 'r':
            df = pd.read_csv(''.join([iostr, '.csv']), index_col=0)
            return df

    def __remove_outliers__(self, sd=1.5, verbose=False):
        self.data = analyze.remove_outliers(self.data, sd=sd, verbose=verbose)

    def __get_default_inits__(self):
        self.inits = theta.get_default_inits(kind=self.kind, dynamic=self.dynamic, depends_on=self.depends_on)

    def __get_optimized_params__(self, include_ss=False, fit_noise=False):
        params = theta.get_optimized_params(kind=self.kind, dynamic=self.dynamic, depends_on=self.depends_on)
        return params

    def __check_inits__(self, inits=None, pro_ss=False, fit_noise=False):
        if inits is None:
            inits = dict(deepcopy(self.inits))
        self.inits = theta.check_inits(inits=inits, pdep=list(self.depends_on), kind=self.kind, pro_ss=pro_ss, fit_noise=fit_noise)

    def __rename_bad_cols__(self):
        self.data = analyze.rename_bad_cols(self.data)
