#!/usr/local/bin/env python
from __future__ import division
from future.utils import listvalues
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from numpy import array
from scipy.stats.mstats import mquantiles as mq
from lmfit import fit_report
from radd.tools import theta, analyze, messages

class RADDCore(object):
    """ Parent class for constructing attributes and methods used by
    of Model objects. Not meant to be used directly.

    Contains methods for building dataframes, generating observed data vectors
    that are entered into cost function during fitting as well as calculating
    summary measures and weight matrix for weighting residuals during optimization.
    """
    def __init__(self, data=None, kind='xdpm', inits=None, fit_on='average', depends_on={'all':'flat'}, dynamic='hyp', quantiles=np.array([.1, .3, .5, .7, .9]), ssd_method=None, weighted=True, verbose=False):

        self.verbose = verbose
        self.kind = kind
        self.fit_on = fit_on
        self.ssd_method = ssd_method
        self.dynamic = dynamic
        self.weighted = weighted
        self.quantiles = quantiles
        self.tb = data[data.response == 1].rt.max()
        self.idx = list(data.idx.unique())
        self.nidx = len(self.idx)
        self.depends_on = depends_on
        self.conds = np.unique(listvalues(depends_on)).tolist()
        if 'flat' in self.conds:
            self.is_flat = True
            data = data.copy()
            data['flat'] = 'flat'
        else:
            self.is_flat = False
        self.nconds = len(self.conds)
        self.clmap = {c: np.sort(data[c].unique()) for c in self.conds}
        self.nlevels = np.sum([len(lvls) for lvls in listvalues(self.clmap)])
        self.groups = np.hstack([['idx'], self.conds]).tolist()
        self.data = data
        self.inits = inits
        self.__prepare_fit__()
        self.iter_flat = zip(self.observed_flat, self.flat_wts)
        self.iter_cond = zip(self.observed, self.cond_wts)

    def __prepare_fit__(self):
        """ model setup and initiates dataframes. Automatically run when Model object is initialized
        *pc_map is a dict containing parameter names as keys with values
                corresponding to the names given to that parameter in Parameters object
                (see optmize.Optimizer).
        *Parameters (p[pkey]=pval) that are constant across conditions are broadcast as [pval]*n.
                Conditional parameters are treated as arrays with distinct values [V1, V2...Vn], one for
                each condition.
        pc_map (dict): see bound __format_pcmap__ method
        """
        from radd.fit import Optimizer
        from radd.models import Simulator
        from radd.dfhandler import DataHandler
        # initial parameters
        if self.inits is None:
            self.__get_default_inits__()
        # pc_map (see docstrings)
        self.pc_map = {}
        if not self.is_flat:
            self.__format_pcmap__()
        # initialize dataframe handler
        self.handler = DataHandler(self)
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
            self.is_prepared = messages.saygo(depends_on=self.depends_on, cond_map=self.cond_map, kind=self.kind, fit_on=self.fit_on, dynamic=self.dynamic)
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
        self.yhatDF = self.handler.yhatDF
        # dataframe with same dim as observeddf for storing fit info
        self.fitDF = self.handler.fitDF
        # dataframe containing cost_function wts (see dfhandler docs)
        self.wtsDF = self.handler.wtsDF

    def __set_wts__(self):
        """ wrapper for analyze functions used to calculate
        weights used in cost function
        """
        # list of arrays containing conditional costfx weights
        self.cond_wts = self.handler.cond_wts
        # list of arrays containing flat costfx weights
        self.flat_wts = self.handler.flat_wts

    def set_fitparams(self, **kwargs):
        """ dictionary of fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'fitparams'):
            y = self.observed_flat[0]
            wts = self.flat_wts[0]
            # initialize with default values and first arrays in observed_flat, flat_wts
            self.fitparams = {'idx':0, 'y':y, 'wts':wts, 'ntrials': 20000, 'maxfev': 2000,
            'disp':True, 'tol': 1.e-5, 'method': 'nelder', 'tb': self.tb, 'nlevels': 1,
            'fit_on': self.fit_on, 'clmap': self.clmap, 'dynamic':self.dynamic,
            'quantiles': self.quantiles, 'flat': True, 'depends_on': self.depends_on,
            'kind': self.kind}
        else:
            # fill with kwargs (i.e. y, wts, idx, etc) for the upcoming fit
            for kw_arg, kw_val in kwargs.items():
                self.fitparams[kw_arg] = kw_val
        if hasattr(self, 'ssd'):
            self.__set_ssd_info__()
        if hasattr(self, 'opt'):
            self.opt.fitparams = self.fitparams
            self.opt.simulator.__update__(fitparams=self.opt.fitparams)

    def set_basinparams(self, **kwargs):
        """ dictionary of global fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'basinparams'):
            self.basinparams =  {'nrand_inits': 5, 'nrand_samples': 5000, 'interval': 10,
            'disp': False, 'stepsize': .05, 'niter_success': 30, 'tol': 1.e-5, 'method': 'TNC'}
        else:
            # fill with kwargs for the upcoming fit
            for kw_arg, kw_val in kwargs.items():
                self.basinparams[kw_arg] = kw_val
            self.opt.basinparams = self.basinparams

    def __set_ssd_info__(self):
        """ set ssd_info for upcoming fit and store in fitparams dict
        """
        if self.fit_on=='average':
            ssd = np.array(self.ssd).mean(axis=0)
        else:
            # get ssd vector for fit number idx
            ssd = self.ssd[self.fitparams['idx']]
        if self.fitparams['flat']:
            # single vector (nlevels=1), don't squeeze
            ssd = np.mean(ssd, axis=0, keepdims=True)
        nssd = ssd.shape[-1]
        nss = int((.5 * self.fitparams['ntrials']))
        nss_per_ssd = int(nss/nssd)
        ssd_ix = np.arange(nssd)
        # store all ssd_info in fitparams, accessed by Simulator
        self.fitparams['ssd_info'] = [ssd, nssd, nss, nss_per_ssd, ssd_ix]

    def __format_pcmap__(self):
        """ dict used by Simulator to extract conditional parameter values by name
        from lmfit Parameters object
            |<--- PARAMETERS OBJECT [LMFIT] <------- [IN]
            |---> p = {'v_bsl': V1, 'v_pnl': V2...} --->|
            |<--- pc_map = {'v':['v_bsl', 'v_pnl']} <---|
            |---> p['v'] = array([V1, V2]) -------> [OUT]
        """
        for p, cond in self.depends_on.items():
            levels = np.sort(self.data[cond].unique())
            self.pc_map[p] = ['_'.join([p, lvl]) for lvl in levels]

    def __remove_outliers__(self, sd=1.5, verbose=False):
        """ remove slow rts (>sd above mean) from main data DF
        """
        self.data = analyze.remove_outliers(self.data.copy(), sd=sd, verbose=verbose)

    def __get_default_inits__(self):
        """ if inits not provided by user, initialize with default values
        see tools.theta.get_default_inits
        """
        self.inits = theta.get_default_inits(kind=self.kind, dynamic=self.dynamic, depends_on=self.depends_on)

    def __check_inits__(self, inits):
        """ ensure inits dict is appropriate for Model kind
        see tools.theta.check_inits
        """
        inits = dict(deepcopy(inits))
        checked = theta.check_inits(inits=inits, depends_on=self.depends_on, kind=self.kind)
        return checked
