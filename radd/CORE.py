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
from radd.tools import messages
from radd import theta, vis

class RADDCore(object):
    """ Parent class for constructing attributes and methods used by
    of Model objects. Not meant to be used directly.

    Contains methods for building dataframes, generating observed data vectors
    that are entered into cost function during fitting as well as calculating
    summary measures and weight matrix for weighting residuals during optimization.
    """
    def __init__(self, data=None, kind='xdpm', inits=None, fit_on='average', depends_on={'all':'flat'}, quantiles=np.arange(.1, 1.,.1), ssd_method=None, weighted=True, verbose=False, custompath=None, nested_models=None):
        self.kind = kind
        self.fit_on = fit_on
        self.ssd_method = ssd_method
        self.weighted = weighted
        self.quantiles = quantiles
        self.tb = data[data.response == 1].rt.max()
        self.idx = list(data.idx.unique())
        self.nidx = len(self.idx)
        self.inits = inits
        self.data = data
        self.set_conditions(depends_on)
        self.__prepare_fit__()
        self.finished_sampling = False
        self.track_subjects = False
        self.track_basins = False
        self.pbars = None
        self.is_nested = False

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
        from radd.optimize import Optimizer
        from radd.models import Simulator
        if self.inits is None:
            self.__get_default_inits__()
        # pc_map (see docstrings)
        self.__format_pcmap__()
        # create model_id string for naming output
        self.generate_model_id()
        # initialize DataHandler & generate I/O dataframes
        self.__make_dataframes__()
        # set fit parameters with default values
        self.set_fitparams()
        # set basinhopping parameters with default values
        self.set_basinparams()
        # initialize model simulator, mainly accessed by the model optimizer object
        self.simulator = Simulator(fitparams=self.fitparams, kind=self.kind, pc_map=self.pc_map)
        # initialize optimizer object for controlling fit routines
        # (updated with fitparams/basinparams whenever params are set)
        self.optimizer = Optimizer(simulator=self.simulator, basinparams=self.basinparams)

    def __make_dataframes__(self):
        """ wrapper for dfhandler.DataHandler.make_dataframes
        """
        from radd.dfhandler import DataHandler
        # initialize dataframe handler
        self.handler = DataHandler(self)
        # make dataframes
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
        # list of arrays containing conditional costfx weights
        self.cond_wts = self.handler.cond_wts
        # list of arrays containing flat costfx weights
        self.flat_wts = self.handler.flat_wts
        # define iterables containing fit y & wts for each fit
        self.iter_flat = zip(self.observed_flat, self.flat_wts)
        self.iter_cond = zip(self.observed, self.cond_wts)

    def set_fitparams(self, force_conditional=False, **kwargs):
        """ dictionary of fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'fitparams'):
            # initialize with default values and first arrays in observed_flat, flat_wts
            self.fitparams = {'ix':0, 'ntrials': 20000, 'tol': 1.e-30, 'method': 'nelder',
                'maxfev': 3000, 'tb': self.tb, 'nlevels': 1, 'fit_on': self.fit_on,
                'kind': self.kind, 'clmap': self.clmap, 'quantiles': self.quantiles,
                'model_id': self.model_id,  'depends_on': self.depends_on}
            self.fitparams = pd.Series(self.fitparams)
        else:
            # fill with kwargs (i.e. y, wts, ix, etc) for the upcoming fit
            for kw_arg, kw_val in kwargs.items():
                self.fitparams[kw_arg] = kw_val
        if 'quantiles' in list(kwargs):
            self.__update_quantiles__()
        if 'depends_on' in list(kwargs):
            reformat_dataframes = False
            if self.is_flat:
                reformat_dataframes=True
            self.set_conditions(kwargs['depends_on'])
            if reformat_dataframes:
                self.__prepare_fit__()
        self.update_data(force_conditional)
        if hasattr(self, 'ssd'):
            self.__set_ssd_info__()
        if hasattr(self, 'optimizer'):
            self.simulator = self.optimizer.update(fitparams=self.fitparams, pc_map=self.pc_map, get_simulator=True)

    def set_basinparams(self, **kwargs):
        """ dictionary of global fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'basinparams'):
            self.basinparams =  {'ninits': 3, 'nsamples': 3000, 'interval': 10, 'T': 1.,
            'stepsize': .05,  'niter': 100, 'nsuccess': 60, 'tol': 1.e-20, 'method': 'TNC',
            'init_sample_method': 'best', 'progress': False, 'disp': False}
        else:
            # fill with kwargs for the upcoming fit
            for kw_arg, kw_val in kwargs.items():
                self.basinparams[kw_arg] = kw_val
        if hasattr(self, 'optimizer'):
            self.optimizer.update(basinparams=self.basinparams)

    def update_data(self, force_conditional=False):
        """ called when ix (int) is passed to fitparams as kwarg.
        Fills fitparams with y and wts vectors corresponding to ix'th
        arrays in observed(_flat) and (flat/cond)_wts lists.
        """
        if force_conditional:
            self.fitparams['nlevels'] = self.nlevels
        nlevels = self.fitparams.nlevels
        i = self.fitparams['ix']
        if nlevels>1:
            self.fitparams['y'] = self.observed[i]
            self.fitparams['wts'] = self.cond_wts[i]
        else:
            self.fitparams['y'] = self.observed_flat[i]
            self.fitparams['wts'] = self.flat_wts[i]

    def set_conditions(self, depends_on=None):
        data = self.data.copy()
        self.depends_on = depends_on
        self.conds = np.unique(listvalues(self.depends_on)).tolist()
        if 'flat' in self.conds:
            self.is_flat = True
            data['flat'] = 'flat'
            self.data = data.copy()
        else:
            self.is_flat = False
        self.nconds = len(self.conds)
        self.clmap = {c: np.sort(data[c].unique()) for c in self.conds}
        self.nlevels = np.sum([len(lvls) for lvls in listvalues(self.clmap)])
        self.groups = np.hstack([['idx'], self.conds]).tolist()
        self.__format_pcmap__()
        if hasattr(self, 'ssd'):
            self.__set_ssd_info__()
        if hasattr(self, 'fitparams'):
            self.generate_model_id()
            self.set_fitparams(nlevels=self.nlevels, clmap=self.clmap)

    def __format_pcmap__(self):
        """ dict used by Simulator to extract conditional parameter values by name
        from lmfit Parameters object
            |<--- PARAMETERS OBJECT [LMFIT] <------- [IN]
            |---> p = {'v_bsl': V1, 'v_pnl': V2...} --->|
            |<--- pc_map = {'v':['v_bsl', 'v_pnl']} <---|
            |---> p['v'] = array([V1, V2]) -------> [OUT]
        """
        pc_map = {}
        if not self.is_flat:
            for p, cond in self.depends_on.items():
                levels = np.sort(self.data[cond].unique())
                pc_map[p] = ['_'.join([p, lvl]) for lvl in levels]
        self.pc_map = pc_map
        if hasattr(self, 'handler'):
            self.handler.pc_map = pc_map

    def __set_ssd_info__(self):
        """ set ssd_info for upcoming fit and store in fitparams dict
        """
        if self.fit_on=='average':
            ssd = np.array(self.ssd).mean(axis=0)
        else:
            # get ssd vector for fit index == ix
            ssd = self.ssd[self.fitparams['ix']]
        if self.fitparams.nlevels==1:
            # single vector (nlevels=1), don't squeeze
            ssd = np.mean(ssd, axis=0, keepdims=True)
        nssd = ssd.shape[-1]
        nss = int((.5 * self.fitparams.ntrials))
        nss_per_ssd = int(nss/nssd)
        ssd_ix = np.arange(nssd) * np.ones((ssd.shape[0], ssd.shape[-1])).astype(np.int)
        # store all ssd_info in fitparams, accessed by Simulator
        self.fitparams['ssd_info'] = [ssd, nssd, nss, nss_per_ssd, ssd_ix]

    def update_quantiles(self):
        """ recalculate observed dataframes w/ passed quantiles array
        """
        self.quantiles = self.fitparams.quantiles
        self.__make_dataframes__()
        self.fitparams['y'] = self.observed_flat[self.fitparams['ix']]
        self.fitparams['wts'] = self.flat_wts[self.fitparams['ix']]

    def sample_param_sets(self, pkeys=None, nsamples=None, nkeep=None):
        """ sample *nsamples* (default=5000, see set_fitparams) different
        parameter sets (param_sets) and get model yhat for each set (param_yhats)
        """
        if pkeys is None:
            pkeys = np.sort(list(self.inits))
        if nsamples is None:
            nsamples = self.basinparams['nsamples']
        if nkeep is None:
            nkeep = self.basinparams['ninits']
        unfiltered = theta.random_inits(pkeys, ninits=nsamples, kind=self.kind, as_list=True)
        self.filter_params(unfiltered, nsamples=nsamples, nkeep=nkeep)
        self.finished_sampling = True

    def filter_params(self, p_sets=None, nsamples=None, nkeep=None):
        """ sample *nsamples* (default=5000, see set_fitparams) different
        parameter sets (param_sets) and get model yhat for each set (param_yhats)
        """
        if nsamples is None:
            nsamples = self.basinparams['nsamples']
        if nkeep is None:
            nkeep = self.basinparams['ninits']
        if p_sets is None:
            p_sets = self.param_sets
        p_fmins = [self.optimizer.simulator.cost_fx(p, sse=1) for p in p_sets]
        method = self.basinparams['init_sample_method']
        self.param_sets, self.gmin = theta.filter_params(p_sets, p_fmins, nkeep=nkeep, method=method)

    def log_fit_info(self, finfo=None, popt=None, yhat=None):
        """ write meta-information about latest fit
        to logfile (.txt) in working directory
        """
        finfo, popt, yhat =self.set_results(finfo, popt, yhat)
        fp = dict(deepcopy(self.fitparams))
        fp['yhat'] = self.yhat
        # lmfit-structured fit_report to write in log file
        param_report = self.optimizer.param_report
        # log all fit and meta information in working directory
        messages.logger(param_report, finfo=finfo, popt=popt, fitparams=fp, kind=self.kind)

    def set_results(self, finfo=None, popt=None, yhat=None):
        if finfo is None:
            finfo = self.finfo
        if popt is None:
            popt = self.popt
        if yhat is None:
            yhat = self.yhat
        return finfo, popt, yhat

    def generate_model_id(self, appendstr=None):
        """ generate an identifying string with model information.
        used for reading and writing model output
        """
        model_id = list(self.depends_on)
        if 'all' in model_id:
            model_id = ['flat']
        model_id.append(self.fit_on)
        model_id.insert(0, self.kind)
        if appendstr is not None:
            model_id.append(appendstr)
        self.model_id = '_'.join(model_id)
        if hasattr(self, 'fitparams'):
            self.fitparams['model_id'] = self.model_id

    def set_testing_params(self, tol=1e-3, nsuccess=10, nsamples=50, ninits=2, maxfev=200, progress=True):
        self.set_fitparams(tol=tol, maxfev=maxfev)
        self.set_basinparams(tol=tol, ninits=ninits, nsamples=nsamples, nsuccess=nsuccess)
        self.optimizer.update(basinparams=self.basinparams, progress=progress)

    def __remove_outliers__(self, sd=1.5, verbose=False):
        """ remove slow rts (>sd above mean) from main data DF
        """
        from radd.tools.analyze import remove_outliers
        self.data = analyze.remove_outliers(self.data.copy(), sd=sd, verbose=verbose)

    def __get_default_inits__(self):
        """ if inits not provided by user, initialize with default values
        see tools.theta.get_default_inits
        """
        self.inits = theta.get_default_inits(kind=self.kind, depends_on=self.depends_on)

    def __check_inits__(self, inits):
        """ ensure inits dict is appropriate for Model kind
        see tools.theta.check_inits
        """
        inits = dict(deepcopy(inits))
        checked = theta.check_inits(inits=inits, depends_on=self.depends_on, kind=self.kind)
        return checked
