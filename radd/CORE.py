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
        self.verbose = verbose
        self.kind = kind
        self.fit_on = fit_on
        self.ssd_method = ssd_method
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
        self.finished_sampling = False
        self.track_subjects = False
        self.track_basins = False
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
        # initial parameters
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
        # initialize optimizer object for controlling fit routines
        # (updated with fitparams/basinparams whenever params are set)
        self.opt = Optimizer(fitparams=self.fitparams, basinparams=self.basinparams, kind=self.kind, inits=self.inits, depends_on=self.depends_on, pc_map=self.pc_map)
        # initialize model simulator, mainly accessed by the model optimizer object
        self.opt.simulator = Simulator(fitparams=self.fitparams, kind=self.kind, pc_map=self.pc_map)
        if self.verbose:
            self.is_prepared = messages.saygo(depends_on=self.depends_on, cond_map=self.clmap, kind=self.kind, fit_on=self.fit_on)
        else:
            self.prepared = True

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

    def set_fitparams(self, **kwargs):
        """ dictionary of fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'fitparams'):
            # initialize with default values and first arrays in observed_flat, flat_wts
            self.fitparams = {'idx':0, 'y': self.observed_flat[0], 'wts': self.flat_wts[0],
                'ntrials': 20000, 'tol': 1.e-30, 'method': 'nelder', 'maxfev': 2000,
                'tb': self.tb, 'nlevels': 1, 'fit_on': self.fit_on, 'kind': self.kind,
                'clmap': self.clmap, 'quantiles': self.quantiles, 'model_id': self.model_id,
                'depends_on': self.depends_on, 'flat': True, 'disp':True}
            self.fitparams = pd.Series(self.fitparams)
        else:
            # fill with kwargs (i.e. y, wts, idx, etc) for the upcoming fit
            for kw_arg, kw_val in kwargs.items():
                self.fitparams[kw_arg] = kw_val
        if hasattr(self, 'ssd'):
            self.__set_ssd_info__()
        if self.fitparams.quantiles.size != self.quantiles.size:
            self.quantiles = self.fitparams.quantiles
            self.__make_dataframes__()
            self.fitparams['y'] = self.observed_flat[self.fitparams.idx]
            self.fitparams['wts'] = self.flat_wts[self.fitparams.idx]
        if list(self.depends_on)[0]!=list(self.fitparams.depends_on)[0]:
            self.depends_on = self.fitparams.depends_on
            self.__format_pcmap__()
            self.fitparams['model_id'] = self.generate_model_id(get_id=True)
            self.is_flat = False
        if hasattr(self, 'opt'):
            self.opt.fitparams = self.fitparams
            self.opt.pc_map = self.pc_map
            self.opt.simulator.pc_map = self.pc_map
            self.opt.simulator.__update__(fitparams=self.opt.fitparams)

    def set_basinparams(self, **kwargs):
        """ dictionary of global fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'basinparams'):
            self.basinparams =  {'ninits': 5, 'nsamples': 5000, 'interval': 10, 'T': 1.,
            'disp': False, 'stepsize': .05, 'nsuccess': 40, 'tol': 1.e-20, 'method': 'TNC',
            'init_sample_method': 'best'}
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
            ssd = self.ssd[self.fitparams.idx]
        if self.fitparams.nlevels==1:
            # single vector (nlevels=1), don't squeeze
            ssd = np.mean(ssd, axis=0, keepdims=True)
        nssd = ssd.shape[-1]
        nss = int((.5 * self.fitparams.ntrials))
        nss_per_ssd = int(nss/nssd)
        ssd_ix = np.arange(nssd) * np.ones((ssd.shape[0], ssd.shape[-1])).astype(np.int)
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
        pc_map = {}
        if not self.is_flat:
            for p, cond in self.depends_on.items():
                levels = np.sort(self.data[cond].unique())
                pc_map[p] = ['_'.join([p, lvl]) for lvl in levels]
        self.pc_map = pc_map
        if hasattr(self, 'opt'):
            self.handler.pc_map = self.pc_map
            self.opt.pc_map = self.pc_map
            self.opt.fitparams = self.fitparams
            self.opt.simulator.pc_map = self.pc_map
            self.opt.simulator.__update__(fitparams=self.opt.fitparams)

    def sample_param_sets(self, pkeys=None, nsamples=None):
        """ sample *nsamples* (default=5000, see set_fitparams) different
        parameter sets (param_sets) and get model yhat for each set (param_yhats)
        """
        if pkeys is None:
            pkeys = np.sort(list(self.inits))
        if nsamples is None:
            nsamples = self.basinparams['nsamples']
        self.param_sets = theta.random_inits(pkeys, ninits=nsamples, kind=self.kind, as_list_of_dicts=True)
        self.param_yhats = [self.opt.simulator.sim_fx(params_i) for params_i in self.param_sets]
        self.finished_sampling = True

    def filter_param_sets(self):
        """ sample *nsamples* (default=5000, see set_fitparams) different
        parameter sets (param_sets) and get model yhat for each set (param_yhats)
        """
        if not hasattr(self, 'param_sets'):
            self.sample_param_sets()
        nkeep = self.basinparams['ninits']
        keep_method = self.basinparams['init_sample_method']
        inits_list, globalmin = theta.filter_param_sets(self.param_sets, self.param_yhats, self.fitparams, nkeep=nkeep, keep_method=keep_method)
        return inits_list, globalmin

    def fill_yhatDF(self, yhat=None, fitparams=None):
        """ wrapper for filling & updating model yhatDF
        """
        if yhat is None:
            yhat = self.yhat
        if fitparams is None:
            fitparams = self.fitparams
        self.handler.fill_yhatDF(data=yhat, fitparams=fitparams)
        self.yhatDF = self.handler.yhatDF.copy()

    def fill_fitDF(self, finfo=None, fitparams=None):
        """ wrapper for filling & updating model fitDF
        """
        if finfo is None:
            finfo = self.finfo
        if fitparams is None:
            fitparams = self.fitparams
        self.handler.fill_fitDF(data=finfo, fitparams=fitparams)
        self.fitDF = self.handler.fitDF.copy()

    def write_results(self, save_observed=False):
        """ wrapper for dfhandler.write_results saves yhatDF and fitDF
        results to model output dir
        ::Arguments::
            save_observed (bool):
                if True will write observedDF & wtsDF to
                model output dir
        """
        self.handler.write_results(save_observed)

    def plot_model_fits(self, y=None, yhat=None, fitparams=None, kde=True, err=None, save=False, bw=.008, sameaxis=False):
        """ wrapper for radd.tools.vis.plot_model_fits """
        from radd import vis
        if fitparams is None:
            fitparams=self.fitparams
        if y is None:
            y = fitparams['y']
        if yhat is None:
            if hasattr(self, 'yhat'):
                yhat = self.yhat
            else:
                yhat = deepcopy(y)
                print("model is unoptimized, no yhat provided")
        if self.fit_on=='average' and err is None:
            err = self.handler.observed_err
        vis.plot_model_fits(y, yhat, fitparams, err=err, save=save, bw=bw, sameaxis=sameaxis)

    def log_fit_info(self, finfo, popt, fitparams):
        """ write meta-information about latest fit
        to logfile (.txt) in working directory
        """
        fp = dict(deepcopy(fitparams))
        # lmfit-structured fit_report to write in log file
        param_report = self.opt.param_report
        # log all fit and meta information in working directory
        messages.logger(param_report, finfo=finfo, popt=popt, fitparams=fp, kind=self.kind)

    def generate_model_id(self, get_id=False, appendstr=None):
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
        if get_id:
            return self.model_id

    def make_progress_bars(self, models=None):
        """ initialize progress bars to track fit progress (subject fits,
        init optimization, etc)
        """
        from radd.tools.utils import NestedProgress
        n = self.basinparams['ninits']
        self.pbars = NestedProgress(name='glb_basin', n=n, title='Global Basin', color='green')
        self.pbars.add_bar(name='lcl_basin', bartype='infobar', title='Current Basin', color='red')
        if self.is_nested:
            self.pbars.add_bar(name='models', n=self.nmodels, title='Nested Fits', color='blue')
        self.track_basins=True
        if self.fit_on=='subjects':
            self.track_subjects = True
            self.pbars.add_bar(name='idx', n=self.nidx, title='Subject Fits', color='blue')
        self.fitparams['disp']=False
        self.basinparams['disp']=False

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
