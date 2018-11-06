#!/usr/local/bin/env python
from __future__ import division
from future.utils import listvalues, listitems
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from numpy import array
from scipy.stats.mstats import mquantiles as mq
from lmfit import fit_report
from radd.tools import messages, utils, analyze
from radd import theta, vis
from itertools import product


class RADDCore(object):
    """ Parent class for constructing attributes and methods used by
    of Model objects. Not meant to be used directly

    Contains methods for building dataframes, generating observed data vectors
    that are entered into cost function during fitting as well as calculating
    summary measures and weight matrix for weighting residuals during optimization.
    """

    def __init__(self, data=None, kind='xdpm', inits=None, fit_on='average', depends_on={'all':'flat'}, ssd_method=None, weighted=True, verbose=False, custompath=None, nested_models=None, learn=False, bwfactors=None, ssdelay=False, gbase=False, quantiles=np.arange(.1, 1.,.1), presample=False, ksfit=False):
        self.kind = kind
        self.fit_on = fit_on
        self.ssd_method = ssd_method
        self.weighted = weighted
        self.quantiles = quantiles
        self.learn = learn
        self.ssdelay = ssdelay
        self.gbase = gbase
        self.custompath = custompath
        self.data = data.copy()
        # self.data = analyze.remove_outliers(data, 3.5)
        self.tb = analyze.estimate_timeboundary(self.data)
        self.idx = list(self.data.idx.unique())
        self.nidx = len(self.idx)
        self.bwfactors = bwfactors
        self.inits = inits
        self.finished_sampling = False
        self.track_subjects = False
        self.track_basins = False
        self.pbars = None
        self.is_nested = False
        self.ksfit=ksfit
        self.__prepare_fit__(depends_on, presample=presample)


    def __prepare_fit__(self, depends_on, presample=False):
        """ model setup and initiates dataframes. Automatically run when Model object is initialized
        *pcmap is a dict containing parameter names as keys with values
                corresponding to the names given to that parameter in Parameters object
                (see optmize.Optimizer).
        *Parameters (p[pkey]=pval) that are constant across conditions are broadcast as [pval]*n.
                Conditional parameters are treated as arrays with distinct values [V1, V2...Vn], one for
                each condition.
        pcmap (dict): see bound __format_pcmap__ method
        """

        # from radd.optimize import Optimizer
        # from radd.models import Simulator
        from radd import optimize

        self.set_conditions(depends_on, self.bwfactors)

        if self.inits is None:
            self.__get_default_inits__()

        # pcmap (see docstrings)
        self.__format_pcmap__()
        # create model_id string for naming output
        self.generate_model_id()

        # initialize DataHandler & generate I/O dataframes
        self.__make_dataframes__()

        # set fit parameters with default values
        self.set_fitparams()

        # # set ssd info
        # self.__set_ssd_info__()

        # set basinhopping parameters with default values
        self.set_basinparams()

        # initialize optimizer object for controlling fit routines
        # (updated with fitparams/basinparams whenever params are set)
        self.opt = optimize.Optimizer(fitparams=self.fitparams, basinparams=self.basinparams, inits=self.inits, data=self.data)
        self.sim = self.opt.sim
        if self.learn:
            self.simRL = self.opt.simRL

        if presample:
            # sample init params
            self.sample_theta()


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
        self.observedErr = self.handler.observedErr.copy()
        # list (nsubjects long) of data arrays (nconds*nlevels x ndatapoints) to fit
        self.observed = self.handler.observed
        # list of flattened data arrays (averaged across conditions)
        self.observed_flat = self.handler.observed_flat

        # observed frequency bins dataframe
        # self.freqDF = self.handler.make_freq_df()

        # dataframe with same dim as observeddf for storing model predictions
        self.yhatdf = self.handler.yhatdf
        # dataframe with same dim as observeddf for storing fit info (& popt)
        self.fitdf = self.handler.fitdf
        # dataframe with same dim as observeddf for storing optimized params as matrix
        self.poptdf = self.handler.poptdf
        # dataframe containing cost_function wts (see dfhandler docs)
        self.wtsDF = self.handler.wtsDF

        try:
            # dataframe containing ssd's per idx (see dfhandler docs)
            self.ssdDF = self.handler.ssdDF
        except Exception:
            pass

        # list of arrays containing conditional costfx weights
        self.cond_wts = self.handler.cond_wts
        # list of arrays containing flat costfx weights
        self.flat_wts = self.handler.flat_wts

        # define iterables containing fit y & wts for each fit
        self.iter_flat = zip(self.observed_flat, self.flat_wts)
        self.iter_cond = zip(self.observed, self.cond_wts)
        # self.resultsdir = os.self.handler.make_results_dir(custompath=self.custompath, get_path=True)
        # get working directory
        self.resultsdir = os.path.abspath('./')


    def set_fitparams(self, force=None, **kwargs):
        """ dictionary of fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'fitparams'):

            # initialize with default values and first arrays in observed_flat, flat_wts
            #local methods = ['nelder', 'powell', 'lbfgsb' 'tnc', 'cobyla']
            self.fitparams = {'ix':0,
                            'ntrials': 10000,
                            'si': .1,
                            'dt':.003,
                            'tol': 1.e-25,
                            'method': 'nelder',
                            'maxfev': 800,
                            'maxiter': 800,
                            'kind': self.kind,
                            'clmap': self.clmap,
                            'pcmap':self.pcmap,
                            'depends_on': self.depends_on,
                            'ssd_method': self.handler.ssd_method,
                            'quantiles': self.quantiles,
                            'fit_on': self.fit_on,
                            'model_id': self.model_id,
                            'learn': self.learn,
                            'inits': self.inits,
                            'nlevels': 1,
                            'nidx': self.nidx,
                            'idx': self.idx[0],
                            'tb': self.tb}

            self.fitparams = pd.Series(self.fitparams)

        else:
            # fill with kwargs (i.e. y, wts, ix, etc)
            for kw_arg, kw_val in kwargs.items():
                self.fitparams[kw_arg] = kw_val

        if 'quantiles' in list(kwargs):
            self.update_quantiles()

        if 'depends_on' in list(kwargs):
            self.__prepare_fit__(kwargs['depends_on'])
            self.generate_model_id()
            self.set_fitparams(nlevels = self.nlevels, clmap=self.clmap, model_id=self.model_id)
            self.fitparams['nlevels'] = self.nlevels

        if force=='cond':
            self.fitparams['nlevels'] = self.nlevels

        elif force=='flat':
            self.fitparams['nlevels'] = 1

        self.update_data(self.fitparams.nlevels)

        if hasattr(self, 'ssdDF') and 'ssd' in self.data.columns:
            self.__set_ssd_info__()

        ksData=None
        if self.ksfit:
            ksData = self.get_ksdata(nlevels=self.fitparams['nlevels'])

        if hasattr(self, 'opt'):
            self.opt.update(fitparams=self.fitparams, inits=self.fitparams.inits, ksData=ksData)
            self.sim = self.opt.sim
            if self.learn:
                self.simRL = self.opt.simRL


    def set_basinparams(self, **kwargs):
        """ dictionary of global fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'basinparams'):
            self.basinparams =  {'ninits': 4,
                                'nsamples': 2500,
                                'interval': 10,
                                'T': .015,
                                'mutation': (0.5, 1),
                                'stepsize': .25,
                                'niter': 300,
                                'maxiter': 300,
                                'nsuccess': 100,
                                'polish_tol': 1.e-25,
                                'tol': .001,
                                'method': 'evolution',
                                'local_method': 'powell', #'L-BFGS-B', 'TNC'
                                'sample_method':'random',
                                'popsize': 15,
                                'recombination': .65,
                                'progress': True,
                                'strategy': 'best1bin',
                                'disp': False}
        else:
            # fill with kwargs for the upcoming fit
            for kw_arg, kw_val in kwargs.items():
                self.basinparams[kw_arg] = kw_val
            if self.basinparams['method']=='basin':
                self.basinparams['local_method']='TNC'
                self.basinparams['tol'] = 1e-20

        if hasattr(self, 'opt'):
            old_method = self.opt.basinparams['method']
            self.opt.update(basinparams=self.basinparams)
            self.sim = self.opt.sim
            if old_method != self.basinparams['method']:
                self.opt.make_progress_bars()


    def get_ksdata(self, nlevels=1):
        self.ksDataCond = {'corRT':[], 'errRT':[], 'goAcc':[], 'stopAcc':[]}
        self.ksDataFlat = deepcopy(self.ksDataCond)


        df = self.data.copy()
        self.ksDataFlat['corRT'].append(df[(df.ttype=='go')&(df.acc==1)].rt.values)
        self.ksDataFlat['errRT'].append(df[(df.ttype=='stop')&(df.acc==0)].rt.values)
        self.ksDataFlat['goAcc'].append(df[df.ttype=='go'].acc.mean())

        if self.ssd_method=='all':
            nssd = self.fitparams.ssd_info[1]
            self.ksDataFlat['stopAcc'].append(df.iloc[:, 3:3+nssd].mean())
        else:
            self.ksDataFlat['stopAcc'].append(df[df.ttype=='stop'].acc.mean())

        for col, vals in listitems(self.clmap):
            for lvl in vals:
                if lvl.isalnum():
                    try:
                        lvl = int(lvl)
                    except ValueError:
                        lvl = lvl
                dfi = self.data[self.data[col]==lvl]
                self.ksDataCond['corRT'].append(dfi[(dfi.ttype=='go')&(dfi.acc==1)].rt.values)
                self.ksDataCond['errRT'].append(dfi[(dfi.ttype=='stop')&(dfi.acc==0)].rt.values)
                self.ksDataCond['goAcc'].append(dfi[dfi.ttype=='go'].acc.mean())

                if self.ssd_method=='all':
                    dfStop = self.observedDF[self.observedDF[col]==lvl]
                    nssd = self.fitparams.ssd_info[1]
                    self.ksDataCond['stopAcc'].append(dfStop.iloc[:, 3:3+nssd].mean().values)
                else:
                    self.ksDataCond['stopAcc'].append(dfi[dfi.ttype=='stop'].acc.mean())

                #self.ksDataCond['stopAcc'].append(dfi[(dfi.ttype=='stop')&(dfi.probe==1)].acc.mean())

        if nlevels==1:
            return self.ksDataFlat
        return self.ksDataCond


    def update_data(self, nlevels=1):
        """ called when ix (int) is passed to fitparams as kwarg.
        Fills fitparams with y and wts vectors corresponding to ix'th
        arrays in observed(_flat) and (flat/cond)_wts lists.
        """
        i = self.fitparams['ix']
        if nlevels>1:
            self.fitparams['y'] = self.observed[i]
            self.fitparams['wts'] = self.cond_wts[i]
        else:
            self.fitparams['y'] = self.observed_flat[i]
            self.fitparams['wts'] = self.flat_wts[i]
        if self.fit_on=='subjects':
            self.fitparams['idx']=str(self.idx[i])
        else:
            self.fitparams['idx'] = 'avg'


    def sample_theta(self):

        pkeys = list(self.inits)
        nsamples = self.basinparams['nsamples']
        method = self.basinparams['sample_method']
        kind = self.kind

        if self.ssdelay:
            kind = '_'.join([kind, 'sso'])
        if self.gbase:
            kind = '_'.join([kind, 'z'])

        datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'inits')
        theta_fname = "{}_{}".format(method, kind)
        yhat_fname = "{}_{}_yhat".format(method, kind)
        thetaAll = pd.read_pickle(os.path.join(datadir, theta_fname), compression='xz')
        yhatAll = pd.read_pickle(os.path.join(datadir, yhat_fname), compression='xz')
        self.init_params = thetaAll.to_dict('records')
        groupedYhat = yhatAll.groupby('pset')
        yhats_ = groupedYhat.apply(analyze.rangl_data, self.ssd_method, self.quantiles)
        self.init_yhats = pd.DataFrame(np.vstack(yhats_.values))
        self.opt.init_params = self.init_params
        self.opt.init_yhats = self.init_yhats
        self.opt.sample_theta()


    def set_conditions(self, depends_on=None, bwfactors=None):
        data = self.data.copy()
        self.depends_on = depends_on
        self.conds = np.unique(np.hstack(listvalues(self.depends_on))).tolist()
        self.nconds = len(self.conds)
        if 'flat' in self.conds:
            self.is_flat = True
            data['flat'] = 'flat'
            self.data = data.copy()
        else:
            self.is_flat = False
        clevels = [np.sort(data[c].unique()) for c in self.conds]
        clevels = [np.array([str(lvl) for lvl in levels]) for levels in clevels]
        self.clmap = {c: lvls for c, lvls in zip(self.conds, clevels)}
        self.cond_matrix = np.array([lvls.size for lvls in clevels])
        self.nlevels = np.cumprod(self.cond_matrix)[-1]
        self.groups = np.hstack([['idx'], self.conds]).tolist()

        self.__format_pcmap__()
        if hasattr(self, 'ssdDF') and not self.is_nested:
            self.__set_ssd_info__()

        if hasattr(self, 'fitparams') and not self.is_nested:
            self.generate_model_id()
            self.set_fitparams(nlevels=self.nlevels, clmap=self.clmap, model_id=self.model_id)


    def __format_pcmap__(self):
        """ dict used by Simulator to extract conditional parameter values by name
        from lmfit Parameters object
            |<--- PARAMETERS OBJECT [LMFIT] <------- [IN]
            |---> p = {'v_bsl': V1, 'v_pnl': V2...} --->|
            |<--- pcmap = {'v':['v_bsl', 'v_pnl']} <---|
            |---> p['v'] = array([V1, V2]) -------> [OUT]
        """
        pcmap = {}
        for p, conds in self.depends_on.items():
            if isinstance(conds, list):
                levels = []
                for cond in conds:
                    levels.append(self.clmap[cond])
                level_data = list(product(*levels))
                clevels = ['_'.join([str(lvl) for lvl in lvls]) for lvls in level_data]
            else:
                clevels = [lvl for lvl in self.clmap[conds]]

            param_clevels = ['{}_{}'.format(p, clvl) for clvl in clevels]
            pcmap[p] = param_clevels

        self.pcmap = pcmap
        if hasattr(self, 'handler'):
            self.handler.pcmap = pcmap


    def __set_ssd_info__(self):
        """ set ssd_info for upcoming fit and store in fitparams dict
        """
        if self.fit_on=='average':
            ssdDF = self.ssdDF.copy()
            ssdDF = ssdDF.drop('idx', axis=1)
            ssd = ssdDF.groupby(self.conds).mean().values
        else:
            idx = self.idx[self.fitparams['ix']]
            # get ssd vector for fit index == ix
            ssd = self.ssdDF[self.ssdDF['idx'] == idx].groupby(self.conds).mean().values[:,1:]
            # if self.bwfactors is not None and hasattr(self, 'sim'):
            #     ix = self.conds.index([c for c in self.conds if c!=self.bwfactors][0])
            #     ssd = ssd[self.sim.pvary_ix[ix]]
            self.ssd = ssd
        if self.fitparams.nlevels==1:
            # single vector (nlevels=1), don't squeeze
            ssd = np.mean(ssd, axis=0, keepdims=True)
        nssd = ssd.shape[-1]
        nss = int((.5 * self.fitparams.ntrials))
        # nss = self.fitparams.ntrials
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
        model_id.insert(0, self.kind)
        fit_on = 'avg'
        if self.fit_on=='subjects':
            fit_on = 'idx'
        model_id.append(fit_on)
        if appendstr is not None:
            model_id.append(appendstr)
        self.model_id = '_'.join(model_id)
        if hasattr(self, 'fitparams'):
            self.fitparams['model_id'] = self.model_id


    def set_testing_params(self, tol=1e-20, nsuccess=50, nsamples=1000, ninits=2, maxfev=1000, progress=True):
        self.set_fitparams(tol=tol, maxfev=maxfev)
        self.set_basinparams(tol=tol, ninits=ninits, nsamples=nsamples, nsuccess=nsuccess)
        self.opt.update(basinparams=self.basinparams, progress=progress)


    def toggle_pbars(self, progress=False, models=None):
        self.set_basinparams(progress=progress)
        if not progress:
            return None
        if self.fit_on=='subjects':
            status = ''.join(['Subj {}', '/{}'.format(self.nidx)])
            self.idxbar = utils.PBinJ(n=self.nidx, color='y', status=status)
        if models is not None:
            pvary = [list(depends_on) for depends_on in models]
            pnames = [vis.parameter_name(p, False) for p in pvary]
            self.mbar = utils.PBinJ(n=len(pnames), color='b', status='{}')
            return pnames


    def __remove_outliers__(self, sd=1.5, verbose=False):
        """ remove slow rts (>sd above mean) from main data DF
        """
        from radd.tools.analyze import remove_outliers
        self.data = analyze.remove_outliers(self.data.copy(), sd=sd, verbose=verbose)


    def __get_default_inits__(self):
        """ if inits not provided by user, initialize with default values
        see tools.theta.get_default_inits
        """
        self.inits = theta.get_default_inits(kind=self.kind, depends_on=self.depends_on, learn=self.learn, ssdelay=self.ssdelay, gbase=self.gbase)


    def __check_inits__(self, inits):
        """ ensure inits dict is appropriate for Model kind
        see tools.theta.check_inits
        """
        inits = dict(deepcopy(inits))
        checked = theta.check_inits(inits=inits, depends_on=self.depends_on, kind=self.kind)
        return checked
