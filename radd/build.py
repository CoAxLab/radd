#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.models import Simulator
from radd.tools import messages
from radd.CORE import RADDCore
from radd.tools.vis import plot_model_fits

class Model(RADDCore):
    """ Main class for instantiating, fitting, and simulating models.
    Inherits from RADDCore parent class (see CORE module).
    Many of the naming conventions as well as the logic behind constructing parameter
    dependencies on task condition are taken from HDDM (http://ski.clps.brown.edu/hddm_docs/)
    ::Arguments::
        data (pandas DF):
            data frame with columns 'idx', 'rt', 'acc', 'ttype', 'response',
            <Condition Name> declared in depends_on values
        kind (str):
            declares model type ['dpm', 'irace', 'pro']
            append 'x' to front of model name to include a dynamic
            bias signal in model
        inits (dict):
            dictionary of parameters (v, a, tr, ssv, z) used to initialize model
        fit_on (str):
            set if model fits 'average', 'subjects', 'bootstrap' data
        depends_on (dict):
            set parameter dependencies on task conditions
            (ex. depends_on={'v': 'Condition'})
        weighted (bool):
            if True (default), perform fits using a weighted least-squares approach
        dynamic (str):
            set dynamic bias signal to follow an exponential or hyperbolic
            form when fitting models with 'x' included in <kind> attr
        quantiles (array):
            set the RT quantiles used to fit model
    """

    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', weighted=True, depends_on={'all':'flat'}, ssd_method=None, dynamic='hyp', quantiles=np.array([.1, .3, .5, .7, .9])):

        super(Model, self).__init__(data=data, inits=inits, fit_on=fit_on, depends_on=depends_on, kind=kind, dynamic=dynamic, quantiles=quantiles, weighted=weighted, ssd_method=ssd_method)

    def optimize(self, fit_flat=True, fit_cond=True, multiopt=True, best_inits=None, progress=False, plot_fits=False, saveplot=False, kde_quant_plots=True):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        """
        nfits = len(self.observed)
        for i in range(nfits):
            popt = self.__check_inits__(self.inits)
            if fit_flat or self.is_flat:
                y, wts = self.iter_flat[i]
                self.set_fitparams(idx=i, y=y, wts=wts, nlevels=1, flat=True)
                finfo, popt, yhat = self.optimize_flat(popt, multiopt=multiopt, best_inits=best_inits, progress=progress)
            if fit_cond and not self.is_flat:
                y, wts = self.iter_cond[i]
                self.set_fitparams(idx=i, y=y, wts=wts, nlevels=self.nlevels, flat=False)
                finfo, popt, yhat = self.optimize_conditional(popt, multiopt)
            self.assess_fit(finfo, popt, yhat)
            if plot_fits:
                self.plot_model_fits(y=y, yhat=yhat, kde_quant=kde_quant_plots, save=saveplot)

    def optimize_flat(self, p, multiopt=True, best_inits=None, progress=False):
        """ optimizes flat model to data collapsing across all conditions
        ::Arguments::
            p (dict):
                parameter dictionary to initalize model, if None uses init params
                passed by Model object
        ::Returns::
            yhat_flat (array): model-predicted data array
            finfo_flat (pd.Series): fit info (AIC, BIC, chi2, redchi, etc)
            popt_flat (dict): optimized parameters dictionary
        """
        if multiopt:
            # Global Optimization w/ Basinhopping (+TNC)
            ntrials = self.fitparams['ntrials']
            self.set_fitparams(ntrials=10000)
            p = self.opt.hop_around(p, best_inits=best_inits, progress=progress)
            self.set_fitparams(ntrials=ntrials)
            print('Finished Hopping Around')
        # Flat Simplex Optimization of Parameters at Global Minimum
        finfo, popt, yhat = self.opt.gradient_descent(inits=p, is_flat=True)
        return finfo, popt, yhat

    def optimize_conditional(self, p, multiopt=True):
        """ optimizes full model to all conditions in data
        ::Arguments::
            p (dict): parameter dictionary to initalize model, if None uses init params
        ::Returns::
            yhat (array): model-predicted data array
            finfo (pd.Series): fit info (AIC, BIC, chi2, redchi, etc)
            popt (dict): optimized parameters dictionary
        """
        if multiopt:
            ntrials = self.fitparams['ntrials']
            self.set_fitparams(ntrials=10000)
            # Pretune Conditional Parameters
            p, funcmin = self.opt.run_basinhopping(p, is_flat=False)
            self.set_fitparams(ntrials=ntrials)
        # Final Simplex Optimization
        finfo, popt, yhat = self.opt.gradient_descent(inits=p, is_flat=False)
        return finfo, popt, yhat

    def assess_fit(self, finfo, popt, yhat):
        """ wrapper for analyze.assess_fit calculates and stores
        rchi, AIC, BIC and other fit statistics
        """
        fp = dict(deepcopy(self.fitparams))
        fp['yhat'] = yhat.flatten()
        y = fp['y'].flatten()
        wts = fp['wts'].flatten()
        # fill finfo dict with goodness-of-fit info
        finfo['chi'] = np.sum(wts * (fp['yhat'] - y)**2).astype(np.float32)
        finfo['ndata'] = len(fp['yhat'])
        finfo['df'] = finfo.ndata - finfo.nvary
        finfo['rchi'] = finfo.chi / finfo.df
        finfo['logp'] = finfo.ndata * np.log(finfo.rchi)
        finfo['AIC'] = finfo.logp + 2 * finfo.nvary
        finfo['BIC'] = finfo.logp + np.log(finfo.ndata * finfo.nvary)
        self.log_fit_info(finfo, popt, fp)
        self.finfo = finfo
        self.popt = popt
        self.yhat = yhat
        try:
            self.fill_df(data=yhat, fitparams=fp, dftype='yhat')
            self.fill_df(data=finfo, fitparams=fp, dftype='fit')
        except Exception:
            print('fill_df error, already optimized? try new model')
            print('self.finfo, self.popt, and self.yhat still accessible from last fit')

    def fill_df(self, data, fitparams, dftype='fit'):
        if dftype=='fit':
            data['idx'] = self.idx[fitparams['idx']]
            next_row = np.argmax(self.fitDF.isnull().any(axis=1))
            keys = self.handler.f_cols
            self.fitDF.loc[next_row, keys] = data
            if self.fit_on=='average':
                fit_df = self.fitDF.dropna().copy()
                fit_df.idx='average'
                self.fitDF = fit_df.set_index('idx').T
        elif dftype=='yhat':
            nl = fitparams['nlevels']
            data = data.reshape(nl, int(data.size/nl))
            next_row = np.argmax(self.yhatDF.isnull().any(axis=1))
            keys = self.handler.idx_cols[next_row]
            yhat_df = self.yhatDF.copy()
            for i in range(nl):
                data_series = pd.Series(data[i], index=keys)
                yhat_df.loc[next_row+i, keys] = data_series
            if self.fit_on=='average':
                yhat_df = yhat_df.dropna()
                yhat_df.idx='average'
                self.yhatDF = yhat_df.copy()

    def log_fit_info(self, finfo, popt, fitparams):
        """ write meta-information about latest fit
        to logfile (.txt) in working directory
        """
        fp = dict(deepcopy(fitparams))
        # lmfit-structured fit_report to write in log file
        param_report = self.opt.param_report
        # log all fit and meta information in working directory
        messages.logger(param_report, finfo=finfo, popt=popt, fitparams=fp, kind=self.kind)

    def plot_model_fits(self, y, yhat, fitparams=None, kde_quant=True, save=False):
        """ wrapper for radd.tools.vis.plot_model_fits """
        if fitparams is None:
            fitparams=self.fitparams
        plot_model_fits(y, yhat, fitparams, kde_quant=kde_quant, save=save)

    def simulate(self, p=None, analyze=True):
        """ simulate yhat vector using
        :: Arguments ::
            p (dict):
                parameters dictionary
            analyze (bool):
                if True (default) returns yhat vector. else, returns decision traces
        :: Returns ::
            out (array):
                1d array if analyze is True, else ndarray of decision traces
        """
        if p is None:
            p = self.inits
        return self.opt.simulator.sim_fx(p, analyze=analyze)
