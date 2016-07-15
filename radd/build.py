#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.models import Simulator
from radd.tools import messages
from radd.CORE import RADDCore
from radd.vis import plot_model_fits

class Model(RADDCore):
    """ Main class for instantiating, fitting, and simulating models.
    Inherits from RADDCore parent class (see CORE module).
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
        quantiles (array):
            set the RT quantiles used to fit model
    """

    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', weighted=True, depends_on={'all':'flat'}, ssd_method=None, quantiles=np.array([.1, .3, .5, .7, .9])):
        super(Model, self).__init__(data=data, inits=inits, fit_on=fit_on, depends_on=depends_on, kind=kind, quantiles=quantiles, weighted=weighted, ssd_method=ssd_method)

    def optimize(self, fit_flat=True, fit_cond=True, progress=True, plot_fits=True, saveplot=False, keeplog=False, save_results=True, save_observed=False, custompath=None, pbars=None, sameaxis=False):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        """
        if np.any([keeplog, saveplot, save_results]):
            self.results_dir = self.handler.make_results_dir(custompath, get_path=True)
        if progress:
            pbars = self.make_progress_bars()
        for i in range(len(self.observed)):
            if self.track_subjects:
                pbars.update(name='idx', i=i, new_progress=self.idx[i])
            y, wts = self.iter_flat[i]
            self.set_fitparams(idx=i, y=y, wts=wts, nlevels=1, flat=True)
            finfo, popt, yhat = self.optimize_flat(pbars=pbars)
            if not self.is_flat:
                y, wts = self.iter_cond[i]
                self.set_fitparams(idx=i, y=y, wts=wts, nlevels=self.nlevels, flat=False)
                finfo, popt, yhat = self.optimize_conditional(p=popt)
            self.assess_fit(finfo, popt, yhat, keeplog)
            if plot_fits:
                self.plot_model_fits(y=y, yhat=yhat, save=saveplot, sameaxis=sameaxis)
        if progress:
            pbars.clear()
        if save_results:
            self.write_results(save_observed)

    def optimize_flat(self, pbars=None):
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
        globalmin = 1.;
        if not self.finished_sampling:
            self.sample_param_sets()
        inits, globalmin = self.filter_param_sets()
        if pbars is not None:
            pbars.reset_bar('glb_basin', init_state=globalmin)
        # Global Optimization w/ Basinhopping (+TNC)
        p = self.opt.hop_around(inits=inits, pbars=pbars)
        # Flat Simplex Optimization of Parameters at Global Minimum
        finfo, popt, yhat = self.opt.gradient_descent(p=p)
        return finfo, popt, yhat

    def optimize_conditional(self, p=None):
        """ optimizes full model to all conditions in data
        ::Arguments::
            p (dict): parameter dictionary to initalize model, if None uses init params
        ::Returns::
            yhat (array): model-predicted data array
            finfo (pd.Series): fit info (AIC, BIC, chi2, redchi, etc)
            popt (dict): optimized parameters dictionary
        """
        if p is None:
            p = self.__check_inits__(self.inits)
        # Pretune Conditional Parameters
        p, fmin = self.opt.run_basinhopping(p)
        # Final Simplex Optimization
        finfo, popt, yhat = self.opt.gradient_descent(p=p)
        return finfo, popt, yhat

    def assess_fit(self, finfo, popt, yhat, keep_log=False):
        """ wrapper for analyze.assess_fit calculates and stores
        rchi, AIC, BIC and other fit statistics
        """
        fp = deepcopy(self.fitparams)
        fp['yhat'] = yhat.flatten()
        y = fp['y'].flatten()
        wts = fp['wts'].flatten()
        # fill finfo dict with goodness-of-fit info
        finfo['chi'] = np.sum(wts * (fp['yhat'] - y)**2)
        finfo['ndata'] = len(fp['yhat'])
        finfo['df'] = finfo.ndata - finfo.nvary
        finfo['rchi'] = finfo.chi / finfo.df
        finfo['logp'] = finfo.ndata * np.log(finfo.rchi)
        finfo['AIC'] = finfo.logp + 2 * finfo.nvary
        finfo['BIC'] = finfo.logp + np.log(finfo.ndata * finfo.nvary)
        self.finfo = finfo
        self.popt = popt
        self.yhat = yhat
        if keep_log:
            self.log_fit_info(finfo, popt, fp)
        try:
            self.fill_yhatDF(yhat=yhat, fitparams=fp)
            self.fill_fitDF(finfo=finfo, fitparams=fp)
        except Exception:
            print('fill_df error, already optimized? try new model')
            print('latest finfo, popt, yhat available as model attributes')

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
