    #!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.models import Simulator
from radd.CORE import RADDCore
from radd import vis
from radd.tools import utils

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

    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', depends_on={'all':'flat'}, weighted=True, ssd_method=None, quantiles=np.arange(.1, 1.,.1)):

        super(Model, self).__init__(data=data, inits=inits, fit_on=fit_on, depends_on=depends_on, kind=kind, quantiles=quantiles, weighted=weighted, ssd_method=ssd_method)

    def optimize(self, flat_popt=None, plotfits=True, saveplot=False, saveresults=True, saveobserved=False, custompath=None, progress=False):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        ::Arguments::
            plotfits (bool):
                if True (default), plot model predictions over observed data
            saveplot (bool):
                if True (default is False), save plots to "~/<self.model_id>/"
            saveresults (bool):
                if True (default), save fitdf, yhatdf, and txt logs to "~/<self.model_id>/"
            saveobserved (bool):
                if True (default is False), save observedDF to "~/<self.model_id>/"
            custompath (str):
                path starting from any subdirectory of "~/" (e.g., home).
                all saved output will write to "~/<custompath>/<self.model_id>/"
            progress (bool):
                track progress across ninits and basinhopping
        """
        self.toggle_pbars(progress=progress)
        if np.any([saveplot, saveresults]):
            self.handler.make_results_dir(custompath=custompath)
        if flat_popt is None:
            self.sample_param_sets()
        for ix in range(len(self.observed)):
            if hasattr(self, 'idxbar'):
                self.idxbar.update(value=ix, status=self.idx[ix])
            if flat_popt is None:
                self.set_fitparams(ix=ix, nlevels=1)
                flat_popt = self.optimize_flat(self.param_sets[ix])
            if not self.is_flat:
                self.set_fitparams(ix=ix, nlevels=self.nlevels)
                self.optimize_conditional(flat_popt)
            if plotfits:
                self.plot_model_fits(save=saveplot)
            if saveresults:
                self.handler.save_results(saveobserved)
        if progress and not self.is_nested:
            self.optimizer.gbar.clear()
            self.optimizer.ibar.clear()

    def optimize_flat(self, init_list_ix):
        """ optimizes flat model to data collapsing across all conditions
        ::Arguments::
            None
        ::Returns::
            None
        ::Attributes Created::
            yhat_flat (array): model-predicted data array (ndim = self.levels)
            finfo_flat (pd.Series): fit info (AIC, BIC, chi2, redchi, etc)
            popt_flat (dict): optimized parameters dictionary
        """
        init_list_ix = deepcopy(init_list_ix)
        # Global Optimization w/ Basinhopping (+TNC)
        p = self.optimizer.hop_around(inits=init_list_ix)
        # Flat Simplex Optimization of Parameters at Global Minimum
        self.finfo, self.popt, self.yhat = self.optimizer.gradient_descent(p=p)
        if self.is_flat:
            self.write_results()
        return self.popt

    def optimize_conditional(self, flat_popt):
        """ optimizes full model to all conditions in data
        ::Arguments::
            None
        ::Returns::
            None
        ::Attributes Created::
            yhat (array): model-predicted data array (ndim = self.levels)
            finfo (pd.Series): fit info (AIC, BIC, chi2, redchi, etc)
            popt (dict): optimized parameters dictionary
            flat_popt (dict): deepcopy of popt
        """
        p = self.__check_inits__(deepcopy(flat_popt))
        # Pretune Conditional Parameters
        p, fmin = self.optimizer.run_basinhopping(p)
        # Final Simplex Optimization
        self.finfo, self.popt, self.yhat = self.optimizer.gradient_descent(p=p, flat=False)
        self.write_results()

    def nested_optimize(self, models=[], flat_popt=None, saveplot=True, plotfits=True, custompath=None, progress=False, saveresults=True, saveobserved=False):
        """ optimize a series of models using same init parameters where the i'th model
            has depends_on = {<models[i]> : <cond>}.
            NOTE: only for models with fit_on='average'
        ::Arguments::
            models (list):
                list of depends_on dictionaries to fit using a single set of init parameters (self.flat_popt)
            plotfits (bool):
                if True (default), plot model predictions over observed data
            saveplot (bool):
                if True (default), save plots to model.handler.results_dir
            custompath (str):
                path starting from any subdirectory of "~/" (e.g., home).
                all saved output will write to "~/<custompath>/<self.model_id>/"
            progress (bool):
                track progress across model fits, ninits, and basinhopping
        """
        self.is_nested = True
        if np.any([saveplot, saveresults]):
            self.handler.make_results_dir(custompath=custompath)
        if flat_popt is None:
            models = [{'all': 'flat'}] + models
            self.sample_param_sets()
        pnames = self.toggle_pbars(progress=progress, models=models)
        # loop over depends_on dictionaries and optimize cond models
        for i, depends_on in enumerate(models):
            self.set_fitparams(depends_on=depends_on)
            if progress:
                self.mbar.update(value=i, status=pnames[i])
            if flat_popt is None:
                flat_popt = self.optimize_flat(self.param_sets[0])
                continue
            self.optimize_conditional(flat_popt)
            if plotfits:
                self.plot_model_fits(save=saveplot)
            if saveresults:
                self.handler.save_results(saveobserved)
        if progress:
            self.mbar.clear()
            self.optimizer.gbar.clear()
            self.optimizer.ibar.clear()

    def recover_model(self, popt=None, yhat=None, nsamples=None, ninits=None, plotparams=True, plotfits=False, progress=False):
        """ fit model to synthetic data similar to observed y-vector
        and compare init params to recovered params to test model identifiabilty
        """
        # set nlevels so simulator result has "conditional" shape
        self.set_fitparams(force_conditional=True)
        if popt is None:
            popt = self.popt
        if yhat is None:
            yhat = self.simulate(p=popt, set_observed=True)
        if nsamples is None:
            nsamples = self.basinparams['nsamples']
        if ninits is None:
            ninits = self.basinparams['ninits']
        nkeep = self.basinparams['ninits']+1
        self.sample_param_sets(nsamples=nsamples, nkeep=nkeep)
        self.optimize(progress=progress, plotfits=plotfits)
        if plotparams:
            # plot init_params agains optimized param estimates
            vis.compare_param_estimates(popt, self.popt, self.depends_on)

    def write_results(self, finfo=None, popt=None, yhat=None):
        """ logs fit info to txtfile, fills yhatdf and fitdf
        """
        finfo, popt, yhat = self.set_results(finfo, popt, yhat)
        self.log_fit_info(finfo, popt, self.fitparams)
        self.yhatdf = self.handler.fill_yhatdf(yhat=yhat, fitparams=self.fitparams)
        self.fitdf = self.handler.fill_fitdf(finfo=finfo, fitparams=self.fitparams)
        self.poptdf = self.handler.fill_poptdf(popt=popt, fitparams=self.fitparams)

    def plot_model_fits(self, y=None, yhat=None, kde=True, err=None, save=False, bw=.008, savestr=None):
        """ wrapper for radd.tools.vis.plot_model_fits
        """
        if y is None:
            y = self.fitparams.y
        if yhat is None:
            try:
                yhat = self.yhat
            except AttributeError:
                yhat = deepcopy(y)
        if self.fit_on=='average' and err is None:
            err = self.handler.observed_err
        vis.plot_model_fits(y, yhat, self.fitparams, err=err, save=save, bw=bw, savestr=savestr)

    def simulate(self, p=None, analyze=True, set_observed=False):
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
            p = self.popt
        p = deepcopy(p)
        yhat = self.simulator.sim_fx(p, analyze=analyze)
        if set_observed:
            yhat = yhat.reshape(self.observed[0].shape)
            self.observed = [yhat]
            self.observed_flat = [yhat.mean(axis=0)]
        return yhat
