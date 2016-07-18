    #!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.models import Simulator
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

    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', weighted=True, depends_on={'all':'flat'}, ssd_method=None, quantiles=np.arange(.1, 1.,.1)):

        super(Model, self).__init__(data=data, inits=inits, fit_on=fit_on, depends_on=depends_on, kind=kind, quantiles=quantiles, weighted=weighted, ssd_method=ssd_method)

    def optimize(self, plotfits=True, saveplot=False, keeplog=True, saveresults=True, saveobserved=False, custompath=None, sameaxis=False, progress=False):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        """
        if np.any([keeplog, saveplot, saveresults]):
            self.results_dir = self.handler.make_results_dir(custompath, get_path=True)
        if progress:
            self.make_progress_bars()
        for i in range(len(self.observed)):
            if self.track_subjects:
                self.pbars.update(name='idx', i=i, new_progress=self.idx[i])
            if not hasattr(self, 'init_params'):
                y, wts = self.iter_flat[i]
                self.set_fitparams(idx=i, y=y, wts=wts, nlevels=1, flat=True)
                finfo, popt, yhat = self.optimize_flat(progress=progress)
                self.init_params = deepcopy(popt)
            else:
                popt = self.init_params
            if not self.is_flat:
                y, wts = self.iter_cond[i]
                self.set_fitparams(idx=i, y=y, wts=wts, nlevels=self.nlevels, flat=False)
                finfo, popt, yhat = self.optimize_conditional(p=popt)
            self.assess_fit(finfo, popt, yhat, keeplog)
            if plotfits:
                self.plot_model_fits(y=y, yhat=yhat, save=saveplot, sameaxis=sameaxis)
        if progress and not self.is_nested:
            self.pbars.clear()
        if saveresults:
            self.write_results(saveobserved)

    def optimize_flat(self, progress=False):
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
        pbars=None
        if not self.finished_sampling:
            self.sample_param_sets()
        inits, globalmin = self.filter_param_sets()
        if progress:
            self.pbars.reset_bar('glb_basin', init_state=globalmin)
            pbars = self.pbars
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
            p = self.__check_inits__(self.init_params)
        # Pretune Conditional Parameters
        p, fmin = self.opt.run_basinhopping(p)
        # Final Simplex Optimization
        finfo, popt, yhat = self.opt.gradient_descent(p=p)
        return finfo, popt, yhat

    def assess_fit(self, finfo, popt, yhat, keeplog=False):
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
        if keeplog:
            self.log_fit_info(finfo, popt, fp)
        self.fill_yhatDF(yhat=yhat, fitparams=fp)
        self.fill_fitDF(finfo=finfo, fitparams=fp)

    def optimize_nested_models(self, models=[], saveplot=True, plotfits=True, progress=False, keeplog=True):
        """ optimize externally defined models in model_list using same init parameters
        as the current model for conditional fits (NOTE: only works with fit_on='average')
        """
        self.is_nested = True; self.nmodels=len(models)
        if not hasattr(self, 'init_params'):
            self.optimize(plotfits=plotfits, progress=progress)
        for i, pdep in enumerate(models):
            if progress:
                self.pbars.update(name='models', i=i)
            self.set_fitparams(depends_on = {pdep: list(self.clmap)[0]})
            p = self.__check_inits__(deepcopy(self.init_params))
            finfo, popt, yhat = self.optimize_conditional(p=p)
            self.assess_fit(finfo, popt, yhat, keeplog)
            if plotfits:
                self.plot_model_fits(self.fitparams.y, yhat, self.fitparams, save=saveplot)
        if progress:
            self.pbars.clear()

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
