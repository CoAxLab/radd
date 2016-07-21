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

    def optimize(self, plotfits=True, saveplot=False, saveresults=True, saveobserved=False, custompath=None, sameaxis=False, progress=False):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        """
        if np.any([saveplot, saveresults]):
            self.handler.make_results_dir(custompath)
        if progress and not self.is_nested:
            self.make_progress_bars()
        for ix in range(len(self.observed)):
            if self.track_subjects:
                self.pbars.update(name='idx', i=ix, new_progress=self.idx[i])
            if not hasattr(self, 'flat_popt'):
                self.set_fitparams(ix=ix, nlevels=1)
                self.optimize_flat()
            if not self.is_flat:
                self.set_fitparams(ix=ix, nlevels=self.nlevels)
                self.optimize_conditional()
            if plotfits:
                self.plot_model_fits(save=saveplot, sameaxis=sameaxis)
        if progress and not self.is_nested:
            self.pbars.clear()
        if saveresults:
            self.handler.save_results(saveobserved)

    def optimize_flat(self):
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
        if hasattr(self, 'pbars'):
            self.pbars.reset_bar('glb_basin', init_state=globalmin)
            pbars = self.pbars
        # Global Optimization w/ Basinhopping (+TNC)
        p = self.opt.hop_around(inits=inits, pbars=pbars)
        # Flat Simplex Optimization of Parameters at Global Minimum
        self.finfo, self.popt, self.yhat = self.opt.gradient_descent(p=p, flat=False)
        self.flat_popt = self.popt
        if self.is_flat:
            self.write_results()

    def optimize_conditional(self, p=None):
        """ optimizes full model to all conditions in data
        ::Arguments::
            p (dict): parameter dictionary to initalize model, if None uses init params
        ::Returns::
            yhat (array): model-predicted data array
            finfo (pd.Series): fit info (AIC, BIC, chi2, redchi, etc)
            popt (dict): optimized parameters dictionary
        """
        p = self.__check_inits__(deepcopy(self.flat_popt))
        # Pretune Conditional Parameters
        p, fmin = self.opt.run_basinhopping(p)
        # Final Simplex Optimization
        self.finfo, self.popt, self.yhat = self.opt.gradient_descent(p=p, flat=False)
        self.write_results()

    def optimize_nested_models(self, models=[], cond=None, saveplot=True, plotfits=True, progress=False, keeplog=True):
        """ optimize externally defined models in model_list using same init parameters
        as the current model for conditional fits (NOTE: only works with fit_on='average')
        """
        self.is_nested = True;
        self.nmodels=len(models)
        if not self.is_flat:
            cond = list(self.clmap)[0]
            models.insert(0, list(self.depends_on)[0])
        else:
            if cond is None:
                print("Must provide <cond>")
                return
        for i, pvary in enumerate(models):
            if progress and i>0:
                self.pbars.update(name='models', i=i)
            self.set_fitparams(depends_on={pvary: cond})
            self.optimize(plotfits=plotfits, progress=progress, saveplot=saveplot)
        if progress:
            self.pbars.clear()

    def write_results(self, finfo=None, popt=None, yhat=None):
        """ logs fit info to txtfile, fills yhatDF and fitDF
        """
        finfo, popt, yhat =self.set_results(finfo, popt, yhat)
        self.log_fit_info(finfo, popt, self.fitparams)
        self.handler.fill_yhatDF(data=yhat, fitparams=self.fitparams)
        self.handler.fill_fitDF(data=finfo, fitparams=self.fitparams)

    def plot_model_fits(self, y=None, yhat=None, kde=True, err=None, save=False, bw=.008, sameaxis=False):
        """ wrapper for radd.tools.vis.plot_model_fits
        """
        if y is None:
            y = self.fitparams.y
        if yhat is None:
            try:
                yhat = self.yhat
            except NameError:
                yhat = deepcopy(y)
        if self.fit_on=='average' and err is None:
            err = self.handler.observed_err
        plot_model_fits(y, yhat, self.fitparams, err=err, save=save, bw=bw, sameaxis=sameaxis)

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
