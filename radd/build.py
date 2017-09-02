#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.models import Simulator
from radd.CORE import RADDCore
from radd import vis
from radd.tools import utils, analyze, messages
from radd.tools.analyze import pandaify_results, rangl_data


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

    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', depends_on={'all':'flat'}, weighted=True, ssd_method=None, learn=False, bwfactors=None, custompath=None, ssdelay=False, quantiles=np.array([.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95])):

        super(Model, self).__init__(data=data, inits=inits, fit_on=fit_on, depends_on=depends_on, kind=kind, quantiles=quantiles, weighted=weighted, ssd_method=ssd_method, learn=learn, bwfactors=bwfactors, custompath=custompath, ssdelay=ssdelay)


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
        self.custompath=custompath

        for ix in range(len(self.observed)):
            if hasattr(self, 'idxbar'):
                self.idxbar.update(value=ix, status=self.idx[ix])
            if flat_popt is None:
                self.set_fitparams(ix=ix, force='flat', nlevels=1)
                self.sample_param_sets()
                flat_popt = self.optimize_flat(self.param_sets)
            if not self.is_flat:
                self.set_fitparams(ix=ix, force='cond')
                self.optimize_conditional(flat_popt)
            if plotfits:
                self.plot_model_fits(save=saveplot)
            if saveresults:
                self.handler.save_results(saveobserved)
        if progress and not self.is_nested:
            self.opt.gbar.clear()
            self.opt.ibar.clear()


    def optimize_flat(self, param_sets=None):
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
        # Global Optimization w/ Basinhopping (+TNC)
        self.basin = self.opt.hop_around(param_sets)
        # Flat Simplex Optimization of Parameters at Global Minimum
        self.finfo, self.popt, self.yhat = self.opt.gradient_descent(p=self.basin)
        if self.is_flat:
            self.write_results()
        return self.popt


    def optimize_conditional(self, flatp, hop=False):
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
        basin = self.__check_inits__(deepcopy(flatp))
        self.set_fitparams(force='cond', inits=basin)
        # Pretune Conditional Parameters
        if hop:
            basin, fmin = self.opt.run_basinhopping(basin)
            self.opt.update(inits=basin)
        # Final Simplex Optimization
        self.finfo, self.popt, self.yhat = self.opt.gradient_descent(p=basin)
        # self.write_results()


    def nested_optimize(self, models=[], flatp=None, saveplot=True, plotfits=True, custompath=None, progress=False, saveresults=True, saveobserved=False):
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
        self.custompath = custompath
        if flatp is None:
            models = [{'all': 'flat'}] + models
            self.param_sets = self.opt.sample_param_sets()

        pnames = self.toggle_pbars(progress=progress, models=models)
        # loop over depends_on dictionaries and optimize cond models
        for i, depends_on in enumerate(models):
            self.set_fitparams(depends_on=depends_on)
            if progress:
                self.mbar.update(value=i, status=pnames[i])
            if flatp is None:
                self.param_sets = self.opt.sample_param_sets()
                flatp = self.optimize_flat(self.param_sets)
                continue
            self.optimize_conditional(flatp)
            if plotfits:
                self.plot_model_fits(save=saveplot)
            if saveresults:
                self.handler.save_results(saveobserved=saveobserved)
        if progress:
            self.mbar.clear()
            self.opt.gbar.clear()
            self.opt.ibar.clear()


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


    def log_fit_info(self, finfo=None, popt=None, yhat=None):
        """ write meta-information about latest fit
        to logfile (.txt) in working directory
        """
        self.opt.log_fit_info(finfo, popt, yhat)


    def write_results(self, finfo=None, popt=None, yhat=None):
        """ logs fit info to txtfile, fills yhatdf and fitdf
        """

        finfo, popt, yhat = self.set_results(finfo, popt, yhat)
        self.log_fit_info(finfo, popt, yhat)

        self.yhatdf = self.handler.fill_yhatdf(yhat=yhat, fitparams=self.fitparams)
        self.fitdf = self.handler.fill_fitdf(finfo=finfo, fitparams=self.fitparams)
        self.poptdf = self.handler.fill_poptdf(popt=popt, fitparams=self.fitparams)


    def plot_model_fits(self, y=None, yhat=None, kde=True, err=None, save=False, bw='scott', savestr=None, same_axis=True, clrs=None, lbls=None, cumulative=True, simdf=None, suppressLegend=False, simData=None, condData=None, shade=True, plot_error_rts=True, figure=None):
        """ wrapper for radd.tools.vis.plot_model_fits
        """
        data = self.handler.data.copy()

        if y is None:
            y = self.fitparams.y

        if yhat is None:
            try:
                yhat = self.yhat
            except AttributeError:
                print("No Model Predictions to Plot (need yhat argument)")
                yhat = analyze.rangl_data(simdf, quantiles=self.quantiles)
        if save:
            if savestr is None:
                savestr = self.fitparams.model_id
            if self.fitparams['fit_on']=='subjects':
                savestr = savestr + str(self.fitparams['idx'])

        if err is None:
            err = self.handler.observed_err

        if lbls is None and self.fitparams.nlevels>1:
            from itertools import product
            levels = [self.clmap[cond] for cond in self.conds]
            level_data = list(product(*levels))
            lbls = ['_'.join([str(lvl) for lvl in lvls]) for lvls in level_data]

        if self.ssd_method == 'central':
            ssd = self.ssdDF.groupby(self.conds).mean()[0].values
            ssderr = self.ssdDF.groupby(self.conds).sem()[0].values
        else:
            ssd = self.fitparams.ssd_info[0]
            ssderr = None

        fig = vis.plot_model_fits(y, yhat, err=err, quantiles=self.quantiles, ssd=ssd, ssderr=ssderr, bw=bw, same_axis=same_axis, clrs=clrs, lbls=lbls, cumulative=cumulative, save=save, savestr=savestr, suppressLegend=suppressLegend, shade=shade, plot_error_rts=plot_error_rts, figure=figure)


    def simulate(self, p=None, analyze=True, set_observed=False, raw_data=False):
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
            try:
                p = self.popt
            except Exception:
                p = self.__get_default_inits()
        p = deepcopy(p)
        yhat = self.sim.sim_fx(p, analyze=analyze)
        if set_observed:
            yhat = yhat.reshape(self.observed[0].shape)
            self.observed = [yhat]
            self.observed_flat = [yhat.mean(axis=0)]
        elif raw_data:
            p = self.sim.vectorize_params(deepcopy(p))
            return self.make_simulated_dataframe(yhat[0], yhat[1], p)
        return yhat


    def make_simulated_dataframe(self, DVg, DVs, p):
        """ get rt and accuracy of go and stop process for simulated
        conditions generated from simulate_dpm
        """
        ssd, nssd, nss, nss_per, ssd_ix = self.sim.ssd_info
        nl, ntot = self.sim.nlevels, self.sim.ntot
        gdec = self.sim.go_resp(DVg, p['a'])
        # if dpm, simply ss_resp() uses 0
        # as boundary simply ignores sec. arg
        sdec = self.sim.ss_resp(DVs, p['a'])
        gort = self.sim.go_RT(p['tr'], gdec)
        ssrt = self.sim.ss_RT(ssd, sdec)
        ert = gort[:, :nss].reshape(nl, nssd, nss_per);
        gort[np.isnan(gort)] = self.fitparams['tb'] + self.sim.dt
        ssrt[np.isnan(ssrt)] = self.fitparams['tb'] + self.sim.dt
        return pandaify_results(gort, ssrt, clmap=self.clmap, ssd=ssd)
