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
        self.track_subjects = False
        self.track_basins = False
        super(Model, self).__init__(data=data, inits=inits, fit_on=fit_on, depends_on=depends_on, kind=kind, dynamic=dynamic, quantiles=quantiles, weighted=weighted, ssd_method=ssd_method)

    def optimize(self, fit_flat=True, fit_cond=True, multiopt=True, inits_list=None, progress=True, plot_fits=True, saveplot=False, kde_quant_plots=True, keep_log=False):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        """
        popt = self.__check_inits__(self.inits)
        if multiopt and not hasattr(self, 'param_sets'):
            self.sample_param_sets()
        if progress:
            self.make_progress_bars()
        nfits = len(self.observed)
        for i in range(nfits):
            if self.track_subjects:
                self.pbars.update(name='idx', i=i, new_progress=self.idx[i])
            if fit_flat or self.is_flat:
                y, wts = self.iter_flat[i]
                self.set_fitparams(idx=i, y=y, wts=wts, nlevels=1, flat=True)
                finfo, popt, yhat = self.optimize_flat(inits_list=inits_list, progress=progress)
            if fit_cond and not self.is_flat:
                y, wts = self.iter_cond[i]
                self.set_fitparams(idx=i, y=y, wts=wts, nlevels=self.nlevels, flat=False)
                finfo, popt, yhat = self.optimize_conditional(popt, multiopt)
            self.assess_fit(finfo, popt, yhat, keep_log)
            if plot_fits:
                self.plot_model_fits(y=y, yhat=yhat, kde_quant=kde_quant_plots, save=saveplot)
        if progress:
            self.pbars.clear()

    def optimize_flat(self, inits_list=None, progress=False):
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
        globalmin = 1.; pbars=None;
        if inits_list is None:
            inits_list, globalmin = self.filter_param_sets()
        if self.track_basins:
            pbars = self.pbars.reset_bar('glb_basin', init_state=globalmin)
        # Global Optimization w/ Basinhopping (+TNC)
        p = self.opt.hop_around(inits_list=inits_list, pbars=self.pbars)
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
            # Pretune Conditional Parameters
            p, fmin = self.opt.run_basinhopping(p, is_flat=False)
        # Final Simplex Optimization
        finfo, popt, yhat = self.opt.gradient_descent(inits=p, is_flat=False)
        return finfo, popt, yhat

    def assess_fit(self, finfo, popt, yhat, keep_log=False):
        """ wrapper for analyze.assess_fit calculates and stores
        rchi, AIC, BIC and other fit statistics
        """
        fp = dict(deepcopy(self.fitparams))
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
            print('self.finfo, self.popt, and self.yhat still accessible from last fit')

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

    def plot_model_fits(self, y=None, yhat=None, fitparams=None, kde_quant=True, save=False):
        """ wrapper for radd.tools.vis.plot_model_fits """
        if fitparams is None:
            fitparams=self.fitparams
        if y is None:
            y = fitparams['y']
        if yhat is None:
            if hasattr(self, 'yhat'):
                yhat = self.yhat
            else:
                yhat = deepcopy(y)
                print("model is unoptimized and no yhat array provided")
                print("plotting with yhat as copy of y")
        plot_model_fits(y, yhat, fitparams, kde_quant=kde_quant, save=save)

    def make_progress_bars(self):
        """ initialize progress bars to track fit progress (subject fits,
        init optimization, etc)
        """
        from radd.tools.utils import NestedProgress
        n = self.basinparams['ninits']
        self.pbars = NestedProgress(name='glb_basin', n=n, title='Global Basin')
        self.pbars.add_bar(name='lcl_basin', bartype='infobar', title='Current Basin', color='red')
        self.track_basins=True
        if self.fit_on=='subjects':
            self.track_subjects = True
            self.pbars.add_bar(name='idx', n=self.nidx, title='Subject Fits', color='green')
        self.fitparams['disp']=False
        self.basinparams['disp']=False

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
