#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.models import Simulator
from radd.tools import messages
from radd.CORE import RADDCore

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

    def optimize(self, fit_flat=True, fit_cond=True, multiopt=True):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        """
        nfits = len(self.observed)
        for i in range(nfits):
            popt = self.__check_inits__(self.inits)
            if fit_flat or self.is_flat:
                y, wts = self.iter_flat[i]
                self.set_fitparams(idx=i, y=y, wts=wts, nlevels=1, flat=True)
                finfo, popt, yhat = self.optimize_flat(popt, multiopt)
            if fit_cond and not self.is_flat:
                y, wts = self.iter_cond[i]
                self.set_fitparams(idx=i, y=y, wts=wts, nlevels=self.nlevels, flat=False)
                finfo, popt, yhat = self.optimize_conditional(popt, multiopt)
            self.assess_fit(finfo, popt, yhat)

    def optimize_flat(self, p, multiopt=True):
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
            p = self.opt.hop_around(p)
            print('Finished Hopping Around')
        # Flat Simplex Optimization of Parameters at Global Minimum
        finfo, popt, yhat = self.opt.gradient_descent(inits=p, is_flat=True)
        return finfo, popt, yhat

    def optimize_conditional(self, p, multiopt=True):
        """ optimizes full model to all conditions in data
        ::Arguments::
            p (dict):
                parameter dictionary to initalize model, if None uses init params
                passed by Model object
        ::Returns::
            yhat (array): model-predicted data array
            finfo (pd.Series): fit info (AIC, BIC, chi2, redchi, etc)
            popt (dict): optimized parameters dictionary
        """
        if multiopt:
            # Pretune Conditional Parameters
            p, funcmin = self.opt.run_basinhopping(p, is_flat=False)
        # Final Simplex Optimization
        finfo, popt, yhat = self.opt.gradient_descent(inits=p, is_flat=False)
        return finfo, popt, yhat

    def assess_fit(self, finfo, popt, yhat, log=True):
        """ wrapper for analyze.assess_fit calculates and stores
        rchi, AIC, BIC and other fit statistics
        """
        fp = dict(deepcopy(self.fitparams))
        fp['yhat'] = yhat.flatten()
        y = fp['y'].flatten()
        wts = fp['wts'].flatten()
        # fill finfo dict with goodness-of-fit info
        finfo['chi'] = np.sum(wts * (y - fp['yhat']) ** 2)
        finfo['ndata'] = len(fp['yhat'])
        finfo['df'] = finfo.ndata - finfo.nvary
        finfo['rchi'] = finfo.chi / finfo.df
        finfo['logp'] = finfo.ndata * np.log(finfo.rchi)
        finfo['AIC'] = finfo.logp + 2 * finfo.nvary
        finfo['BIC'] = finfo.logp + np.log(finfo.ndata * finfo.nvary)
        self.finfo = finfo
        self.popt = popt
        self.log_fit_info(finfo, popt, fp)

    def log_fit_info(self, finfo, popt, fitparams):
        """ write meta-information about latest fit
        to logfile (.txt) in working directory
        """
        fp = dict(deepcopy(fitparams))
        # lmfit-structured fit_report to write in log file
        param_report = self.opt.param_report
        # log all fit and meta information in working directory
        messages.logger(param_report, finfo=finfo, popt=popt, fitparams=fp, kind=self.kind)

    def make_simulator(self, fitparams=None, p=None):
        """ initializes Simulator object as Model attr
        using popt or inits if model is not optimized
        """
        if fitparams is None:
            fitparams = dict(deepcopy(self.fitparams))
        if p is None:
            p = dict(deepcopy(self.inits))
        self.simulator = Simulator(fitparams=fitparams, inits=p, kind=self.kind, pc_map=self.pc_map)
        if hasattr(self, 'opt'):
            self.opt.simulator = self.simulator
            self.opt.fitparams = fitparams

    def simulate(self, p=None, analyze=True, return_traces=False):
        """ simulate yhat vector using popt or inits
        if model is not optimized
        :: Arguments ::
          analyze (bool):
            if True (default) returns yhat vector
            if False, returns decision traces
        :: Returns ::
          out (array):
            1d array if analyze is True
            ndarray of decision traces if False
        """
        if p is None:
            if hasattr(self, 'popt'):
                p = self.popt
            else:
                p = self.inits
        if not hasattr(self.opt, 'simulator'):
            simulator = Simulator(fitparams=self.fitparams,
                inits=p, kind=self.kind, pc_map=self.pc_map)
        else:
            simulator = self.opt.simulator
        out = self.simulator.sim_fx(p, analyze=analyze)
        if not analyze:
            out = self.simulator.predict_data(out, p)
        return out
