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
    """

    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', depends_on={None: None}, weighted=True, dynamic='hyp', percentiles=np.array([.1, .3, .5, .7, .9])):

        super(Model, self).__init__(data=data, inits=inits, fit_on=fit_on, depends_on=depends_on, kind=kind, dynamic=dynamic, percentiles=percentiles, weighted=weighted)



    def optimize(self, inits=None, fit_flat=True, fit_cond=True, multiopt=True):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        """

        if not inits:
            inits = self.inits

        self.multiopt = multiopt
        if inits is None:
            inits = self.inits
        self.__check_inits__(inits=inits)
        inits = dict(deepcopy(self.inits))

        # make sure inits only contains subsets of these params
        pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
        pfit = list(set(list(self.inits)).intersection(pnames))
        p_flat = dict(deepcopy({pk: self.inits[pk] for pk in pfit}))

        self.yhat_list, self.finfo_list, self.popt_list = [], [], []
        self.yhat_flat_list, self.finfo_flat_list, self.popt_flat_list = [], [], []

        for idx in range(len(self.observed)):

            if fit_flat:
                flat_y = self.observed_flat[idx]
                flat_wts = self.flat_cost_wts[idx]
                self.set_fitparams(idx=idx, y=flat_y, wts=flat_wts)
                yhat_flat, finfo_flat, popt_flat = self.optimize_flat(p=p_flat)

                self.yhat_flat_list.append(yhat_flat)
                self.finfo_flat_list.append(finfo_flat)
                self.popt_flat_list.append(dict(deepcopy(popt_flat)))
                p_flat = dict(deepcopy(popt_flat))

            if fit_cond:
                y = self.observed[idx]
                wts = self.cost_wts[idx]
                self.set_fitparams(idx=idx, y=y, wts=wts)
                yhat, finfo, popt = self.optimize_conditional(p=p_flat)

                # optimize params iterating over subjects/bootstraps
                self.yhat_list.append(deepcopy(yhat))
                self.finfo_list.append(deepcopy(finfo))
                self.popt_list.append(dict(deepcopy(popt)))


    def optimize_flat(self, p):
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
        if self.multiopt:
            # Global Optimization w/ Basinhopping (+TNC)
            p = self.opt.hop_around(p)
            print('Finished Hopping Around')

        # Flat Simplex Optimization of Parameters at Global Minimum
        yhat_flat, finfo_flat, popt_flat = self.opt.gradient_descent(inits=p, is_flat=True)
        return yhat_flat, finfo_flat, popt_flat


    def optimize_conditional(self, p):
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

        if self.multiopt:
            # Pretune Conditional Parameters
            p, funcmin = self.opt.run_basinhopping(p, is_flat=False)

        # Final Simplex Optimization
        yhat, finfo, popt = self.opt.gradient_descent(inits=p, is_flat=False)
        return yhat, finfo, popt


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
        if not hasattr(self, 'simulator'):
            self.make_simulator()
        if p is None:
            if hasattr(self, 'popt'):
                p = self.popt
            else:
                p = self.inits
        p = self.simulator.vectorize_params(p)
        out = self.simulator.sim_fx(p, analyze=analyze)
        if not analyze and not return_traces:
            out = self.simulator.predict_data(out, p)
        return out
