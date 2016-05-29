#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.models import Simulator
from radd.fit import Optimizer
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
        depends_on (dict):
            set parameter dependencies on task conditions
            (ex. depends_on={'v': 'Condition'})
        fit_on (str):
            set if model fits 'average', 'subjects', 'bootstrap' data
        fit_whole_model (bool):
            fit model holding all fixed except depends_on keys
            or fit model with all free before fitting depends_on keys
        tb (float):
            timeboundary: time alloted in task for making a response
        dynamic (str):
            set dynamic bias signal to follow an exponential or hyperbolic
            form when fitting models with 'x' included in <kind> attr
        hyp_effect_dir (str):
            up or down: apriori hypothesized relationship between key:value pairs
            in depends_on dict
    """

    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', depends_on=None, niter=50, fit_noise=False, fit_whole_model=True, tb=None, weighted=True, dynamic='hyp', tol=1.e-10, percentiles=np.array([.1, .3, .5, .7, .9]), verbose=False, hyp_effect_dir=None, *args, **kws):

        self.data = data
        self.weighted = weighted
        self.verbose = verbose

        super(Model, self).__init__(data=self.data, inits=inits, fit_on=fit_on, depends_on=depends_on, niter=niter, fit_whole_model=fit_whole_model, kind=kind, tb=tb, fit_noise=fit_noise, dynamic=dynamic, hyp_effect_dir=hyp_effect_dir, percentiles=percentiles)

        self.__prepare_fit__()



    def make_optimizer(self, inits=None, ntrials=10000, tol=1.e-5, maxfev=5000, disp=True, bdisp=False, multiopt=True, nrand_inits=2, niter=40, interval=10, stepsize=.05, nsuccess=20, method='nelder', bmethod='TNC', btol=1.e-3, maxiter=20):
        """ init Optimizer class as Model attr
        """

        self.multiopt = multiopt

        fp = self.set_fitparams(tol=tol, maxfev=maxfev, ntrials=ntrials, niter=niter, disp=disp, get_params=True, method=method)
        bp = self.set_basinparams(btol=btol, interval=interval, niter=niter, maxiter=maxiter, method=bmethod, nsuccess=nsuccess, stepsize=stepsize, nrand_inits=nrand_inits, bdisp=bdisp, get_params=True)

        if inits is None:
            inits = self.inits
        self.__check_inits__(inits=inits)
        inits = dict(deepcopy(self.inits))

        self.opt = Optimizer(fitparams=fp, basinparams=bp, kind=self.kind, inits=inits, depends_on=self.depends_on, pc_map=self.pc_map)


    def optimize(self, inits=None, ntrials=10000, tol=1.e-5, maxfev=5000, disp=True, bdisp=False, multiopt=True, nrand_inits=2, niter=40, interval=10, stepsize=.05, nsuccess=20, method='nelder', bmethod='TNC', btol=1.e-3, maxiter=20, fit_flat=True):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        """

        self.make_optimizer(inits=inits, ntrials=ntrials, tol=tol, maxfev=maxfev, disp=disp, bdisp=bdisp, nrand_inits=nrand_inits, niter=niter, method=method, multiopt=multiopt, btol=btol, interval=interval, stepsize=stepsize, nsuccess=nsuccess, maxiter=maxiter)

        # make sure inits only contains subsets of these params
        pnames = ['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
        pfit = list(set(self.inits.keys()).intersection(pnames))
        p_flat = dict(deepcopy({pk: self.inits[pk] for pk in pfit}))

        self.yhat_list, self.finfo_list, self.popt_list = [], [], []
        self.yhat_flat_list, self.finfo_flat_list, self.popt_flat_list = [], [], []

        for idx in xrange(len(self.observed)):

            if fit_flat:
                flat_y = self.observed_flat[idx]
                flat_wts = self.flat_cost_wts[idx]
                fp_flat = self.set_fitparams(idx=idx, y=flat_y, wts=flat_wts, nlevels=1, get_params=True)
                yhat_flat, finfo_flat, popt_flat = self.optimize_flat(fp_flat, p=p_flat)
                self.yhat_flat_list.append(yhat_flat)
                self.finfo_flat_list.append(finfo_flat)
                self.popt_flat_list.append(dict(deepcopy(popt_flat)))

                pflat = dict(deepcopy(popt_flat))

            y = self.observed[idx]
            wts = self.cost_wts[idx]
            nlevels = self.fitparams['nlevels']
            fp_cond = self.set_fitparams(idx=idx, y=y, wts=wts, nlevels=nlevels, get_params=True)
            yhat, finfo, popt = self.optimize_conditional(fp_cond, p=popt_flat)
            #self.update_optimizer_fitparams(idx, is_flat=True)

            # optimize params iterating over subjects/bootstraps
            self.yhat_list.append(deepcopy(yhat))
            self.finfo_list.append(deepcopy(finfo))
            self.popt_list.append(dict(deepcopy(popt)))



    def optimize_flat(self, fitparams, p=None, random_init=True):
        """ optimizes flat model to data collapsing across all conditions

        ::Arguments::
          random_init (bool <False>):
                if True performs random initializaiton by sampling from parameter distributions and uses basinhopping alg. to find global minimum before entering stage 1 simplex
          p0 (dict):
                parameter dictionary to initalize model, if None uses init params
                passed by Model object
          y (ndarray):
                data to be fit; must be same shape as flat_wts vector
        """

        self.simulator = Simulator(fitparams=fitparams, kind=self.kind, inits=p, pc_map=self.pc_map)
        self.opt.simulator = self.simulator

        if random_init and self.multiopt:
            # hop_around --> basinhopping_full
            p = self.opt.hop_around(p)
        elif p is None:
            # p0: (Initials/Global Minimum)
            p = dict(deepcopy(self.inits))

        # p1: STAGE 1 (Initial Simplex)
        yhat_flat, finfo_flat, popt_flat = self.opt.gradient_descent(inits=p, is_flat=True)
        return yhat_flat, finfo_flat, popt_flat


    def optimize_conditional(self, fitparams, p=None, precond=True):
        """ optimizes full model to all conditions in data

        ::Arguments::
        <OPTIONAL>
              precond (bool <True>):
                    if True performs pre-conditionalizes params (p)  using
                    basinhopping alg. to find global minimum for each condition
                    before entering final simplex
              p (dict):
                    parameter dictionary, if None uses default init params passed by Model object
              y (ndarray):
                    data to be fit; must be same shape as avg_wts vector
        """

        if p is None:
            p = dict(deepcopy(self.inits))

        self.simulator = Simulator(fitparams=fitparams, kind=self.kind, inits=p, pc_map=self.pc_map)
        self.opt.simulator = self.simulator

        # STAGE 2: (Nudge/BasinHopping)
        p2 = self.__nudge_params__(p)
        if precond and self.multiopt:
            # pretune conditional parameters (1/time)
            p2 = self.opt.single_basin(p2)

        # STAGE 3: (Final Simplex)
        yhat, finfo, popt = self.opt.gradient_descent(inits=p2, is_flat=False)
        return yhat, finfo, popt


    def make_simulator(self, p=None, idx=0, is_flat=True):
        """ initializes Simulator object as Model attr
        using popt or inits if model is not optimized
        """

        fp = dict(deepcopy(self.fitparams))
        if p is None:
            if hasattr(self, 'popt'):
                p = self.popt
            else:
                p = self.inits

        self.simulator = Simulator(fitparams=fp, kind=self.kind, inits=p, pc_map=self.pc_map)


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
