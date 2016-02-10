#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.tools.messages import saygo
from radd import fit, models, analyze
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
    """

    def __init__(self, data=pd.DataFrame, kind='dpm', inits=None, fit_on='average', depends_on=None, niter=50, fit_noise=False, fit_whole_model=True, tb=None, weighted=True, pro_ss=False, dynamic='hyp', tol=1.e-10, split=50, verbose=False, include_zero_rts=False, *args, **kws):

        self.data = data
        self.weighted = weighted
        self.verbose = verbose

        super(Model, self).__init__(data=self.data, inits=inits, fit_on=fit_on, depends_on=depends_on, niter=niter, fit_whole_model=fit_whole_model, kind=kind, tb=tb, fit_noise=fit_noise, pro_ss=pro_ss, split=split, dynamic=dynamic)

        self.prepare_fit()


    def make_optimizer(self, inits=None, ntrials=10000, tol=1.e-5, maxfev=5000, disp=True, bdisp=False, multiopt=True, nrand_inits=2, niter=40, interval=10, stepsize=.05, nsuccess=20, method='TNC', btol=1.e-3, maxiter=20):
        """ init Optimizer class as Model attr
        """
        fp = self.set_fitparams(tol=tol, maxfev=maxfev, ntrials=ntrials, niter=niter, disp=disp, get_params=True)
        bp = self.set_basinparams(btol=btol, interval=interval, niter=niter, maxiter=maxiter, method=method,                                  nsuccess=nsuccess, stepsize=stepsize, nrand_inits=nrand_inits, bdisp=bdisp, get_params=True)

        if inits is None:
            inits = self.inits
        self.__check_inits__(inits=inits)
        inits = dict(deepcopy(self.inits))

        self.opt = fit.Optimizer(dframes=self.dframes, fitparams=fp, basinparams=bp, kind=self.kind, inits=inits, depends_on=self.depends_on, fit_on=self.fit_on, wts=self.avg_wts, pc_map=self.pc_map,  multiopt=multiopt)


    def optimize(self, inits=None, y=None, ntrials=10000, tol=1.e-5, maxfev=5000, disp=True, bdisp=False, nrand_inits=2, niter=40, interval=10, stepsize=.05, nsuccess=20, method='TNC', btol=1.e-3, maxiter=20, multiopt=True, stage='full', save=True, savepth='./'):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        """

        self.make_optimizer(inits=inits, ntrials=ntrials, tol=tol, maxfev=maxfev, disp=disp, bdisp=bdisp, nrand_inits=nrand_inits, niter=niter, method=method, multiopt=multiopt, btol=btol, interval=interval, stepsize=stepsize, nsuccess=nsuccess, maxiter=maxiter)

        if stage == 'flat':
            y, fi, p = self.opt.optimize_flat(p0=inits, y=y, random_init=multiopt)
            self.flat_fits, self.flat_fitinfo, self.flat_popt = y, fi, p
        elif stage == 'cond':
            if inits is None and hasattr(self, 'flat_popt'):
                inits = self.flat_popt
            y, fi, p = self.opt.optimize_conditional(p=inits, y=y, precond=multiopt)
            self.fits, self.fitinfo, self.popt = y, fi, p
        else:
            y, fi, p = self.opt.optimize_model(save=save, savepth=savepth)
            self.fits, self.fitinfo, self.popt = y, fi, p

        # get Simulator object used by
        # Optimizer to fit the model
        self.simulator = self.opt.simulator


    def make_simulator(self, p=None):
        """ initializes Simulator object as Model attr
        using popt or inits if model is not optimized
        """
        if p is None:
            if hasattr(self, 'popt'):
                p = self.popt
            else:
                p = self.inits
        self.simulator = models.Simulator(fitparams=self.fitparams, kind=self.kind, inits=p, pc_map=self.pc_map)


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


    def prepare_fit(self):
        """ performs model setup and initiates dataframes. Automatically run when Model object is initialized
            *   pc_map is a dict containing parameter names as keys with values
                    corresponding to the names given to that parameter in Parameters object
                    (see optmize.Optimizer).
            *   Parameters (p[pkey]=pval) that are constant across conditions are broadcast as [pval]*n.
                    Conditional parameters are treated as arrays with distinct values [V1, V2...Vn], one for
                    each condition.

            pc_map (dict):    keys: conditional parameter names (i.e. 'v')
                              values: keys + condition names ('v_bsl, v_pnl')

            |<--- PARAMETERS OBJECT [LMFIT] <-------- [IN]
            |
            |---> p = {'v_bsl': V1, 'v_pnl': V2...} --->|
                                                        |
            |<--- pc_map = {'v':['v_bsl', 'v_pnl']} <---|
            |
            |---> p['v'] = array([V1, V2]) -------> [OUT]
        """

        if not isinstance(self.labels[0], str):
            ints = sorted([int(l * 100) for l in self.labels])
            self.labels = [str(intl) for intl in ints]
        else:
            self.labels = sorted(self.labels)

        params = sorted(self.inits.keys())
        cond_inits = lambda a, b: pd.Series(dict(zip(a, b)))
        self.pc_map = {}
        for d in self.depends_on.keys():
            params.remove(d)
            self.pc_map[d] = ['_'.join([d, l]) for l in self.labels]
            params.extend(self.pc_map[d])

        qp_cols = self.__get_header__(params, cond=self.cond)
        # MAKE DATAFRAMES FOR OBSERVED DATA, POPT, MODEL PREDICTIONS
        self.__make_dataframes__(qp_cols)
        # CALCULATE WEIGHTS FOR COST FX
        if self.weighted:
            self.get_wts()
        else:
            # MAKE PSEUDO WEIGHTS
            self.flat_wts = np.ones_like(self.flat_y.flatten())
            self.avg_wts = np.ones_like(self.avg_y.flatten())
        if self.verbose:
            self.is_prepared = saygo(depends_on=self.depends_on, labels=self.labels, kind=self.kind, fit_on=self.fit_on, dynamic=self.dynamic)
        else:
            self.prepared = True

        self.set_fitparams()
