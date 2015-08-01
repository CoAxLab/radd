#!/usr/local/bin/env python
from __future__ import division
import os
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from radd.misc.messages import saygo
from radd.models import Simulator
from radd.fit import Optimizer
from radd.RADD import RADDCore

class Model(RADDCore):

      """ Main class for instantiating, fitting, and simulating models.
      Inherits from RADDCore parent class (see RADD.py).

      :: Arguments ::

            data (pandas DF):
                  data frame with columns 'idx', 'rt', 'acc', 'ttype', 'response',
                  <Condition Name> declared in depends_on values

            kind (str):
                  declares model type ['radd', 'irace', 'pro']

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

      """


      def __init__(self, data=pd.DataFrame, kind='radd', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, tb=None, weighted=True, scale_rts=False, *args, **kws):

            self.data=data
            self.weighted=weighted

            super(Model, self).__init__(data=self.data, inits=inits, fit_on=fit_on, depends_on=depends_on, niter=niter, fit_whole_model=fit_whole_model, kind=kind, tb=tb, scale_rts=scale_rts)

            self.prepare_fit()


      def optimize(self, save=True, savepth='./', ntrials=10000, ftol=1.e-4, xtol=1.e-4, maxfev=1000, niter=500, log_fits=True, disp=True, prob=np.array([.1, .3, .5, .7, .9])):
            """ Method to be used for accessing fitting methods in Optimizer class
            see Optimizer method optimize()
            """
            fp = self.set_fitparams(xtol=xtol, ftol=xtol, maxfev=maxfev, ntrials=ntrials, niter=niter, disp=disp, log_fits=log_fits, prob=prob, get_params=True)
            inits = dict(deepcopy(self.inits))
            self.opt = Optimizer(dframes=self.dframes, fitparams=fp, kind=self.kind, inits=inits, depends_on=self.depends_on, fit_on=self.fit_on, wts=self.wts, pc_map=self.pc_map)

            self.fits, self.fitinfo, self.popt = self.opt.optimize_model(save=save, savepth=savepth)
            self.residual = self.opt.residual
            self.simulator = self.opt.simulator


      def simulate(self):

            if not hasattr(self, 'simulator'):
                  theta=self.inits
                  self.set_fitparams()
                  self.simulator=Simulator(fitparams=self.fitparams, kind=self.kind, inits=theta, pc_map=self.pc_map)
            else:
                  theta=self.popt

            theta = self.simulator.vectorize_params(theta, as_dict=True)

            if 'radd' in self.kind:
                  dvg, dvs = self.simulator.simulate_radd(theta)
                  yhat = self.simulator.analyze_radd(dvg, dvs, theta)
            if 'pro' in self.kind:
                  dvg = self.simulator.simulate_pro(theta)
                  yhat = self.simulator.analyze_pro(dvg, theta)
            if 'irace' in self.kind:
                  dvg, dvs = self.simulator.simulate_irace(theta)
                  yhat = self.simulator.analyze_irace(dvg, dvs, theta)

            return yhat


      def prepare_fit(self):

            if 'trial_type' in self.data.columns:
                  self.data.rename(columns={'trial_type':'ttype'}, inplace=True)

            # if numeric, sort first then convert to string
            if not isinstance(self.labels[0], str):
                  ints = sorted([int(l*100) for l in self.labels])
                  self.labels = [str(intl) for intl in ints]
            else:
                  self.labels = sorted(self.labels)

            if 'pro' in self.kind:
                  if 'z' in self.inits.keys():
                        z=self.inits.pop('z')
                        self.inits['a']=self.inits['a']-z
                  if 'ssv' in self.inits.keys():
                        ssv=self.inits.pop('ssv')
            if 'x' in self.kind and 'xb' not in self.inits.keys():
                  self.inits['xb'] = 1

            params = sorted(self.inits.keys())
            self.pc_map = {}
            for d in self.depends_on.keys():
                  params.remove(d)
                  params_dep = ['_'.join([d, l]) for l in self.labels]
                  self.pc_map[d] = params_dep
                  params.extend(params_dep)
            qp_cols = self.__get_header__(params)
            self.__make_dataframes__(qp_cols)
            if self.weighted:
                  self.get_wts()
            else:
                  self.fwts=np.ones_like(self.flat_y.flatten())
                  self.wts=np.ones_like(self.avg_y.flatten())

            self.is_prepared=saygo(depends_on=self.depends_on, labels=self.labels, kind=self.kind, fit_on=self.fit_on)
