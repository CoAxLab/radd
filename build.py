#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from radd.misc.messages import saygo
from radd.fit import Simulator
from radd.RADD import RADDCore


class Model(RADDCore):


      def __init__(self, data=pd.DataFrame, kind='reactive', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, style='RADD', weight_presp=True, *args, **kws):

            self.data=data
            self.weight_presp=weight_presp
            self.is_optimized=False

            super(Model, self).__init__(data=self.data, inits=inits, fit_on=fit_on, depends_on=depends_on, niter=niter, fit_whole_model=fit_whole_model, style=style, kind=kind)

            self.prepare_fit()


      def optimize(self, save=True, savepth='./', ntrials=10000, ftol=1.e-4, xtol=1.e-4, maxfev=1000, niter=500, log_fits=True, disp=True, prob=np.array([.1, .3, .5, .7, .9])):

            """
            Main link between Model class and fitting methods in Optimizer class
            see Optimizer method optimize()
            """

            self.opt = Optimizer(kind=self.kind, inits=self.inits, style=self.style, depends_on=self.depends_on, fit_on=self.fit_on, dataframes=self.df_dict, wts=self.wts, pc_map=self.pc_map)

            self.fits, self.fitinfo, self.popt = self.opt.optimize(save=save, savepth=savepth, log_fits=log_fits, disp=disp, xtol=xtol, ftol=ftol, maxfev=maxfev, ntrials=ntrials, niter=niter, prob=prob)

            self.is_optimized=True


      def simulate(self):

            try:
                  theta=self.popt
            except Exception:
                  theta=self.inits
                  self.set_fitparams()

            self.simulator = Simulator(fitparams=self.fitparams, kind=self.kind, style=self.style, inits=self.inits, pc_map=self.pc_map)
            theta = self.simulator.vectorize_params(theta, sim_info=False, as_dict=True)

            if self.kind=='reactive':
                  dvg, dvs = self.simulator.core_radd(theta)
                  yhat = self.simulator.analyze_reactive(dvg, dvs, theta)
            elif self.kind=='proactive':
                  dvg = self.simulator.pro_radd(theta)
                  yhat = self.simulator.analyze_proactive(dvg, theta)

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

            params = sorted(self.inits.keys())
            self.pc_map = {}
            for d in self.depends_on.keys():
                  params.remove(d)
                  params_dep = ['_'.join([d, l]) for l in self.labels]
                  self.pc_map[d] = params_dep
                  params.extend(params_dep)
            qp_cols = self.__get_header__(params)
            self.__make_dataframes__(qp_cols)
            self.get_wts()
            self.is_prepared=saygo(depends_on=self.depends_on, labels=self.labels, kind=self.kind, fit_on=self.fit_on)



class Optimizer(RADDCore):

      def __init__(self, dataframes=None, kind='reactive', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, style='RADD', method='nelder', pc_map=None, wts=None, *args, **kws):

            dfs=dict(deepcopy(dataframes))
            self.observed=dfs['observed']
            self.fits=dfs['fits']
            self.fitinfo=dfs['fitinfo']
            self.avg_y=dfs['avg_y']
            self.flat_y=dfs['flat_y']
            self.dat=dfs['dat']
            self.data = dfs['data']
            self.pc_map = pc_map
            self.wts=wts
            self.method=method

            super(Optimizer, self).__init__(kind=kind, data=self.data, fit_on=fit_on, depends_on=depends_on, inits=inits, style=style, fit_whole_model=fit_whole_model, niter=niter)


      def optimize(self, save=True, savepth='./', log_fits=True, disp=True, xtol=1.e-4, ftol=1.e-4, maxfev=1000, ntrials=10000, niter=500, prob=np.array([.1, .3, .5, .7, .9]), tb=None):

            self.set_fitparams(xtol=xtol, ftol=xtol, maxfev=maxfev, ntrials=ntrials, niter=niter, disp=disp, log_fits=log_fits, prob=prob)

            self.simulator = Simulator(fitparams=self.fitparams, kind=self.kind, style=self.style, inits=self.inits, method=self.method, pc_map=self.pc_map)

            if self.fit_on=='average':
                  self.yhat, self.fitinfo, self.popt = self.__opt_routine__(self.avg_y)
                  return self.yhat, self.fitinfo, self.popt
            else:
                  self.__indx_optimize__(save=save, savepth=savepth)
                  return self.fits, self.fitinfo, self.popt


      def __indx_optimize__(self, save=True, savepth='./'):

            ri=0; radv=self.ncond
            pcols=self.fitinfo.columns
            for i, y in enumerate(self.dat):
                  # rejoin grouped y vector, with mean eq
                  # this flattens the full vector, gets
                  # reshaped before weights are applied
                  yhat, finfo, popt = self.__opt_routine__(y)
                  self.fitinfo.iloc[i]=pd.Series({pc: finfo[pc] for pc in pcols})

                  if self.kind=='reactive':
                        self.fits.iloc[ri:ri+radv, radv:] = yhat
                        ri+=radv
                  else:
                        self.fits.iloc[i] = yhat
                  if save:
                        self.fits.to_csv(savepth+"fits.csv")
                        self.fitinfo.to_csv(savepth+"fitinfo.csv")

            self.popt=self.fitinfo.mean()


      def __opt_routine__(self, y):

            p = dict(deepcopy(self.inits))
            if self.fit_flat:
                  to_fit = [self.flat_y, y]
                  flat=[1, 0]
                  ncond=[1, self.ncond]
            else:
                  to_fit = [y]
                  flat=[0]
                  ncond=[self.ncond]

            for i, yy in enumerate(to_fit):
                  self.simulator.ncond=ncond[i]
                  yhat, finfo, popt = self.simulator.optimize_theta(y=yy, inits=p, flat=flat[i])
                  p = deepcopy(popt)

            return yhat, finfo, popt
