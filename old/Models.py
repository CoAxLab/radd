#!/usr/local/bin/env python
from __future__ import division
import os
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from scipy import optimize
from scipy.io import loadmat
from radd import fitre, fitpro, utils
from numba.decorators import jit
import seaborn as sns
import matplotlib.pyplot as plt

rpal = lambda nc: sns.blend_palette(['#e88379', '#9e261b'], n_colors=nc)
bpal = lambda nc: sns.blend_palette(['#81aedb', '#2a6095'], n_colors=nc)
gpal = lambda nc: sns.blend_palette(['#65b88f', '#2c724f'], n_colors=nc)
ppal = lambda nc: sns.blend_palette(['#848bb6', '#4c527f'], n_colors=nc)


class Model(object):

      def __init__(self, kind='radd', model='radd', inits={}, data=pd.DataFrame, depends_on={'xx':'XX'}, fit='bootstrap', niter=50, *args, **kwargs):

            self.model = model
            self.inits = inits
            self.kind = kind
            self.data = data
            self.niter = niter
            self.depends_on = depends_on
            self.depends = self.depends_on.keys()
            self.cond = self.depends_on.values()[0]
            self.fitparams = None
            self.live_update = True
            self.prob=([.1, .3, .5, .7, .9])
            self.i = 0
            self.fit = fit
            if self.fit=='bootstrap':
                  self.indx=range(niter)
            elif self.fit=='subjects':
                  self.indx=list(self.data.idx.unique())


      def set_fitparams(self, ntrials=2000, ftol=1.e-3, xtol=1.e-3, maxfev=500, niter=500, get_params=False):

            self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'ftol':ftol, 'xtol':xtol, 'niter':niter}

            if self.fit=='bootstrap':
                  self.indx=range(self.fitparams['niter'])

            fitp = self.fitparams

            if get_params:
                  return fitp['ntrials'], fitp['ftol'], fitp['xtol'], fitp['maxfev'], fitp['niter']



      def global_opt(self, xtol=1.e-3, ftol=1.e-3, maxfev=500, ntrials=2000, niter=500):

            if not self.isprepared:
                  self.prepare_fit()

            ntrials, ftol, xtol, maxfev, niter = self.set_fitparams(xtol=xtol, ftol=xtol, maxfev=maxfev, ntrials=ntrials, niter=niter, get_params=True)
            inits = self.inits
            y = self.dat.mean(axis=0)

            if self.kind=='radd':
                  self.gopt, self.ghat = fitre.fit_reactive_model(y, inits=inits, ntrials=ntrials, model=self.model, depends=['xx'], maxfev=maxfev, ftol=ftol, xtol=xtol, all_params=1)
            elif self.kind=='pro':
                  self.gopt, self.ghat = fitpro.fit_proactive_model(y, inits=inits, ntrials=ntrials, model=self.model, depends=['xx'], maxfev=maxfev, ftol=ftol, xtol=xtol, all_params=1)



      def prepare_fit(self):

            print "preparing %s model using %s method" % (self.kind, self.fit)

            inits=self.inits; data=self.data; cond=self.cond; indx = self.indx
            labels = data[cond].unique();


            delays = sorted(data.query('ttype=="stop"').ssd.unique().astype(np.int))

            if self.fit=='bootstrap':
                  if self.kind=='radd':
                        ifx = utils.resample_reactive
                  elif self.kind=='pro':
                        ifx = utils.resample_proactive

                  #CREATE ITERABLE OBJECT CONTAINING NITER of RESAMPLED DATA FOR FITTING
                  self.dat = np.vstack([[ifx(cdf) for i in indx] for c, cdf in self.data.groupby(self.cond)])

            elif self.fit=='subjects':
                  if self.kind=='radd':
                        ifx = utils.rangl_re
                  elif self.kind=='pro':
                        ifx = utils.rangl_pro

                  #CREATE ITERABLE OBJECT CONTAINING ALL INDIVIDUAL IDX DATA FOR FITTING
                  self.dat = np.vstack([[ifx(idxdf) for idx, idxdf in cdf.groupby('idx')] for c, cdf in self.data.groupby(self.cond)]).astype(np.float32)

            popt_cols = np.hstack(['chi', inits.keys()])
            qp_cols = ['Go'] + delays +['c5','c25','c50','c75','c95'] + ['e5','e25','e50','e75','e95']
            ixdf = pd.concat([pd.DataFrame({'indx': indx, 'cond':c}, columns=['indx', 'cond']) for c in labels]).reset_index(drop=True)

            self.observed = pd.concat([ixdf, pd.DataFrame(self.dat, columns=qp_cols)], axis=1)
            self.fits = pd.concat([ixdf, pd.DataFrame(np.zeros_like(self.dat), columns=qp_cols)], axis=1)
            self.popt = pd.concat([ixdf, pd.DataFrame(columns=popt_cols, index=ixdf.index)], axis=1)
            self.isprepared = True


      def run_model(self, save=False, savepth='./', live_update=True, all_params=0, disp=False, xtol=1.e-3, ftol=1.e-3, maxfev=500, ntrials=2000, niter=500, fit_global=False, **kwargs):

            if "depends_on" in kwargs.keys():
                  self.depends_on = kwargs['depends_on']
                  self.depends = self.depends_on.keys()
                  self.cond = self.depends_on.values()[0]

            inits = self.inits; model=self.model; depends=self.depends;
            ntrials, ftol, xtol, maxfev, niter = self.set_fitparams(xtol=xtol, ftol=xtol, maxfev=maxfev, ntrials=ntrials, niter=niter, get_params=True)

            if not self.isprepared:
                  # initialize data & fit storage
                  self.prepare_fit()

            if fit_global:
                  global_opt()

            if fit_wls:
                    wtgc, wtsc, wtgq, wteq = self.get_wts(m.observed)
                    wts = [wtgc, wtsc, wtgq, wteq]
            else:
                    wts = None

            for i, y in enumerate(self.dat):

                  if self.kind=='radd':
                        params, yhat = fitre.fit_reactive_model(y, inits=inits, ntrials=ntrials, wts=wts, model=model, depends=depends, maxfev=maxfev, ftol=ftol, xtol=xtol, all_params=all_params, disp=disp)

                  elif self.kind=='pro':
                        inits['pGo']=cdf.pGo.mean()
                        params, yhat = fitpro.fit_proactive_model(y, inits=inits, ntrials=ntrials, model=model, depends=depends, maxfev=maxfev, ftol=ftol, xtol=xtol, all_params=0, disp=disp)
                        #self.store_procost(indx[i], label, params, yhat)

                  self.popt.iloc[self.i, 2:] = params
                  self.fits.iloc[self.i, 2:] = yhat

                  if save and live_update:
                        self.fits.to_csv(savepth+model+"_fits.csv", index=False)
                        self.popt.to_csv(savepth+model+"_popt.csv", index=False)

            if save:
                  self.fits.to_csv(savepth+model+"_fits.csv", index=False)
                  self.popt.to_csv(savepth+model+"_popt.csv", index=False)
                  self.observed.to_csv(savepth+model+"_data.csv", index=False)


      def set_rt_cutoff(self, rt_cutoff=None):

            if rt_cutoff==None:
                  if self.kind=='radd':
                        self.rt_cutoff = .650
                  elif self.kind=='pro':
                        self.rt_cutoff = .54502
                  else:
                        self.rt_cutoff=self.data.query('response==1').rt.max()
            else:
                  self.rt_cutoff=rt_cutoff


      def get_wts(self):

              """
              need to set up for different conditions
              but this should be the final weighting policy

              wtc (weights correct rt quantiles)
                *P(Cor|NoSSD) * sd(.5cQ, ... .95cQ)

              wte (weights error rt quantiles)
                *P(Err|SSDall) * sd(.5eQ, ... .95eQ)


              """

              sd = self.observed.std()

              half = int(len(m.dat)/len(m.data['Cond'].unique()))

              pc = m.dat[:half].mean(axis=0)[0]
              pe = np.mean(1-m.dat[:half].mean(axis=0)[1:6])

              wtc = sd.loc['c5':'c95'].values*pc
              wte = sd.loc['e5':'e95'].values*pe

              self.wts = [wt.min()/wt for wt in [wtc, wte]]


      def plot_fits(self, bw=.1, plot_acc=False):


            sns.set_context('notebook', font_scale=1.6)

            gq = self.observed.loc[:, 'c5':'c95'].mean()
            eq = self.observed.loc[:, 'e5':'e95'].mean()
            fit_gq = self.fits.loc[:, 'c5':'c95'].mean()
            fit_eq = self.fits.loc[:, 'e5':'e95'].mean()

            if plot_acc:

                  f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))

                  gacc = self.observed['Go'].mean()
                  sacc = self.observed.loc[:, 200:400].mean()
                  fit_gacc = self.fits['Go'].mean()
                  fit_sacc = self.fits.loc[:, 200:400].mean()
            else:
                  f, ax1 = plt.subplots(1, figsize=(5,5))

            # Fit RT quantiles to KDE function in radd.utils
            quant_list = [gq, fit_gq, eq, fit_eq]
            kdefits = [utils.kde_fit_quantiles(q, bw=bw) for q in quant_list]

            sns.kdeplot(kdefits[0], cumulative=True, label='data gQP', linestyle='-', color=gpal(2)[0], ax=ax1, linewidth=3.5)
            sns.kdeplot(kdefits[1], cumulative=True, label='model gQP', linestyle='--', color=gpal(2)[1], ax=ax1, linewidth=3.5)
            sns.kdeplot(kdefits[2], cumulative=True, label='data eQP', linestyle='-', color=rpal(2)[0], ax=ax1, linewidth=3.5)
            sns.kdeplot(kdefits[3], cumulative=True, label='model eQP', linestyle='--', color=rpal(2)[1], ax=ax1, linewidth=3.5)

            ax1.set_xlim(4.3, 6.5)
            ax1.set_ylabel('P(RT<t)')
            ax1.set_xlabel('RT (s)')
            ax1.set_ylim(-.05, 1.05)
            ax1.set_xticklabels(ax1.get_xticks()*.1)

            if plot_acc:
                  # Plot observed and predicted stop curves
                  vis.scurves([sacc, fit_sacc], labels=['data Stop', 'model Stop'], colors=bpal(2), linestyles=['-','--'], ax=ax2)

            plt.tight_layout()
            sns.despine()
