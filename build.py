#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from radd import fit, fit_flat
import seaborn as sns
import matplotlib.pyplot as plt


def rangl_data(data, cutoff=.650, kind='reactive', prob=np.array([.1, .3, .5, .7, .9])):

      if kind == 'reactive':
            gac = data.query('trial_type=="go"').acc.mean()
            sacc = data.query('trial_type=="stop"').groupby('ssd').mean()['acc'].values
            grt = data.query('trial_type=="go" & acc==1').rt.values
            ert = data.query('response==1 & acc==0').rt.values
            gq = mq(grt, prob=prob)
            eq = mq(ert, prob=prob)

            return np.hstack([gac, sacc, gq*10, eq*10]).astype(np.float32)

      elif kind=='proactive':
            godf = data.query('response==1')
            gotrials=godf[godf.rt<=rt_cutoff]
            pgo = data.response.mean()
            gp = pgo*prob
            gq = mq(gotrials.rt, prob=gp)
            gmu = gotrials.rt.mean()
            return np.hstack([gq*10, gp, gmu, pgo])


def resample_data(data, n=120, kind='reactive'):

      df=data.copy(); bootlist=list()
      if n==None: n=len(df)

      if kind=='reactive':
            for ssd, ssdf in df.groupby('ssd'):
                  boots = ssdf.reset_index(drop=True)
                  orig_ix = np.asarray(boots.index[:])
                  resampled_ix = rwr(orig_ix, get_index=True, n=n)
                  bootdf = ssdf.irow(resampled_ix)
                  bootlist.append(bootdf)
                  #concatenate and return all resampled conditions
                  return rangl_re(pd.concat(bootlist))

      else:
                  boots = df.reset_index(drop=True)
                  orig_ix = np.asarray(boots.index[:])
                  resampled_ix = rwr(orig_ix, get_index=True, n=n)
                  bootdf = df.irow(resampled_ix)
                  bootdf_list.append(bootdf)
                  return rangl_pro(pd.concat(bootdf_list), rt_cutoff=rt_cutoff)


class Model(object):

      def __init__(self, kind='reactive', model='radd', inits=None, data=pd.DataFrame, fit_on='subjects', depends_on=None, niter=50, cond=None, prepare=False, wls=True, *args, **kws):

            self.model = model
            self.inits = inits
            self.kind = kind
            self.data = data
            self.niter = niter
            self.wls=wls

            if depends_on is None:
                  self.is_flat=True
                  self.ncond=1
            else:
                  self.is_flat=False
                  self.depends_on = depends_on
                  self.depends = depends_on.keys()
                  self.cond = depends_on.values()[0]
                  self.labels = list(data[self.cond].unique())
                  self.ncond=len(self.labels)

            self.i = 0
            self.fit_on = fit_on
            self.isprepared=False

            self.delays = sorted(data.query('trial_type=="stop"').ssd.unique().astype(np.int))

            if self.fit_on=='bootstrap':
                  self.indx=range(niter)
                  self.ifx = resample_data
            elif self.fit_on=='subjects' or self.fit_on=='average':
                  self.indx=list(self.data.idx.unique())
                  self.ifx = rangl_data
            if prepare:
                  self.prepare_fit()
            #if self.depends_on == None:
            #	self.depends_on = {}

      def prepare_fit(self):

            if self.is_flat:
                  self.__prepare_flat_model__()
            else:
                  self.__prepare_indx_model__()

      def __prepare_flat_model__(self):

            print "preparing %s model to fit on %s data" % (self.kind, self.fit_on)

            data=self.data.copy(); delays = self.delays;
            datdf = data.groupby(['idx']).apply(rangl_data, kind=self.kind)
            self.dat = datdf.copy()

            qp_cols = ['Go'] + delays +['c5','c25','c50','c75','c95'] + ['e5','e25','e50','e75','e95']
            ixdf = pd.DataFrame({'idx': data.idx.unique(), 'flat': 'flat'}, columns=['idx', 'flat'])
            self.observed = pd.concat([ixdf, pd.DataFrame(data=np.vstack(datdf.values), columns=qp_cols)], axis=1)

            self.get_wts(self.wls)
            self.isprepared=True


      def __prepare_indx_model__(self):

            print "preparing %s model to fit on %s data" % (self.kind, self.fit_on)

            if self.inits is None:
                  self.load_default_inits()

            inits=self.inits
            data=self.data.copy()
            ncond=self.ncond
            delays = self.delays
            indx=self.indx
            depends_on = self.depends_on
            depends = self.depends
            cond = self.cond;
            labels = self.labels

            grouped = data.groupby(['idx', cond])
            datdf = grouped.apply(rangl_data, kind=self.kind)

            if self.fit_on=='bootstrap':
                  if hasattr(self, 'cond'):
                        boots = data.groupby([cond])
                  else:
                        boots = data
                  #CREATE ITERABLE OBJECT CONTAINING NITER of RESAMPLED DATA FOR FITTING
                  self.dat = np.vstack([boots.apply(self.ifx, kind=self.kind).values for i in indx])


            else:
                  #CREATE ITERABLE OBJECT CONTAINING ALL INDIVIDUAL IDX DATA FOR FITTING
                  self.dat = np.array([np.vstack(cset) for cset in datdf.unstack().values])

            indx_vals = np.sort(np.hstack(indx*ncond))
            cond_vals = np.array(labels*len(indx))

            params = sorted(self.inits.keys())
            for d in depends:
                  params.remove(d[0])
                  params.extend([d+str(i) for i in range(ncond)])

            self.infolabels = np.hstack([params,'nfev','chi','rchi','AIC','BIC','CNVRG'])
            qp_cols = ['Go'] + delays +['c5','c25','c50','c75','c95'] + ['e5','e25','e50','e75','e95']
            ixdf = pd.DataFrame({'idx': indx_vals, cond: cond_vals}, columns=['idx', cond])

            self.observed = pd.concat([ixdf, pd.DataFrame(data=np.vstack(datdf.values), columns=qp_cols)], axis=1)
            self.fits = pd.concat([ixdf, pd.DataFrame(np.zeros_like(np.vstack(datdf.values)), columns=qp_cols)], axis=1)
            self.popt = pd.DataFrame(columns=self.infolabels, index=indx)

            self.get_wts(self.wls)

            self.isprepared = True


      def optimize(self, save=False, savepth='./', live_update=True, log_fits=True, disp=True, xtol=1.e-3, ftol=1.e-3, maxfev=500, ntrials=2000, niter=500):

            ntrials, ftol, xtol, maxfev, niter = self.set_fitparams(xtol=xtol, ftol=xtol, maxfev=maxfev, ntrials=ntrials, niter=niter, get_params=True)

            if not self.isprepared:
                  self.prepare_fit()

            if self.is_flat:
                  self.y = self.dat.mean(axis=0)
                  self.finfo, self.fitp , self.yhat = fit_flat.optimize_theta_flat(self.y, inits=self.inits, wts=self.wts, log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)

                  return self.fitp, self.yhat

            elif self.fit_on=='average':
                  self.y = self.dat.mean(axis=0)
                  self.finfo, self.fitp , self.yhat = fit.optimize_theta(self.y, inits=self.inits, wts=self.wts, ncond=self.ncond, bias=self.depends, log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)

                  return self.fitp, self.yhat

            for i, y in enumerate(self.dat):

                  finfo, fitp , yhat = fit.optimize_theta(y, inits=self.inits, wts=self.wts, ncond=self.ncond, bias=self.depends, log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)

                  self.finfo = finfo; self.fitp=fitp; self.yhat=yhat
                  self.popt.iloc[:, 2:]=pd.Series({info: finfo[info] for info in self.infolabels})
                  self.fits.iloc[self.i: self.i+ncond, ncond:] = yhat
                  self.i+=self.ncond

                  if save and live_update:
                        self.fits.to_csv(savepth+model+"_fits.csv", index=False)
                        self.popt.to_csv(savepth+model+"_popt.csv", index=False)

            if save:
                  self.fits.to_csv(savepth+model+"_fits.csv", index=False)
                  self.popt.to_csv(savepth+model+"_popt.csv", index=False)
                  self.observed.to_csv(savepth+model+"_data.csv", index=False)


      def set_rt_cutoff(self, rt_cutoff=None):

            if rt_cutoff==None:
                  if self.kind=='reactive':
                        self.rt_cutoff = .650
                  elif self.kind=='proactive':
                        self.rt_cutoff = .54502
                  else:
                        self.rt_cutoff=self.data.query('response==1').rt.max()
            else:
                  self.rt_cutoff=rt_cutoff


      def set_fitparams(self, ntrials=2000, ftol=1.e-3, xtol=1.e-3, maxfev=500, niter=500, get_params=False):

            self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'ftol':ftol, 'xtol':xtol, 'niter':niter}

            if self.fit_on=='bootstrap':
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

            if self.kind=='reactive':
                  self.gopt, self.ghat = fitre.fit_reactive_model(y, inits=inits, ntrials=ntrials, model=self.model, depends=['xx'], maxfev=maxfev, ftol=ftol, xtol=xtol, all_params=1)
            elif self.kind=='proactive':
                  self.gopt, self.ghat = fitpro.fit_proactive_model(y, inits=inits, ntrials=ntrials, model=self.model, depends=['xx'], maxfev=maxfev, ftol=ftol, xtol=xtol, all_params=1)

      def load_default_inits(self):

            self.inits = {'a': 0.44, 'ssv': 0.947, 'tr': 0.3049, 'v': 1.1224, 'z': 0.15}

      def get_wts(self, fit_wls=True):

            """
            need to set up for different conditions
            but this should be the final weighting policy

            wtc (weights correct rt quantiles)
            *P(Cor|NoSSD) * sd(.5cQ, ... .95cQ)

            wte (weights error rt quantiles)
            *P(Err|SSDall) * sd(.5eQ, ... .95eQ)
            """

            ncond = self.ncond#len(self.data[self.cond].unique())

            if fit_wls and self.is_flat:
                  sd = self.observed.std()

                  pc = self.data.query('trial_type=="go"').response.mean()
                  pe = self.data.query('trial_type=="stop"').response.mean()

                  sdc = sd.loc['c5':'c95'].values
                  sde = sd.loc['e5':'e95'].values

                  wtc = (pc*(sdc.min(axis=0)/sdc.T)).T
                  wte = (pe*(sde.min(axis=0)/sde.T)).T

                  self.wts = np.array([wtc, wte])

            elif fit_wls and not self.is_flat:

                  sd = self.observed.groupby(self.cond).std()

                  pc = self.data.query('trial_type=="go"').groupby(self.cond).response.mean().values
                  pe = self.data.query('trial_type=="stop"').groupby(self.cond).response.mean().values

                  sdc = sd.loc[:,'c5':'c95'].values
                  sde = sd.loc[:,'e5':'e95'].values

                  wtc = (pc*(sdc.min(axis=1)/sdc.T)).T
                  wte = (pe*(sde.min(axis=1)/sde.T)).T

                  self.wts = np.array([wtc, wte]).reshape(10,ncond).T


            else:

                  self.wts = np.ones((2,10))
