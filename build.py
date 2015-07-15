#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from radd import fit, fit_flat
from radd.misc import messages
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


def append_eq(yy):

      return np.hstack([np.append(yi, yy[-5:]) for yi in yy[:-5].reshape(2, 11)]).reshape(2,16)


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


      def __init__(self, kind='reactive', model='radd', inits=None, data=pd.DataFrame, fit_on='subjects', depends_on=None, niter=50, cond=None, prepare=False, *args, **kws):

            self.model = model
            self.inits = inits
            self.kind = kind
            self.data = data
            self.niter = niter

            if depends_on is None:
                  self.is_flat=True
                  self.ncond=1
                  grouped=self.data.groupby('idx')
                  self.datdf = grouped.apply(rangl_data, kind=self.kind)
            else:
                  self.is_flat=False
                  self.depends_on = depends_on
                  self.depends = depends_on.keys()
                  self.cond = depends_on.values()[0]
                  self.labels = list(data[self.cond].unique())
                  self.ncond=len(self.labels)
                  grouped = data.groupby(['idx', self.cond])
                  self.datdf = grouped.apply(rangl_data, kind=self.kind)

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


      def prepared_message(self):


            if self.is_flat:
                  strings = (self.fit_on, self.kind)

                  print "model is prepared to fit on %s %s data, with all parameters free" % strings

            else:
                  pdeps = self.depends_on.keys()
                  deplist = []
                  if 'a' in pdeps:
                        deplist.append('Boundary Height')
                  if 'tr' in pdeps:
                        deplist.append('Onset Time')
                  if 'v' in pdeps:
                        deplist.append('Drift-Rate')

                  if len(self.depends_on.keys())>1:
                        pdep = ' and '.join(deplist)
                  else:
                        pdep = deplist[0]

                  dep = self.depends_on.values()[0]
                  lbls = ', '.join(self.labels)
                  msg = messages.get_one()
                  strings = (self.fit_on, self.kind, pdep, dep, lbls, msg)

                  print """
                  Model is prepared to fit on %s %s data,
                  allowing %s to vary across
                  levels of %s (%s)  \n\n
                  %s \n\n """ % strings

            self.isprepared=True


      def prepare_fit(self):

            if self.is_flat:
                  self.__prepare_flat_model__()
            else:
                  self.__prepare_indx_model__()



      def __prepare_flat_model__(self):

            data=self.data.copy(); delays = self.delays;
            self.dat = self.datdf.copy()

            qp_cols = ['Go'] + delays +['c5','c25','c50','c75','c95'] + ['e5','e25','e50','e75','e95']
            ixdf = pd.DataFrame({'idx': data.idx.unique(), 'flat': 'flat'}, columns=['idx', 'flat'])
            self.observed = pd.concat([ixdf, pd.DataFrame(data=np.vstack(datdf.values), columns=qp_cols)], axis=1)

            self.get_wts()
            self.prepared_message()


      def __prepare_indx_model__(self):

            if self.inits is None:
                  self.load_default_inits()

            inits=self.inits; data=self.data; ncond=self.ncond; kind=self.kind
            delays=self.delays; depends_on=self.depends_on; datdf=self.datdf
            cond=self.cond; labels=self.labels; indx=self.indx;

            if self.fit_on=='bootstrap':
                  if hasattr(self, 'cond'):
                        boots = data.groupby([cond])
                  else:
                        boots = data
                  #CREATE ITERABLE CONTAINING NITER of RESAMPLED DATA FOR FITTING
                  dat = np.vstack([boots.apply(self.ifx, kind=kind).values for i in indx])
            else:
                  #CREATE ITERABLE CONTAINING ALL INDIVIDUAL IDX DATA FOR FITTING
                  dat = np.array([np.vstack(cset) for cset in datdf.unstack().values])

            # separate [go acc, sc, cor quantiles] | error quantiles
            # and average error quantile est. across conditions
            self.error_quantiles = np.vstack(dat[:,:,-5:]).mean(axis=0)
            self.dat = dat[:,:,:-5]

            indx_vals = np.sort(np.hstack(indx*ncond))
            cond_vals = np.array(labels*len(indx))

            params = sorted(inits.keys())
            self.pc_map = {}
            for d in depends_on.keys():
                  params.remove(d)
                  params_dep = ['_'.join([d, l]) for l in labels]
                  self.pc_map[d] = params_dep
                  params.extend(params_dep)

            self.infolabels = np.hstack([params,'nfev','chi','rchi','AIC','BIC','CNVRG'])
            qp_cols = ['Go'] + delays +['c5','c25','c50','c75','c95'] + ['e5','e25','e50','e75','e95']
            ixdf = pd.DataFrame({'idx': indx_vals, cond: cond_vals}, columns=['idx', cond])

            self.observed = pd.concat([ixdf, pd.DataFrame(data=np.vstack(datdf.values), columns=qp_cols)], axis=1)
            self.fits = pd.concat([ixdf, pd.DataFrame(np.zeros_like(np.vstack(datdf.values)), columns=qp_cols)], axis=1)
            self.popt = pd.DataFrame(columns=self.infolabels, index=indx)

            self.get_wts()
            self.prepared_message()


      def optimize(self, save=False, savepth='./', live_update=True, log_fits=True, disp=True, xtol=1.e-3, ftol=1.e-3, maxfev=500, ntrials=2000, niter=500):

            ntrials, ftol, xtol, maxfev, niter = self.set_fitparams(xtol=xtol, ftol=xtol, maxfev=maxfev, ntrials=ntrials, niter=niter, get_params=True)

            if not self.isprepared:
                  self.prepare_fit()

            if self.is_flat:
                  self.__flat_optimize__(log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)

                  return self.fitp, self.yhat


            elif self.fit_on=='average':
                  self.__avg_optimize__(log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)

                  return self.fitp, np.append(self.yhat[:-5].reshape(self.ncond, 11), self.yhat[-5:])


            elif self.fit_on in ['subjects', 'bootstrap']:
                  self.__indx_optimize__(log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp, save=save, savepth=savepth, live_update=live_update, niter=niter)

                  return self.fitp, np.append(self.yhat[:-5].reshape(self.ncond, 11), self.yhat[-5:])


      def __flat_optimize__(self, log_fits, ntrials, maxfev, ftol, xtol, disp):

            self.y = self.dat.mean(axis=0)

            self.finfo, self.fitp , self.yhat = fit_flat.optimize_theta_flat(self.y, inits=self.inits, wts=self.wts, log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)



      def __avg_optimize__(self, log_fits, ntrials, maxfev, ftol, xtol, disp):

            y = np.append(self.dat.mean(axis=0), self.error_quantiles)

            self.finfo, self.fitp , self.yhat = fit.optimize_theta(y, inits=self.inits, wts=self.wts, ncond=self.ncond, pc_map=self.pc_map, log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)


      def __indx_optimize__(self, log_fits, ntrials, maxfev, ftol, xtol, disp, save, savepth, live_update, niter):

            for i, y in enumerate(self.dat):

                  # rejoin grouped y vector, with mean eq
                  # this flattens the full vector, gets
                  # reshaped before weights are applied
                  y = np.append(y, self.error_quantiles)

                  self.finfo, self.fitp , self.yhat = fit.optimize_theta(y, inits=self.inits, wts=self.wts, ncond=self.ncond, pc_map=self.pc_map, log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)

                  self.popt.iloc[:, 2:]=pd.Series({info: self.finfo[info] for info in self.infolabels})
                  self.fits.iloc[self.i: self.i+self.ncond, self.ncond:] = self.yhat
                  self.i+=self.ncond

                  if save and live_update:
                        self.fits.to_csv(savepth+model+"_fits.csv", index=False)
                        self.popt.to_csv(savepth+model+"_popt.csv", index=False)

      def append_eq(yy):

            return np.hstack([np.append(yi, yy[-5:]) for yi in yy[:-5].reshape(2, 11)]).reshape(2,16)


      def set_fitparams(self, ntrials=2000, ftol=1.e-3, xtol=1.e-3, maxfev=500, niter=500, get_params=False):

            self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'ftol':ftol, 'xtol':xtol, 'niter':niter}

            if self.fit_on=='bootstrap':
                  self.indx=range(self.fitparams['niter'])

            fitp = self.fitparams

            if get_params:
                  return fitp['ntrials'], fitp['ftol'], fitp['xtol'], fitp['maxfev'], fitp['niter']



      def load_default_inits(self):

            self.inits = {'a': 0.44, 'ssv': 0.947, 'tr': 0.3049, 'v': 1.1224, 'z': 0.15}


      def get_wts(self):

            """
            wtc: weights applied to correct rt quantiles in cost f(x)

                  * P(R | No SSD)j * sdj(.5Qj, ... .95Qj)

            wte: weight applied to error rt quantiles in cost f(x)

                  * P(R | SSD) * sd(.5eQ, ... .95eQ)

            wts are calculated as a function of the observed variablitity in RT
            quantiles across subjects

            the variability of these statistics did not change significantly across
            conditions, thus the weights are caclulated collapsing across across
            all conditions

                  * P(R | SSD) * sd(.5eQ, ... .95eQ)
            """

            #if self.is_flat:
            sd = self.observed.std()
            pc = self.data.query('trial_type=="go"').response.mean()
            sdc = sd.loc['c5':'c95'].values
            wtc = (pc*(sdc.min(axis=0)/sdc.T)).T

            pe = self.data.query('trial_type=="stop"').response.mean()
            sde = sd.loc['e5':'e95'].values
            wte = (pe*(sde.min(axis=0)/sde.T)).T
            self.wts = np.append(wtc, wte)
            #else:
            #sd = self.observed.groupby(self.cond).std()
            #pc = self.data.query('trial_type=="go"').groupby(self.cond).response.mean().values
            #sdc = sd.loc[:,'c5':'c95'].values
            #wtc = (pc*(sdc.min(axis=1)/sdc.T)).T
            #sde = sd.loc[:,'e5':'e95'].values
            #wte = (pe*(sde.min(axis=1)/sde.T)).T
            #self.wts = np.append(self.wts, wte).reshape(10,ncond).T

      #def lets_see(self, y, yhat, plot_acc=True, )

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
