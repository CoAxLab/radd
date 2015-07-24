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


def rangl_data(data, re_cut=.650, pro_cut=.54502, kind='reactive', prob=np.array([.1, .3, .5, .7, .9])):

      if kind == 'reactive':
            gac = data.query('ttype=="go"').acc.mean()
            sacc = data.query('ttype=="stop"').groupby('ssd').mean()['acc'].values
            grt = data.query('ttype=="go" & acc==1').rt.values
            ert = data.query('response==1 & acc==0').rt.values
            gq = mq(grt, prob=prob)
            eq = mq(ert, prob=prob)

            return np.hstack([gac, sacc, gq*10, eq*10]).astype(np.float32)

      elif kind=='proactive':
            #godf = data.query('response==1')
            #gotrials=godf[godf.rt<=pro_cut]
            #gq = mq(gotrials.rt, prob=prob)
            return data.response.mean().astype(np.float32)

def rt_quantiles(data, cutoff=.54502, split='HiLo', prob=np.arange(0.1,1.0,0.2)):
      rtq = []
      godf = data[data.response==1]

      if split=='HiLo' and 'HiLo' not in godf.columns:
            godf['HiLo']=['x']
            godf[godf['pGo']<=.5, 'HiLo'] = 'Lo'
            godf[godf['pGo']>.5, 'HiLo'] = 'Hi'
      if split != None:
            splitdf = godf.groupby(split)
      else:
            splitdf = godf.copy()

      for c, df in splitdf:
            rts = df[df.rt<=cutoff].rt.values
            rtq.append(mq(rts, prob=prob)*10)

      return np.hstack(rtq)

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
            if 'trial_type' in self.data.columns:
                  self.data.rename(columns={'trial_type':'ttype'}, inplace=True)

            if depends_on is None:
                  self.is_flat=True
                  self.cond='flat'
                  self.labels=[self.cond]
                  self.ncond=1
            else:
                  self.is_flat = False
                  self.depends_on=depends_on
                  self.cond=depends_on.values()[0]
                  self.labels=data[self.cond].unique()
                  self.ncond=len(self.labels)


            self.i = 0
            self.fit_on = fit_on
            self.isprepared=False

            if self.kind=='reactive':
                  self.delays = sorted(data.query('ttype=="stop"').ssd.unique().astype(np.int))

            elif self.kind=='proactive':
                  self.pGo = np.arange(0, 1.2, .2)

            if self.fit_on=='bootstrap':
                  self.indx = range(niter)
                  self.ifx = resample_data
            else:
                  self.indx = list(self.data.idx.unique())
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

            if self.inits is None:
                  self.load_default_inits()

            indx = self.indx
            params = sorted(inits.keys())
            datdf = self.grouped.apply(rangl_data, kind=self.kind)

            self.dat = datdf.copy()
            self.ncond=1; self.labels=['flat']

            self.__make_dataframes__(datdf, indx, params)
            self.get_wts()

            self.prepared_message()


      def __prepare_indx_model__(self):

            if self.inits is None:
                  self.load_default_inits()

            data=self.data; indx=self.indx;
            lbls = self.labels

            # if numeric, sort first then convert to string
            if not isinstance(lbls[0], str):
                  self.labels = [str(intl) for intl in sorted([int(l*100) for l in lbls])]
            else:
                  self.labels = sorted(lbls)

            params = sorted(self.inits.keys())
            self.pc_map = {}
            for d in self.depends_on.keys():
                  params.remove(d)
                  params_dep = ['_'.join([d, l]) for l in self.labels]
                  self.pc_map[d] = params_dep
                  params.extend(params_dep)

            self.set_fitparams()
            self.__make_dataframes__(data, indx, params, self.labels, self.ncond)
            self.get_wts()

            self.prepared_message()


      def __make_dataframes__(self, data, indx, params, labels, ncond):

            prob = self.fitparams['prob']
            cond = self.cond;
            qp_cols = self.get_header(params, prob);

            ic_grp = data.groupby(['idx', cond])
            c_grp = data.groupby([cond])
            i_grp = data.groupby(['idx'])

            if self.fit_on=='bootstrap':
                  self.dat = np.vstack([i_grp.apply(self.ifx, kind=self.kind).values for i in indx]).unstack()
            else:
                  datdf = ic_grp.apply(rangl_data, kind=self.kind).unstack()

            if self.kind=='proactive':
                  rtq =  i_grp.apply(rt_quantiles)
                  rtdat = pd.DataFrame(np.vstack(rtq.values), columns=qp_cols[-10:], index=indx)
                  rtdat[rtdat<1] = np.nan

                  self.observed = pd.concat([datdf, rtdat], axis=1)
                  self.observed.columns = qp_cols
                  self.fits = pd.DataFrame(np.zeros_like(self.observed), columns=qp_cols, index=indx)
                  self.dat = self.observed.values.reshape((len(self.indx), 16))

            elif self.kind=='reactive':
                  # separate [go acc, sc, cor quantiles] | error quantiles
                  # and average error quantile est. across conditions
                  dat = np.array([np.vstack(x) for x in datdf.values])
                  self.error_quantiles = np.vstack(dat[:,:,-5:]).mean(axis=0)
                  self.dat = dat[:,:,:-5]

                  idx = np.sort(np.hstack(indx*ncond)); c = np.array(labels*len(indx))
                  ixdf =pd.DataFrame({'idx':idx, cond:c}, columns=['idx', cond])
                  self.observed = pd.concat([ixdf, pd.DataFrame(np.vstack(datdf.unstack().values), columns=qp_cols)], axis=1)
                  self.fits = pd.concat([ixdf, pd.DataFrame(np.zeros((len(idx),16)), columns=qp_cols)], axis=1)

            self.popt = pd.DataFrame(columns=self.infolabels, index=indx)



      def optimize(self, save=True, savepth='./', live_update=True, log_fits=True, disp=True, xtol=1.e-3, ftol=1.e-3, maxfev=500, ntrials=2000, niter=500, prob=np.array([.1, .3, .5, .7, .9])):

            ntrials, ftol, xtol, maxfev, niter, prob = self.set_fitparams(xtol=xtol, ftol=xtol, maxfev=maxfev, ntrials=ntrials, niter=niter, prob=prob, get_params=True)
            if not self.isprepared:
                  self.prepare_fit()

            if self.is_flat:
                  self.__flat_optimize__(log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp, prob=prob)
                  return self.fitp, self.yhat

            elif self.fit_on=='average':
                  self.__avg_optimize__(log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp, prob=prob)
                  return self.fitp, np.append(self.yhat[:-5].reshape(self.ncond, 11), self.yhat[-5:])

            elif self.fit_on in ['subjects', 'bootstrap']:
                  self.__indx_optimize__(log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp, save=save, savepth=savepth, live_update=live_update, niter=niter, prob=prob)

                  if self.kind=='reactive':
                        yhat = np.append(self.yhat[:-5].reshape(self.ncond, 11), self.yhat[-5:])
                  elif self.kind=='proactive':
                        yhat = self.yhat.reshape(self.ncond, 6)
                  return self.fitp, yhat



      def __flat_optimize__(self, log_fits, ntrials, maxfev, ftol, xtol, disp, prob):

            if self.kind=='reactive':
                  self.y = self.dat.mean(axis=0)
            elif self.kind=='proactive':
                  y = y.flatten()

            self.finfo, self.fitp , self.yhat = fit_flat.optimize_theta_flat(self.y, inits=self.inits, wts=self.wts, prob=prob, log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)



      def __avg_optimize__(self, log_fits, ntrials, maxfev, ftol, xtol, disp, prob):

            if self.kind=='reactive':
                  self.y = np.append(self.dat.mean(axis=0), self.error_quantiles)
            elif self.kind=='proactive':
                  y = y.flatten()

            self.finfo, self.fitp , self.yhat = fit.optimize_theta(self.y, inits=self.inits, wts=self.wts, ncond=self.ncond, pc_map=self.pc_map, kind=self.kind, prob=prob, log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)



      def __indx_optimize__(self, log_fits, ntrials, maxfev, ftol, xtol, disp, save, savepth, live_update, niter, prob):

            self.y = np.append(self.dat.mean(axis=0), self.error_quantiles)

            for i, y in enumerate(self.dat):

                  # rejoin grouped y vector, with mean eq
                  # this flattens the full vector, gets
                  # reshaped before weights are applied
                  if self.kind=='reactive':
                        y = np.append(y, self.error_quantiles)
                  elif self.kind=='proactive':
                        y = y.flatten()

                  self.finfo, self.fitp , self.yhat = fit.optimize_theta(y, inits=self.inits, wts=self.wts, ncond=self.ncond, pc_map=self.pc_map, kind=self.kind, prob=prob, log_fits=log_fits, ntrials=ntrials, maxfev=maxfev, ftol=ftol, xtol=xtol, disp=disp)

                  self.popt.iloc[i]=pd.Series({info: self.finfo[info] for info in self.infolabels})
                  self.fits.iloc[self.i: self.i+ self.ncond, self.ncond:] = np.vstack(append_eq(self.yhat))
                  self.i+=self.ncond

                  if save and live_update:
                        self.fits.to_csv(savepth+"fits.csv")
                        self.popt.to_csv(savepth+"popt.csv")



      def set_fitparams(self, ntrials=2000, ftol=1.e-3, xtol=1.e-3, maxfev=500, niter=500, prob=np.array([.1, .3, .5, .7, .9]), get_params=False):

            self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'ftol':ftol, 'xtol':xtol, 'niter':niter, 'prob':prob}

            if self.fit_on=='bootstrap':
                  self.indx=range(self.fitparams['niter'])

            fitp = self.fitparams
            if get_params:
                  return fitp['ntrials'], fitp['ftol'], fitp['xtol'], fitp['maxfev'], fitp['niter'], fitp['prob']


      def get_header(self, params=None, prob=np.array([.1, .3, .5, .7, .9])):

            info = ['nfev','chi','rchi','AIC','BIC','CNVRG']
            cq = ['c'+str(int(n*100)) for n in prob]

            if self.kind == 'reactive':
                  cq = ['c'+str(int(n*100)) for n in prob]
                  eq = ['e'+str(int(n*100)) for n in prob]
                  qp_cols = ['Go'] + self.delays + cq + eq
            else:
                  hi = ['hi'+str(int(n*100)) for n in prob]
                  lo = ['lo'+str(int(n*100)) for n in prob]
                  qp_cols = self.labels + hi + lo
            if params is not None:
                  self.infolabels = params + info

            return qp_cols


      def load_default_inits(self):

            self.inits = {'a': 0.44, 'ssv': 0.947, 'tr': 0.3049, 'v': 1.1224, 'z': 0.15}


      def get_wts(self):
            """
            wtc: weights applied to correct rt quantiles in cost f(x)
                  * P(R | No SSD)j * sdj(.5Qj, ... .95Qj)
            wte: weight applied to error rt quantiles in cost f(x)
                  * P(R | SSD) * sd(.5eQ, ... .95eQ)
            """

            if self.kind == 'reactive':

                  sd = self.observed.std()
                  pc = self.data.query('ttype=="go"').response.mean()
                  sdc = sd.loc['c10':'c90'].values
                  wtc = (pc*(sdc.min(axis=0)/sdc.T)).T

                  pe = self.data.query('ttype=="stop"').response.mean()
                  sde = sd.loc['e10':'e90'].values
                  wte = (pe*(sde.min(axis=0)/sde.T)).T
                  self.wts = np.append(wtc, wte)

            elif self.kind == 'proactive':

                  sd = self.observed.std()
                  sdp = sd.loc['0':'100'].values
                  sdhi = sd.loc['hi10':'hi90'].values
                  sdlo = sd.loc['lo10':'lo90'].values

                  pGo = np.sort(self.data[self.cond].unique())
                  presp = self.data.groupby(self.cond).response.mean().values
                  pc = 1-abs(presp - pGo)

                  wtc = (pc*(sdp.min(axis=0)/sdp.T)).T
                  self.wts = np.hstack([np.ones(len(self.labels)), pc[:3].mean()*(sdhi.min()/sdhi), pc[3:].mean()*(sdlo.min()/sdlo)])


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
