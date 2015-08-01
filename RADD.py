#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from radd.misc.messages import saygo


class RADDCore(object):

      """ Parent class for constructing shared attributes and methods
      of Model & Optimizer objects. Not meant to be used directly.

      Contains methods for building dataframes, generating observed data vectors
      that are entered into cost function during fitting, calculating summary measures
      and weight matrix for weighting residuals during optimization.

      TODO: COMPLETE DOCSTRINGS
      """



      def __init__(self, kind='radd', inits=None, data=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, tb=None, scale_rts=False, *args, **kws):

            self.data = data
            self.inits = inits
            self.kind = kind
            self.depends_on = depends_on
            self.fit_on = fit_on

            if scale_rts:
                  self.scale = 10.
            else:
                  self.scale = 1.

            if 'ssd' in inits.keys():
                  del self.inits['ssd']
            if 'pGo' in inits.keys():
                  del self.inits['pGo']

            if self.kind in ['radd', 'irace', 'xradd', 'xirace']:
                  self.data_style='re'

            elif self.kind in ['pro', 'xpro']:
                  self.data_style='pro'

            if fit_whole_model:
                  self.fit_flat=True

            if self.depends_on != None:
                  self.cond=depends_on.values()[0]
                  self.labels=data[self.cond].unique()
                  self.ncond=len(self.labels)

            if self.data_style=='re':
                  ssd = data[data.ttype=="stop"].ssd.unique()
                  self.pGo = len(data[data.ttype=='go'])/len(data)
                  self.delays = sorted(ssd.astype(np.int))
                  self.ssd = np.array(self.delays)*.001

            elif self.data_style=='pro':
                  self.pGo = sorted(self.data.pGo.unique())
                  self.ssd=np.array([.450])

            if tb is None:
                  self.tb = data[data.response==1].rt.max()
            else:
                  self.tb=tb
            if self.fit_on=='bootstrap':
                  self.indx = range(niter)
            else:
                  self.indx = list(data.idx.unique())


      def rangl_data(self, data, kind='radd', prob=np.array([.1, .3, .5, .7, .9])):

            if self.data_style=='re':
                  gac = data.query('ttype=="go"').acc.mean()
                  sacc = data.query('ttype=="stop"').groupby('ssd').mean()['acc'].values
                  grt = data.query('ttype=="go" & acc==1').rt.values
                  ert = data.query('response==1 & acc==0').rt.values
                  gq = mq(grt, prob=prob)
                  eq = mq(ert, prob=prob)
                  return np.hstack([gac, sacc, gq*self.scale, eq*self.scale])

            elif self.data_style=='pro':
                  godf = data[data.response==1]
                  godf['response']=np.where(godf.rt<self.tb, 1, 0)
                  data = pd.concat([godf, data[data.response==0]])
                  return 1-data.response.mean()



      def rt_quantiles(self, data, split='HiLo', prob=np.arange(0.1,1.0,0.2)):
            rtq = []
            godfx = data[(data.response==1) & (data.pGo>0.)]
            godfx.loc[:, 'response'] = np.where(godfx.rt<self.tb, 1, 0)
            godf = godfx.query('response==1')

            if split=='HiLo':
                  godf['HiLo']='x'
                  godf.ix[godf['pGo'].isin([.2,.4, .6]), 'HiLo'] = 'Lo'
                  godf.ix[godf['pGo'].isin([.6, .8, 1.0]), 'HiLo'] = 'Hi'
            if split != None:
                  splitdf = godf.groupby(split)
            else:
                  rts = godf[godf.rt<=self.tb].rt.values
                  return mq(rts, prob=prob)*self.scale

            for c, df in splitdf:
                  rts = df[df.rt<=self.tb].rt.values
                  rtq.append(mq(rts, prob=prob)*self.scale)

            return np.hstack(rtq)


      def resample_data(self, data, n=120, kind='radd'):

            df=data.copy(); bootlist=list()
            if n==None: n=len(df)

            if self.data_style=='re':
                  for ssd, ssdf in df.groupby('ssd'):
                        boots = ssdf.reset_index(drop=True)
                        orig_ix = np.asarray(boots.index[:])
                        resampled_ix = rwr(orig_ix, get_index=True, n=n)
                        bootdf = ssdf.irow(resampled_ix)
                        bootlist.append(bootdf)
                        #concatenate and return all resampled conditions
                        return rangl_re(pd.concat(bootlist))
            elif self.data_style=='pro':
                  boots = df.reset_index(drop=True)
                  orig_ix = np.asarray(boots.index[:])
                  resampled_ix = rwr(orig_ix, get_index=True, n=n)
                  bootdf = df.irow(resampled_ix)
                  bootdf_list.append(bootdf)
                  return rangl_pro(pd.concat(bootdf_list), rt_cutoff=rt_cutoff)


      def set_fitparams(self, ntrials=10000, ftol=1.e-4, xtol=1.e-4, maxfev=1000, niter=500, log_fits=True, disp=True, prob=np.array([.1, .3, .5, .7, .9]), get_params=False, **kwgs):

            if not hasattr(self, 'fitparams'):
                  self.fitparams={}

            self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'disp':disp, 'ftol':ftol, 'xtol':xtol, 'niter':niter, 'prob':prob, 'log_fits':log_fits, 'tb':self.tb, 'ssd':self.ssd, 'wts':self.wts, 'ncond':self.ncond, 'pGo':self.pGo, 'flat_wts':self.fwts, 'scale':self.scale, 'depends_on': self.depends_on}

            if get_params:
                  return self.fitparams


      def __make_dataframes__(self, qp_cols):

            cond = self.cond; ncond = self.ncond
            data = self.data; indx = self.indx
            labels = self.labels

            ic_grp = data.groupby(['idx', cond])
            c_grp = data.groupby([cond])
            i_grp = data.groupby(['idx'])

            if self.fit_on=='bootstrap':
                  self.dat = np.vstack([i_grp.apply(self.resample_data, kind=self.kind).values for i in indx]).unstack()

            if self.data_style=='re':
                  datdf = ic_grp.apply(self.rangl_data, kind=self.kind).unstack().unstack()
                  indxx = pd.Series(indx*ncond, name='idx')
                  obs = pd.DataFrame(np.vstack(datdf.values), columns=qp_cols, index=indxx)
                  obs[cond]=np.sort(labels*len(indx))
                  self.observed = obs.sort_index().reset_index()
                  self.avg_y = self.observed.groupby(cond).mean().loc[:,qp_cols[0] : qp_cols[-1]].values
                  self.flat_y = self.observed.loc[:, qp_cols[0] : qp_cols[-1]].mean().values
                  dat = self.observed.loc[:,qp_cols[0]:qp_cols[-1]].values.reshape(len(indx),ncond,16)
                  fits = pd.DataFrame(np.zeros((len(indxx),len(qp_cols))), columns=qp_cols, index=indxx)

            elif self.data_style=='pro':
                  datdf = ic_grp.apply(self.rangl_data, kind=self.kind).unstack()
                  rtdat = pd.DataFrame(np.vstack(i_grp.apply(self.rt_quantiles).values), index=indx)
                  rtdat[rtdat<.1] = np.nan
                  rts_flat = pd.DataFrame(np.vstack(i_grp.apply(self.rt_quantiles, split=None).values), index=indx)
                  self.observed = pd.concat([datdf, rtdat], axis=1)
                  self.observed.columns = qp_cols
                  self.avg_y = self.observed.mean().values
                  self.flat_y=np.append(datdf.mean().mean(), rts_flat.mean())
                  dat = self.observed.values.reshape((len(indx), len(qp_cols)))
                  fits = pd.DataFrame(np.zeros_like(dat), columns=qp_cols, index=indx)

            fitinfo = pd.DataFrame(columns=self.infolabels, index=indx)
            self.dframes = {'data':self.data, 'flat_y':self.flat_y, 'avg_y':self.avg_y, 'fitinfo': fitinfo, 'fits': fits, 'observed': self.observed, 'dat':dat}


      def __get_header__(self, params=None, prob=np.array([.1, .3, .5, .7, .9])):

            info = ['nfev','chi','rchi','AIC','BIC','CNVRG']
            cq = ['c'+str(int(n*100)) for n in prob]

            if self.data_style=='re':
                  cq = ['c'+str(int(n*100)) for n in prob]
                  eq = ['e'+str(int(n*100)) for n in prob]
                  qp_cols = ['Go'] + self.delays + cq + eq
            elif self.data_style=='pro':
                  hi = ['hi'+str(int(n*100)) for n in prob]
                  lo = ['lo'+str(int(n*100)) for n in prob]
                  qp_cols = self.labels + hi + lo

            if params is not None:
                  self.infolabels = params + info

            return qp_cols


      def get_wts(self):
            """
            wtc: weights applied to correct rt quantiles in cost f(x)
                  * P(R | No SSD)j * sdj(.5Qj, ... .95Qj)
            wte: weight applied to error rt quantiles in cost f(x)
                  * P(R | SSD) * sd(.5eQ, ... .95eQ)
            """


            if self.data_style=='re':
                  obs_var = self.observed.groupby(self.cond).sem().loc[:,'Go':]
                  qvar = obs_var.values[:,6:]
                  pvar = obs_var.values[:,:6]
                  go = self.data.query('ttype=="go"').response.mean()
                  st = self.data.query('ttype=="stop"').response.mean()

                  sq_ratio = (np.median(qvar, axis=1)/qvar.T).T
                  wt_go = (go*sq_ratio[:, :5].T).T
                  wt_err = (st*sq_ratio[:, -5:].T).T
                  qwts = np.hstack(np.vstack(zip(wt_go, wt_err))).reshape(self.ncond, 10)
                  pwts = (np.median(pvar, axis=1)/pvar.T).T
                  self.wts = np.hstack([np.append(p, w) for p, w in zip(pwts, qwts)])
                  # calculate flat weights (collapsing across conditions)
                  self.fwts = self.wts.reshape(self.ncond, 16).mean(axis=0)

            elif self.data_style=='pro':

                  upper = self.data[self.data.isin([.6,.8,1.0])].response.mean()
                  lower = self.data[self.data.pGo.isin([.2,.4,.6])].response.mean()

                  #qvar = self.observed.std().iloc[6:].values
                  #hi = qvar[:5]; lo = qvar[5:]
                  #qwts = np.hstack([upper*(hi[2]/hi), lower*(lo[2]/lo)])

                  pvar = self.data.groupby('pGo').std().response.values
                  psub1 = np.median(pvar[:-1])/pvar[:-1]
                  pwts = np.append(psub1, psub1.max())
                  #pwts = np.array([1.5,1,1,1,1,1.5])
                  #self.wts = np.hstack([pwts, qwts])
                  qvar = self.observed.std().iloc[6:].values.reshape(2,5)
                  #qr = np.median(qvar)/qvar
                  #qwts = np.append(upper*qr[:5], lower*qr[5:])

                  sq_ratio = (np.median(qvar, axis=1)/qvar.T).T
                  wt_hi = upper*sq_ratio[0, :]
                  wt_lo = lower*sq_ratio[1, :]
                  self.wts = np.hstack([pwts, wt_hi, wt_lo])

                  #qwts = np.hstack([upper*(hi[2]/hi), lower*(lo[2]/lo)])
                  #pwts = np.array([1.5,  1,  1,  1,  2, 2])
                  #self.wts = np.hstack([pwts, qwts])
                  # calculate flat weights (collapsing across conditions)
                  nogo = self.wts[:len(self.pGo)].mean()
                  quant = self.wts[len(self.pGo):].reshape(2, 5).mean(axis=0)
                  self.fwts = np.hstack([nogo, quant])

      def load_default_inits(self):

            if self.data_style=='re':
                  self.inits = {'a': 0.44, 'ssv': 0.947, 'tr': 0.3049, 'v': 1.1224, 'z': 0.15}
            elif self.kind in ['pro', 'xpro']:
                  self.inits = {}
