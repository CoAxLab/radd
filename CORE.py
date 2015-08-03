#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from scipy.stats.mstats import mquantiles as mq
from radd.toolbox.messages import saygo


class RADDCore(object):

      """ Parent class for constructing shared attributes and methods
      of Model & Optimizer objects. Not meant to be used directly.

      Contains methods for building dataframes, generating observed data vectors
      that are entered into cost function during fitting, calculating summary measures
      and weight matrix for weighting residuals during optimization.

      TODO: COMPLETE DOCSTRINGS
      """



      def __init__(self, kind='radd', inits=None, data=None, fit_on='subjects', depends_on={'v':'Cond'}, niter=50, scale=1., fit_whole_model=True, tb=None, scale_rts=False, fit_noise=False, pro_ss=False, dynamic='hyp', split='HL', *args, **kws):

            self.data = data
            self.kind = kind
            self.depends_on = depends_on
            self.fit_on = fit_on
            self.scale = scale
            self.fit_flat = fit_whole_model
            self.dynamic = dynamic

            if 'pro' in self.kind:
                  self.data_style='pro'
                  if depends_on is None:
                        depends_on = {'v':'pGo'}
                  self.split=split
                  self.nrt_cond=len(split)
            else:
                  self.data_style='re'
                  self.split=None
                  self.nrt_cond=None

            self.cond = depends_on.values()[0]
            self.labels = np.sort(data[self.cond].unique())
            self.ncond = len(self.labels)
            self.tb = data[data.response==1].rt.max()

            if inits is None:
                  self.inits = self.get_default_inits()
            else:
                  self.inits = inits

            self.__check_inits__(fit_noise=fit_noise, pro_ss=pro_ss)

            if self.data_style=='re':
                  ssd = data[data.ttype=="stop"].ssd.unique()
                  self.pGo = len(data[data.ttype=='go'])/len(data)
                  self.delays = sorted(ssd.astype(np.int))
                  self.ssd = np.array(self.delays)*.001
            elif self.data_style=='pro':
                  self.pGo = sorted(self.data.pGo.unique())
                  self.ssd=np.array([.450])

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


      def rt_quantiles(self, data, split='HL', prob=np.arange(0.1,1.0,0.2)):

            if not hasattr(self, "prort_conds_prepared"):
                  self.__make_proRT_conds__()

            godfx = data[(data.response==1)]# & (data.pGo>0.)]
            godfx.loc[:, 'response'] = np.where(godfx.rt<=self.tb, 1, 0)
            godf = godfx.query('response==1')

            if split == None:
                  rts = godf[godf.rt<=self.tb].rt.values
                  return mq(rts, prob=prob)

            rtq = []
            for i in range(1, self.nrt_cond+1):
                  if i not in godf[split].unique():
                        rtq.append(array([np.nan]*len(prob)))
                  else:
                        rts = godf[(godf[split]==i)&(godf.rt<=self.tb)].rt.values
                        rtq.append(mq(rts, prob=prob))

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


      def set_fitparams(self, ntrials=10000, ftol=1.e-20, xtol=1.e-20, maxfev=5000, niter=500, log_fits=True, disp=True, prob=np.array([.1, .3, .5, .7, .9]), get_params=False, **kwgs):

            if not hasattr(self, 'fitparams'):
                  self.fitparams={}

            self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'disp':disp, 'ftol':ftol, 'xtol':xtol, 'niter':niter, 'prob':prob, 'log_fits':log_fits, 'tb':self.tb, 'ssd':self.ssd, 'wts':self.wts, 'ncond':self.ncond, 'pGo':self.pGo, 'flat_wts':self.fwts, 'scale':self.scale, 'depends_on': self.depends_on, 'dynamic': self.dynamic}

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
                  #if self.split=='HL':
                  hi = ['hi'+str(int(n*100)) for n in prob]
                  lo = ['lo'+str(int(n*100)) for n in prob]
                  rt_cols = hi + lo

                  qp_cols = self.labels + rt_cols

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

            nc = self.ncond; cond=self.cond;


            if self.data_style=='re':
                  obs_var = self.observed.groupby(cond).sem().loc[:,'Go':]
                  qvar = obs_var.values[:,6:]
                  pvar = obs_var.values[:,:6]
                  go = self.data.query('ttype=="go"').response.mean()
                  st = self.data.query('ttype=="stop"').response.mean()

                  sq_ratio = (np.median(qvar, axis=1)/qvar.T).T
                  wt_go = (go*sq_ratio[:, :5].T).T
                  wt_err = (st*sq_ratio[:, -5:].T).T
                  qwts = np.hstack(np.vstack(zip(wt_go, wt_err))).reshape(nc, 10)
                  pwts = (np.median(pvar, axis=1)/pvar.T).T
                  self.wts = np.hstack([np.append(p, w) for p, w in zip(pwts, qwts)])
                  # calculate flat weights (collapsing across conditions)
                  self.fwts = self.wts.reshape(nc, 16).mean(axis=0)

            elif self.data_style=='pro':

                  nrtc = self.nrt_cond;
                  pvar = self.data.groupby(cond).std().response.values
                  psub1 = np.median(pvar[:-1])/pvar[:-1]
                  pwts = np.append(psub1, psub1.max())

                  presponse = self.data.groupby(self.split).mean().response
                  qvar = self.observed.std().iloc[nc:].values.reshape(nrtc, 5)
                  sq_ratio = (np.median(qvar, axis=1)/qvar.T).T
                  qwts = np.hstack((presponse.values.T * sq_ratio.T).T)
                  self.wts = np.hstack([pwts, qwts])

                  #self.wts[self.wts>=5] = 2.5

                  #calculate flat weights (collapsing across conditions)
                  nogo = self.wts[:nc].mean()
                  quant = self.wts[nc:].reshape(nrtc, 5).mean(axis=0)
                  self.fwts = np.hstack([nogo, quant])


      def get_default_inits(self, include_ss=False, fit_noise=False):

            if 'radd' in self.kind:
                  inits = {'a':0.4441, 'ssv':-0.9473, 'tr':0.3049, 'v':1.0919, 'z':0.1542}

            elif 'pro' in self.kind:
                  if len(self.depends_on.keys())>1:
                        inits = {'a':.39, 'tr': 0.2939, 'v': 1.0919}
                  elif 'tr' in self.depends_on.keys():
                        inits = {'a':0.3267, 'tr':0.3192, 'v': 1.3813}
                  elif 'v' in self.depends_on.keys():
                        inits = {'a':0.4748, 'tr':0.2725,'v':1.6961}
                  elif 'xb' in self.depends_on.keys() and self.dynamic == 'exp':
                        inits = {'xb': 1.2148,'a': 0.473022,"tr":0.330223, "v":1.64306}
                  elif 'xb' in self.depends_on.keys() and self.dynamic == 'hyp':
                        inits = {'xb': .01, 'a': 0.473022,"tr":0.330223, "v":1.24306}

            elif 'race' in self.kind:
                  inits = {'a':0.3926740, 'ssv':1.1244, 'tr':0.33502, 'v':1.0379,  'z':0.1501}

            return inits



      def __check_inits__(self, pro_ss=False, fit_noise=False):

            single_bound_models = ['xirace', 'irace', 'xpro', 'pro']
            x = array([1.39320, 1.52083, 1.65874, 1.75701, 1.89732, 1.94935])
            if 'ssd' in self.inits.keys():
                  del self.inits['ssd']
            if 'pGo' in self.inits.keys():
                  del self.inits['pGo']

            if pro_ss and 'ssv' not in self.inits.keys():
                  self.inits['ssv'] = -0.9976

            if self.kind in single_bound_models and 'z' in self.inits.keys():
                  z=self.inits.pop('z')
                  self.inits['a']=self.inits['a']-z

            if 'race' in self.kind:
                  self.inits['ssv']=abs(self.inits['ssv'])
            elif 'radd' in self.kind:
                  self.inits['ssv']=-abs(self.inits['ssv'])

            if 'pro' in self.kind:
                  if pro_ss and 'ssv' not in self.inits.keys():
                        self.inits['ssv'] = -0.9976
                  elif not pro_ss and 'ssv' in self.inits.keys():
                        ssv=self.inits.pop('ssv')

            if 'x' in self.kind and 'xb' not in self.inits.keys():
                  if self.dynamic == 'exp':
                        self.inits['xb'] = 2
                  elif self.dynamic == 'hyp':
                        self.inits['xb'] = .02

            if fit_noise and 'si' not in self.inits.keys():
                  self.inits['si'] = .01


      def __make_proRT_conds__(self):

            if np.any(self.data['pGo'].values > 1):
                  self.data['pGo']=self.data['pGo']*.01
            if np.any(self.data['rt'].values > 5):
                  self.data['rt']=self.data['rt']*.001

            if self.split=='HL':
                  self.data['HL']='x'
                  #godf.ix[godf['pGo'].isin([.2,.4, .6]), 'HiLo'] = 'Lo'
                  #godf.ix[godf['pGo'].isin([.6, .8, 1.0]), 'HiLo'] = 'Hi'
                  self.data.ix[self.data.pGo>.5, 'HL']=1
                  self.data.ix[self.data.pGo<=.5, 'HL']=2

            #if self.split=='HML' or self.ncond%2:
            #      self.split='HML'
            #      self.data['HML']='x'
            #      self.data.ix[self.data['pGo']==1.00, 'HML'] = 1
            #      self.data.ix[(self.data.pGo>.50)&(self.data['pGo']<1.00), 'HML'] = 2
            #      self.data.ix[self.data.pGo<=.50, 'HML'] = 3
            #

            self.prort_conds_prepared = True

      def rename_bad_cols(self):

            if 'trial_type' in self.data.columns:
                  self.data.rename(columns={'trial_type':'ttype'}, inplace=True)
