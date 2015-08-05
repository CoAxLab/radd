#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.toolbox.utils import *
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



      def __init__(self, kind='radd', inits=None, data=None, fit_on='subjects', depends_on=None, niter=50, scale=1., fit_whole_model=True, tb=None, scale_rts=False, fit_noise=False, pro_ss=False, dynamic='hyp', split='HL', *args, **kws):

            self.data = data
            self.kind = kind
            self.depends_on = depends_on
            self.fit_on = fit_on
            self.scale = scale
            self.fit_whole_model = fit_whole_model
            self.dynamic = dynamic

            if 'pro' in self.kind:
                  self.data_style='pro'
                  if depends_on is None:
                        depends_on = {'v':'pGo'}
                  self.split=split
                  self.nrt_cond=len(split)
            else:
                  self.data_style='re'
                  if depends_on is None:
                        depends_on = {'v':'Cond'}
                  self.split=None
                  self.nrt_cond=None

            self.depends_on = depends_on
            self.cond = self.depends_on.values()[0]
            self.labels = np.sort(data[self.cond].unique())
            self.ncond = len(self.labels)

            if inits is None:
                  self.__get_default_inits__()
            else:
                  self.inits = inits

            self.__remove_outliers__(sd=1.5, verbose=False)
            self.__check_inits__(fit_noise=fit_noise, pro_ss=pro_ss)
            self.tb = self.data[self.data.response==1].rt.max()

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

            self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'disp':disp, 'ftol':ftol, 'xtol':xtol, 'niter':niter, 'prob':prob, 'log_fits':log_fits, 'tb':self.tb, 'ssd':self.ssd, 'wts':self.wts, 'ncond':self.ncond, 'pGo':self.pGo, 'flat_wts':self.fwts, 'scale':self.scale, 'depends_on': self.depends_on, 'dynamic': self.dynamic, 'fit_whole_model': self.fit_whole_model}

            if get_params:
                  return self.fitparams

      def __extract_popt_fitinfo__(self, finfo=None):
            """ takes optimized dict or DF of vectorized parameters and
            returns dict with only depends_on.keys() containing vectorized vals.
            Is accessed by fit.Optimizer objects after optimization routine.

            ::Arguments::
                  finfo (dict/DF):
                        finfo is dict if self.fit_on is 'average'
                        and DF if self.fit_on is 'subjects' or 'bootstrap'
                        contains optimized parameters
            ::Returns::
                  popt (dict):
                        dict with only depends_on.keys() containing
                        vectorized vals
            """


            if finfo is None:
                  try:
                        finfo=self.fitinfo.mean()
                  except Exception:
                        finfo=self.fitinfo

            finfo=dict(deepcopy(finfo))
            popt=dict(deepcopy(self.inits))
            pc_map = self.pc_map;

            for pkey in popt.keys():
                  if pkey in self.depends_on.keys():
                        popt[pkey] = np.array([finfo[pc] for pc in pc_map[pkey]])
                        continue
                  popt[pkey]=finfo[pkey]

            return popt

      def __make_dataframes__(self, qp_cols):
            """ Generates the following dataframes and arrays:

            ::Arguments::

                  qp_cols:
                        header for observed/fits dataframes
            ::Returns::

                  None (All dataframes and vectors are stored in dict and assigned
                  as <dframes> attr)

            observed (DF):
                  Contains Prob and RT quant. for each subject
                  used to calc. cost fx weights
            fits (DF):
                  empty DF shaped like observed DF, used to store simulated
                  predictions of the optimized model
            fitinfo (DF):
                  stores all opt. parameter values and model fit statistics
            dat (ndarray):
                  contains all subject/boot. y vectors entered into costfx
            avg_y (ndarray):
                  average y vector for each condition entered into costfx
            flat_y (1d array):
                  average y vector used to initialize parameters prior to fitting
                  conditional model. calculated collapsing across conditions
            """

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

            self.wts, self.fwts = ensure_numerical_wts(self.wts, self.fwts)


      def __remove_outliers__(self, sd=1.5, verbose=False):
            self.data = remove_outliers(self.data, sd=sd, verbose=verbose)

      def __get_header__(self, params=None, data_style='re', labels=[], prob=np.array([.1, .3, .5, .7, .9])):

            if not hasattr(self, 'delays'):
                  self.delays = self.ssd
            qp_cols = get_header(params=params, data_style=self.data_style, labels=self.labels, prob=prob, delays=self.delays)
            if params is not None:
                  self.infolabels = qp_cols[1]
            return qp_cols[0]

      def __get_default_inits__(self, include_ss=False, fit_noise=False):
            self.inits = get_default_inits(kind=self.kind, dynamic=self.dynamic, depends_on=self.depends_on, include_ss=include_ss, fit_noise=fit_noise)

      def __check_inits__(self, pro_ss=False, fit_noise=False):
            self.inits = check_inits(inits=self.inits, kind=self.kind, dynamic=self.dynamic, pro_ss=pro_ss, fit_noise=fit_noise)

      def __make_proRT_conds__(self):
            self.data = make_proRT_conds(self.data, self.split)
            self.prort_conds_prepared = True

      def __rename_bad_cols__(self):
            self.data = rename_bad_cols(self.data)
