#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
import os, re
from numpy import array
from sklearn.neighbors.kde import KernelDensity
from scipy.stats.mstats import mquantiles as mq
from scipy import optimize
import functools


def remove_outliers(df, sd=1.5, verbose=False):

      ssdf=df[df.response==0]
      godf = df[df.response==1]
      bound = godf.rt.std()*sd
      rmslow=godf[godf['rt']<(godf.rt.mean()+bound)]
      clean_go=rmslow[rmslow['rt']>(godf.rt.mean()-bound)]

      clean=pd.concat([clean_go, ssdf])
      if verbose:
            pct_removed = len(clean)*1./len(df)
            print "len(df): %i\nbound: %s \nlen(cleaned): %i\npercent removed: %.5f" % (len(df), str(bound), len(clean), pct_removed)

      return clean


def rangl_data(data, re_cut=.650, pro_cut=.54502, kind='radd', prob=([.1, .3, .5, .7, .9])):

      if kind == 'radd':
            gac = data.query('ttype=="go"').acc.mean()
            sacc = data.query('ttype=="stop"').groupby('ssd').mean()['acc'].values
            grt = data.query('ttype=="go" & acc==1').rt.values
            ert = data.query('response==1 & acc==0').rt.values
            gq = mq(grt, prob=prob)
            eq = mq(ert, prob=prob)
            return np.hstack([gac, sacc, gq*10, eq*10])

      elif kind=='pro':
            return 1-data.response.mean()


def ensure_numerical_wts(wts, fwts):

      # test inf
      wts[np.isinf(wts)] = np.median(wts[~np.isinf(wts)])
      fwts[np.isinf(fwts)] = np.median(fwts[~np.isinf(fwts)])

      # test nan
      wts[np.isnan(wts)] = np.median(wts[~np.isnan(wts)])
      fwts[np.isnan(fwts)] = np.median(fwts[~np.isnan(fwts)])

      return wts, fwts


def kde_fit_quantiles(rtquants, nsamples=1000, bw=.1):
      """ takes quantile estimates and fits cumulative density function
      returns samples to pass to sns.kdeplot()
      """
      kdefit = KernelDensity(kernel='gaussian', bandwidth=bw).fit(rtquants)
      samples = kdefit.sample(n_samples=nsamples).flatten()
      return samples


def aic(model):
      k = len(model.get_stochasticts())
      logp = sum([x.logp for x in model.get_observeds()['node']])
      return 2 * k - 2 * logp


def bic(model):
      k = len(model.get_stochastics())
      n = len(model.data)
      logp = sum([x.logp for x in model.get_observeds()['node']])
      return -2 * logp + k * np.log(n)


def sigmoid(p,x):
      x0,y0,c,k=p
      y = c / (1 + np.exp(k*(x-x0))) + y0
      return y


def residuals(p,x,y):
      return y - sigmoid(p,x)


def res(arr,lower=0.0,upper=1.0):
      arr=arr.copy()
      if lower>upper: lower,upper=upper,lower
      arr -= arr.min()
      arr *= (upper-lower)/arr.max()
      arr += lower
      return arr


def rwr(X, get_index=False, n=None):
      """
      Modified from http://nbviewer.ipython.org/gist/aflaxman/6871948
      """

      if isinstance(X, pd.Series):
            X = X.copy()
            X.index = range(len(X.index))
      if n == None:
            n = len(X)

      resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
      X_resample = (X[resample_i])

      if get_index:
            return resample_i
      else:
            return X_resample


def ssrt_calc(df, avgrt=.3):

      dfstp = df.query('ttype=="stop"')
      dfgo = df.query('choice=="go"')

      pGoErr = ([idf.response.mean() for ix, idf in dfstp.groupby('idx')])
      nlist = [int(pGoErr[i]*len(idf)) for i, (ix, idf) in enumerate(df.groupby('idx'))]

      GoRTs = ([idf.rt.sort(inplace=False).values for ix, idf in dfgo.groupby('idx')])
      ssrt_list = ([GoRTs[i][nlist[i]] for i in np.arange(len(nlist))]) - avgrt

      return ssrt_list


def make_proRT_conds(data, split=None, rt_cix=None):

      if np.any(data['pGo'].values > 1):
            data['pGo']=data['pGo']*.01
      if np.any(data['rt'].values > 5):
            data['rt']=data['rt']*.001

      if split != None:
            pg = split*.01
            data['HL']='x'
            data.ix[data.pGo>pg, 'HL']=1
            data.ix[data.pGo<=pg, 'HL']=2

      # get index to split rts during fits
      rt_cix = len(data.query('HL==1').pGo.unique())
      return data, rt_cix


def get_obs_quant_counts(df, prob=([.10, .30, .50, .70, .90])):

      if type(df) == pd.Series:
            rt=df.copy()
      else:
            rt=df.rt.copy()

      inter_prob = [prob[0]-0] + [prob[i] - prob[i-1] for i in range(1,len(prob))] + [1.00 - prob[-1]]
      obs_quant = mq(rt, prob=prob)
      observed = np.ceil((inter_prob)*len(rt)*.94).astype(int)

      return observed, obs_quant


def get_exp_counts(simdf, obs_quant, n_obs, prob=([.10, .30, .50, .70, .90])):

      if type(simdf) == pd.Series:
            simrt=simdf.copy()
      else:
            simrt = simdf.rt.copy()
      exp_quant = mq(simrt, prob); oq = obs_quant
      expected = np.ceil(np.diff([0] + [pscore(simrt, oq_rt)*.01 for oq_rt in oq] + [1]) * n_obs)

      return expected, exp_quant


def get_intersection(iter1, iter2):

      intersect_set = set(iter1).intersection(set(iter2))
      return ([i for i in intersect_set])


def rename_bad_cols(data):

      if 'trial_type' in data.columns:
            data.rename(columns={'trial_type':'ttype'}, inplace=True)

      return data


#def calc_quant_weights(rtvec, quants):
#
#      rt=rtvec.copy()
#      q=quants.copy()
#
#      first = rt[rt.between(rt.min(), q[0])].std()
#      rest = [rt[rt.between(q[i-1], q[i])].std() for i in range(1,len(q))]
#      #last = rt[rt.between(q[-1], rt.max())].std()
#      sdrt = np.hstack([first, rest])
#
#      wt = (sdrt[np.ceil(len(sdrt)/2)]/sdrt)**2
#
#      return wt


#def get_observed_vector(df, prob=([10, 30, 50, 70, 90])):
#
#      rt = df.rt
#
#      inter_prob = [prob[0]-0] + [prob[i] - prob[i-1] for i in range(1,len(prob))] + [100 - prob[-1]]
#      rtquant = mq(rt, prob=prob*.01)
#      observed = np.ceil((inter_prob)*.01*len(rt)).astype(int)
#      n_obs = np.sum(observed)
#
#      return [observed, rtquant, n_obs]
#
#
#def get_expected_vector(simdf, obs_quant, n_obs):
#
#      simrt = simdf.rt
#      q = obs_quant
#
#      first = ([len(simrt[simrt.between(simrt.min(), q[0])])/len(simrt)])*n_obs
#      middle = ([len(simrt[simrt.between(q[i-1], q[i])])/len(simrt) for i in range(1,len(q))])*n_obs
#      last = ([len(simrt[simrt.between(q[-1], simrt.max())])/len(simrt)])*n_obs
#
#      expected = np.ceil(np.hstack([first, middle, last]))
#      return expected
#
