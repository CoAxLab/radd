#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from scipy.stats.mstats import mquantiles as mq
from scipy import optimize
from scipy.io import loadmat
import os, re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors.kde import KernelDensity
import functools

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

def rt_quantiles(data, cutoff=.560, split='HiLo', prob=np.arange(0.1,1.0,0.2)):

      rtq = []
      godf = data[data.response==1]

      if split=='HiLo' and 'HiLo' not in godf.columns:
            godf['HiLo']=['x']
            godf[godf['pGo']<=.5, 'HiLo'] = 'Lo'
            godf[godf['pGo']>.5, 'HiLo'] = 'Hi'
      if split != None:
            splitdf = godf.groupby(split)
      else:
            rts = godf[godf.rt<=cutoff].rt.values
            return mq(rts, prob=prob)*10

      for c, df in splitdf:
            rts = df[df.rt<=cutoff].rt.values
            rtq.append(mq(rts, prob=prob)*10)

      return np.hstack(rtq)


def resample_data(data, n=120, kind='radd'):

      df=data.copy(); bootlist=list()
      if n==None: n=len(df)

      if kind=='radd':
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




def kde_fit_quantiles(rtquants, nsamples=1000, bw=.1):
      """
      takes quantile estimates and fits cumulative density function
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


def resample_data(data, n=120, kind='radd'):

      df=data.copy(); bootlist=list()
      if n==None: n=len(df)

      if kind=='radd':
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


def get_proactive_params(theta, dep='v', pgo=np.arange(0,120,20)):

      if not type(theta)==dict:
            theta=theta.to_dict()['mean']

      keep=['a', 'z', 'v', 'tr', 'ssv', 'ssd']
      keep.pop(keep.index(dep))

      pdict={pg:theta[dep+str(pg)] for pg in pgo}

      for k in theta.keys():
            if k not in keep:
                  theta.pop(k)

      return theta, pdict


def pstop_meanrt(df, filt_rts=True):

        pstop=1-df.groupby('pGo').mean()['response'].values
        if filt_rts:
              godf=df.query("response==1 and rt<=.5451")
        else:
                godf=df.query('response==1')
        go_rt=godf.groupby('pGo').mean()['rt'].values

        return pstop, go_rt*10


def remove_outliers(df, sd=1.95):

      print "len(df) = %s \n\n" % (str(len(df)))

      ssdf=df[df.response==0]
      godf = df[df.response==1]
      bound = godf.rt.std()*sd
      gdf1=godf[godf['rt']<(godf.rt.mean()+bound)]
      gdf2=gdf1[gdf1['rt']>(gdf1.rt.mean()-bound)]

      df_trimmed=pd.concat([godf, ssdf])

      print "bound = %s \nlen(godf) = %i\n len(trimmed) = %i\n" % (str(bound), len(godf), len(df_trimmed))

      return df_trimmed

def update_params(theta):

      if 't_hi' in theta.keys():
            theta['tr'] = theta['t_lo'] + np.random.uniform() * (theta['t_hi'] - theta['t_lo'])
      else:
            theta['tr']=theta['tr']

      if 'z_hi' in theta.keys():
            theta['z'] = theta['z_lo'] + np.random.uniform() * (theta['z_hi'] - theta['z_lo'])
      else:
            theta['z']=theta['z']

      if 'sv' in theta.keys():
            theta['v'] = theta['sv'] * np.random.randn() + theta['v']
      else:
            theta['v']=theta['v']

      return theta


def get_intervar_ranges(theta):
      """
      :args:
            parameters (dict):	dictionary of theta (Go/NoGo Signal Parameters)
                              and sp (Stop Signal Parameters)
      """
      if 'st' in theta.keys():
            theta['t_lo'] = theta['tr'] - theta['st']/2
            theta['t_hi'] = theta['tr'] + theta['st']/2
      if 'sz' in theta.keys():
            theta['z_lo'] = theta['z'] - theta['sz']/2
            theta['z_hi'] = theta['z'] + theta['sz']/2
      return theta


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

def get_intersection(iter1, iter2):

      intersect_set = set(iter1).intersection(set(iter2))
      return ([i for i in intersect_set])



def get_observed(df, prob=([10, 30, 50, 70, 90])):

      rt = df.rt

      inter_prob = [prob[0]-0] + [prob[i] - prob[i-1] for i in range(1,len(prob))] + [100 - prob[-1]]
      rtquant = mq(rt, prob=prob*.01)
      observed = np.ceil((inter_prob)*.01*len(rt)).astype(int)
      n_obs = np.sum(observed)

      return [observed, rtquant, n_obs]


def get_expected(simdf, obs_quant, n_obs):

      simrt = simdf.rt
      q = obs_quant

      first = ([len(simrt[simrt.between(simrt.min(), q[0])])/len(simrt)])*n_obs
      middle = ([len(simrt[simrt.between(q[i-1], q[i])])/len(simrt) for i in range(1,len(q))])*n_obs
      last = ([len(simrt[simrt.between(q[-1], simrt.max())])/len(simrt)])*n_obs

      expected = np.ceil(np.hstack([first, middle, last]))
      return expected


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


def calc_quant_weights(rtvec, quants):

      rt=rtvec.copy()
      q=quants.copy()

      first = rt[rt.between(rt.min(), q[0])].std()
      rest = [rt[rt.between(q[i-1], q[i])].std() for i in range(1,len(q))]
      #last = rt[rt.between(q[-1], rt.max())].std()
      sdrt = np.hstack([first, rest])

      wt = (sdrt[np.ceil(len(sdrt)/2)]/sdrt)**2

      return wt


def ssrt_calc(df, avgrt=.3):

      dfstp = df.query('ttype=="stop"')
      dfgo = df.query('choice=="go"')

      pGoErr = ([idf.response.mean() for ix, idf in dfstp.groupby('idx')])
      nlist = [int(pGoErr[i]*len(idf)) for i, (ix, idf) in enumerate(df.groupby('idx'))]

      GoRTs = ([idf.rt.sort(inplace=False).values for ix, idf in dfgo.groupby('idx')])
      ssrt_list = ([GoRTs[i][nlist[i]] for i in np.arange(len(nlist))]) - avgrt

      return ssrt_list
