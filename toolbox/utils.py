#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from numpy import array
from scipy.io import loadmat
import os, re



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



def get_default_inits(kind='radd', dynamic='hyp', depends_on={}, include_ss=False, fit_noise=False):

      if 'radd' in kind:
            if set(['v', 'tr']).issubset(depends_on.keys()):
                  inits = {'a':.45, 'ssv':-0.9473, 'tr': 0.2939, 'v': 1.0919, 'z':0.1542}
            elif ['v'] in depends_on.keys():
                  inits = {'a':0.4441, 'v':array([1.1078, 1.0651]), 'ssv':-0.9473, 'tr':0.3049, 'z':0.1542}
            elif ['tr'] == depends_on.keys():
                  inits = {'a': 0.4442, 'ssv': -0.9519, 'tr': array([.3027, 0.3104]), 'v': 1.0959, 'z':0.1541}
            elif ['a'] == depends_on.keys():
                  inits = {'a': array([0.43864876, 44879925]), 'tr': 0.3049, 'v': 1.0842, 'ssv': -0.9293, 'z': 0.1539}

      elif 'pro' in kind:
            if set(['v', 'tr']).issubset(depends_on.keys()):
                  inits = {'a':.39, 'tr': 0.2939, 'v': 1.0919}
            elif ['tr']==depends_on.keys():
                  inits = {'a':0.3267, 'tr':0.3192, 'v': 1.3813}
            elif ['v']==depends_on.keys():
                  inits = {'a':0.4748, 'tr':0.2725,'v':1.6961}
            elif ['xb']==depends_on.keys():
                  inits={'a': 0.473022, "tr":0.330223, "v":1.64306}
                  inits['xb'] = array([0.257877,0.649422,1.03762,1.307329,1.934637,2.101918])

            if 'x' in kind:
                  if 'xb' in depends_on.keys():
                        pass
                  elif dynamic == 'exp':
                        inits = {'a':0.4836, 'xb': 1.4604 , 'tr': 0.3375}
                        inits['v'] = array([1.2628, 1.4304, 1.5705, 1.701, 1.8682, 1.9973])
                  elif dynamic == 'hyp':
                        inits = {'xb': .01, 'a': 0.473022,"tr":0.330223, "v":1.24306}

      elif 'race' in kind:
            if ['v']==depends_on.keys():
                  inits = {'v': array([1.0687, 1.0057]),'a': 0.3926741, 'tr': 0.3379, 'ssv': 1.1243, 'z': 0.1500}
            else:
                  inits = {'v': 1.0306, 'a': 0.3926741, 'tr': 0.3379, 'ssv': 1.1243, 'z': 0.1500}

      return inits


def ensure_numerical_wts(wts, fwts):

      # test inf
      wts[np.isinf(wts)] = np.median(wts[~np.isinf(wts)])
      fwts[np.isinf(fwts)] = np.median(fwts[~np.isinf(fwts)])

      # test nan
      wts[np.isnan(wts)] = np.median(wts[~np.isnan(wts)])
      fwts[np.isnan(fwts)] = np.median(fwts[~np.isnan(fwts)])

      return wts, fwts


def get_header(params=None, data_style='re', labels=[], delays=[], prob=np.array([.1, .3, .5, .7, .9])):

      info = ['nfev','chi','rchi','AIC','BIC','CNVRG']
      if data_style=='re':
            cq = ['c'+str(int(n*100)) for n in prob]
            eq = ['e'+str(int(n*100)) for n in prob]
            qp_cols = ['Go'] + delays + cq + eq
      else:
            hi = ['hi'+str(int(n*100)) for n in prob]
            lo = ['lo'+str(int(n*100)) for n in prob]
            qp_cols = labels + hi + lo

      if params is not None:
            infolabels = params + info
            return [qp_cols, infolabels]
      else:
            return [qp_cols]

def check_inits(inits={}, pdep=[], kind='radd', dynamic='hyp', single_bound_models=['xirace', 'irace', 'xpro', 'pro'], pro_ss=False, fit_noise=False):

      for k, val in inits.items():
            if isinstance(val, np.ndarray) and k not in pdep:
                  inits[k]=val[0]
      if 'ssd' in inits.keys():
            del inits['ssd']
      if 'pGo' in inits.keys():
            del inits['pGo']
      if kind in single_bound_models and 'z' in inits.keys():
            z=inits.pop('z')
            inits['a']=inits['a']-z

      if 'race' in kind:
            inits['ssv']=abs(inits['ssv'])
      elif 'radd' in kind:
            inits['ssv']=-abs(inits['ssv'])

      if 'pro' in kind:
            if pro_ss and 'ssv' not in inits.keys():
                  inits['ssv'] = -0.9976
            elif not pro_ss and 'ssv' in inits.keys():
                  ssv=inits.pop('ssv')

      if 'x' in kind and 'xb' not in inits.keys():
            if dynamic == 'exp':
                  inits['xb'] = 2
            elif dynamic == 'hyp':
                  inits['xb'] = .02

      if fit_noise and 'si' not in inits.keys():
            inits['si'] = .01

      return inits


def make_proRT_conds(data, split):

      if np.any(data['pGo'].values > 1):
            data['pGo']=data['pGo']*.01
      if np.any(data['rt'].values > 5):
            data['rt']=data['rt']*.001

      if split=='HL':
            data['HL']='x'
            data.ix[data.pGo>.5, 'HL']=1
            data.ix[data.pGo<=.5, 'HL']=2
      return data


def rename_bad_cols(data):

      if 'trial_type' in data.columns:
            data.rename(columns={'trial_type':'ttype'}, inplace=True)

      return data
