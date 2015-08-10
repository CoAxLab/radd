#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os, re
from numpy import array


def get_default_inits(kind='radd', dynamic='hyp', depends_on={}, fit_whole_model=True, include_ss=False, fit_noise=False, inits = {}):

      if 'radd' in kind:
            if set(['v', 'tr']).issubset(depends_on.keys()):
                  inits = {'a':.45, 'ssv':-0.9473, 'tr': 0.2939, 'v': 1.0919, 'z':0.1542}
            elif ['v'] == depends_on.keys():
                  inits = {'a':0.4441, 'v':array([1.1078, 1.0651]), 'ssv':-0.9473, 'tr':0.3049, 'z':0.1542}
            elif ['tr'] == depends_on.keys():
                  inits = {'a': 0.4442, 'ssv': -0.9519, 'tr': array([.3027, 0.3104]), 'v': 1.0959, 'z':0.1541}
            elif ['a'] == depends_on.keys():
                  inits = {'a': array([0.43864876, 44879925]), 'tr': 0.3049, 'v': 1.0842, 'ssv': -0.9293, 'z': 0.1539}
            if 'x' in kind:
                  if dynamic=='hyp' and 'xb' not in inits:
                        inits['xb']=.01
                  elif dynamic=='exp' and 'xb' not in inits:
                        inits['xb']=1.5


      elif 'pro' in kind:
            if set(['v', 'tr']).issubset(depends_on.keys()):
                  inits = {'a':.39, 'tr': 0.2939, 'v': 1.0919}
            elif ['tr']==depends_on.keys():
                  inits = {'a':0.3267, 'tr': array([0.36803, 0.34669, 0.32555, 0.30535, 0.28855, 0.28082]), 'v': 1.3813}
            elif ['v']==depends_on.keys():
                  if 'x' not in kind:
                        inits = {'a': 0.474838, 'tr': 0.27253, 'v': array([ 1.39321,  1.52084,  1.65874,  1.75702,  1.89732,  1.94936]), 'z': 0}
                  else:
                        if dynamic=='hyp' and 'x' in kind:
                              inits = {'a':0.4748, 'tr':0.2725, 'v': array([ 1.39321, 1.52084, 1.65874,  1.75702, 1.89732, 1.94936]), 'z':0, 'xb': .009}
                        elif dynamic=='exp' and 'x' in kind:
                              inits = {'a':0.4836, 'tr': 0.3375, 'v': array([ 1.08837, 1.31837, 1.54837,  1.77837, 2.00837, 2.23837]), 'xb':  1.4604, 'z': 0}
            elif ['xb']==depends_on.keys():
                  inits={'a': 0.473022, "tr":0.330223, "v":1.64306, 'xb': array([0.257877, 0.649422, 1.03762, 1.307329, 1.934637, 2.101918])}

      elif 'race' in kind:
            if ['v']==depends_on.keys():
                  inits = {'v': array([1.0687, 1.0057]),'a': 0.3926741, 'tr': 0.3379, 'ssv': 1.1243, 'z': 0.1500}
            else:
                  inits = {'v': 1.0306, 'a': 0.3926741, 'tr': 0.3379, 'ssv': 1.1243, 'z': 0.1500}

      if 'x' in kind and 'xb' not in inits.keys():
            if dynamic=='hyp':
                  inits['xb']=.01
            elif dynamic=='exp':
                  inits['xb']=1.5

      if fit_whole_model and np.any(hasattr(inits, '__iter__')):
            for p in depends_on.keys():
                  inits[p]=np.mean(inits[p])

      return inits


def get_xbias_theta(model=None):
      """ hyperbolic and expon. simulation parameters
      """
      if model.dynamic=='hyp':

            return {'a': 0.47548, 'tr': 0.27264, 'v': array([1.38303, .53767, 1.6675, 1.7762, 1.967, 2.0506]), 'xb': 0.009}

            #return {'a': array([ 0.47302,  0.47302,  0.47302,  0.47302,  0.47302,  0.47302]),
            #      'tr': array([ 0.33022,  0.33022,  0.33022,  0.33022,  0.33022,  0.33022]),
            #      'v': array([ 0.69306,  0.81906,  0.94506,  1.07106,  1.19706,  1.32306]),
            #      'xb': array([ 0.01,  0.01,  0.01,  0.01,  0.01,  0.01]),
            #      'z': 0}

      elif model.dynamic=='exp':
            if 'v' in model.depends_on.keys():
                  return {'a': array([ 0.4836,  0.4836,  0.4836,  0.4836,  0.4836,  0.4836]),
                        'tr': array([ 0.3375,  0.3375,  0.3375,  0.3375,  0.3375,  0.3375]),
                        'v': array([ 1.08837,  1.31837,  1.54837,  1.77837,  2.00837,  2.23837]),
                        'xb': array([ 1.4604,  1.4604,  1.4604,  1.4604,  1.4604,  1.4604]),
                        'z': 0}
            elif 'tr' in model.depends_on.keys():
                  return {'a': array([ 0.39142,  0.39142,  0.39142,  0.39142,  0.39142,  0.39142]),
                        'tr': array([ 0.36599,  0.34991,  0.33453,  0.3239 ,  0.29896,  0.30856]),
                        'v': array([ 1.66214,  1.66214,  1.66214,  1.66214,  1.66214,  1.66214]),
                        'xb': array([ 0.09997,  0.09997,  0.09997,  0.09997,  0.09997,  0.09997]),
                        'z': 0}


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


def check_inits(inits={}, kind='radd', pdep=[], dynamic='hyp', pro_ss=False, fit_noise=False):

      single_bound_models = ['xirace', 'irace', 'xpro', 'pro']

      for k, val in inits.items():
            if isinstance(val, np.ndarray) and k not in pdep:
                  inits[k]=val[0]
      if 'ssd' in inits.keys():
            del inits['ssd']
      if 'pGo' in inits.keys():
            del inits['pGo']

      if pro_ss and 'ssv' not in inits.keys():
            inits['ssv'] = -0.9976

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



def logger(optmod, finfo, fit_id='FLAT', kind='radd', depends_on={}, ):

      finfo['chi'] = optmod.chisqr
      finfo['rchi'] = optmod.redchi
      finfo['CNVRG'] = optmod.pop('success')
      finfo['nfev'] = optmod.pop('nfev')
      finfo['AIC']= optmod.aic
      finfo['BIC']= optmod.bic

      pkeys = depends_on.keys()
      pvals = depends_on.values()
      model_id = "MODEL: %s" % kind
      dep_id = "%s DEPENDS ON %s" % (pvals[0], str(tuple(pkeys)))
      #wts_str = 'wts = array(['+ ', '.join(str(elem)[:6] for elem in self.simulator.wts)+'])'

      with open('fit_report.txt', 'a') as f:
            f.write(str(fit_id)+'\n')
            f.write(str(model_id)+'\n')
            f.write(str(dep_id)+'\n')
            #f.write(wts_str+'\n\n')
            f.write(fit_report(optmod, show_correl=False)+'\n\n')
            f.write('AIC: %.8f' % optmod.aic + '\n')
            f.write('BIC: %.8f' % optmod.bic + '\n')
            f.write('chi: %.8f' % optmod.chisqr + '\n')
            f.write('rchi: %.8f' % optmod.redchi + '\n')
            f.write('Converged: %s' % finfo['CNVRG'] + '\n')
            f.write('--'*20+'\n\n')

      return finfo
