#!/usr/local/bin/env python
from __future__ import division
import time
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize, fit_report
from radd.utils import *
from radd.cRADD import recost
from radd import RADD

def fit_reactive_model(y, inits={}, depends=['xx'], wts=None, model='radd', ntrials=5000, maxfev=5000, ftol=1.e-3, xtol=1.e-3, all_params=0, disp=False, fitid=None, method='nelder'):

      ip = inits.copy()
      lim = {'tr': (.001, .5), 'v': (.01, 4.),  'ssv': (-4., -.01),  'a' : (.01, .6), 'z':(.001, .9)}
      if 'pGo' in ip.keys(): del ip['pGo']
      if 'ssd' in ip.keys(): del ip['ssd']
      ip['ssv']=-abs(ip['ssv'])

      p=Parameters()
      if not all_params:
            d0 = [p.add(d, value=ip.pop(d), vary=1, min=lim[d][0], max=lim[d][1]) for d in depends]
            vary=0
      else:
            vary=1

      if method=='differential_evolution':
            pass
      else:
            aval = ip.pop('a'); zval = ip.pop('z')
            p.add('a', value=aval, vary=1, min=lim['a'][0], max=lim['a'][1])
            p.add('zperc', value=zval/aval, vary=1)
            p.add('z', expr="zperc*a")

      p0 = [p.add(k, value=v, vary=vary, min=lim[k][0], max=lim[k][1]) for k, v in ip.items()]

      popt = minimize(recost, p, args=(y, ntrials), method=method, kws={'wts':wts}, options={'disp':disp, 'xtol':xtol, 'ftol':ftol, 'maxfev':maxfev})

      params = p.valuesdict()
      params['chi']=popt.chisqr
      params['rchi']=popt.redchi
      params['AIC']=popt.aic
      params['BIC']=popt.bic
      yhat = y + popt.residual

      if fitid is None:
            fitid = time.strftime('%H:%M:%S')

      with open('fit_report.txt', 'a') as f:
            f.write(str(fitid)+'\n')
            f.write(fit_report(popt, show_correl=False)+'\n')
            f.write('AIC: %.8f' % popt.aic)
            f.write('BIC: %.8f' % popt.bic)
            f.write('--'*20+'\n\n')
      return params, yhat


def rangl_re(data, cutoff=.650, prob=np.array([.1, .3, .5, .7, .9])):

	gac = data.query('trial_type=="go"').acc.mean()
	sacc = data.query('trial_type=="stop"').groupby('ssd').mean()['acc'].values

	grt = data.query('trial_type=="go" & acc==1').rt.values
	ert = data.query('response==1 & acc==0').rt.values
	gq = mq(grt, prob=prob)
	eq = mq(ert, prob=prob)

	return np.hstack([gac, sacc, gq*10, eq*10]).astype(np.float32)


def ssre_minfunc(p, y, ntrials=2000, model='radd', tb=.650, dflist=[]):

	try:
		theta={k:p[k].value for k in p.keys()}
	except Exception:
		theta=p.copy()
	theta['pGo']=.5; delays=np.arange(0.200, 0.450, .05)

	for ssd in delays:
            theta['ssd']=ssd
            dvg, dvs = RADD.run(theta, ntrials=ntrials, model=model, tb=tb)
            simdf = gen_resim_df(dvg, dvs, theta, tb=tb, model=model)
            dflist.append(simdf)

	yhat = rangl_re(pd.concat(dflist))

	return yhat - y


def gen_resim_df(DVg, DVs, theta, model='radd', tb=.650, dt=.0005):

      """
      Takes Go (DVg) and Stop (DVs) decision traces from RADD.run() and
      Generates a pandas DF with same naming conventions as data
      (#TODO : See docs).
      """


      nss=len(DVs)
      ngo=len(DVg) - nss
      tr=theta['tr']; a=theta['a']; ssd=theta['ssd']
      #define RT functions for upper and lower bound processes
      upper_rt = lambda x, DV: np.array([tr + np.argmax(DVi>=x)*dt if np.any(DVi>=x) else 999 for DVi in DV])
      lower_rt = lambda DV: np.array([ssd + np.argmax(DVi<=0)*dt if np.any(DVi<=0) else 999 for DVi in DV])
      #check for and record go trial RTs
      grt = upper_rt(a, DVg[nss:, :])
      if model=='abias':
            ab=a+theta['ab']
            delay = np.ceil((tb-tr)/dt) - np.ceil((tb-ssd)/dt)
            #check for and record SS-Respond RTs that occur before boundary shift
            ert_pre = upper_rt(a, DVg[:nss, :delay])
            #check for and record SS-Respond RTs that occur POST boundary shift (t=ssd)
            ert_post = upper_rt(ab, DVg[:nss, delay:])
            #SS-Respond RT equals the smallest value
            ert = np.fmin(ert_pre, ert_post)
      else:
            #check for and record SS-Respond RTs
            ert = upper_rt(a, DVg[:nss, :])

      if model=='ipa':
            ssrt = upper_rt(a, DVs)
      else:
            ssrt = lower_rt(DVs)

      # Prepare and return simulations df
      # Compare trialwise SS-Respond RT and SSRT to determine outcome (i.e. race winner)
      stop = np.array([1 if ert[i]>si else 0 for i, si in enumerate(ssrt)])
      response = np.append(np.abs(1-stop), np.where(grt<tb, 1, 0))
      # Add "choice" column to pad for error in later stages
      choice=np.where(response==1, 'go', 'stop')
      # Condition
      ssdlist = [int(ssd*1000)]*nss+[1000]*ngo
      ttypes=['stop']*nss+['go']*ngo
      # Take the shorter of the ert and ssrt list, concatenate with grt
      rt=np.append(np.fmin(ert,ssrt), grt)

      simdf = pd.DataFrame({'response':response, 'choice':choice, 'rt':rt, 'ssd':ssdlist, 'trial_type':ttypes})
      # calculate accuracy for both trial_types
      simdf['acc'] = np.where(simdf['trial_type']==simdf['choice'], 1, 0)
      simdf.rt.replace(999, np.nan, inplace=True)
      return simdf


def simple_resim(theta, ssdlist=range(200, 450, 50), ntrials=2000):

      """
      wrapper for simulating ntrials with RADD.run()
      for a range of SSDs in a reactive stopping task

      outputs a dataframe with same structure as data
      (#TODO : see docs) for further analysis
      """

      ssdlist = np.array(ssdlist)*.001
      dflist = []
      theta['pGo']=.5
      for ssd in ssdlist:
            theta['ssd'] = ssd
            dvg, dvs = RADD.run(theta, tb=.650, ntrials=ntrials)
            df = gen_resim_df(dvg, dvs, theta)
            dflist.append(df)

      return pd.concat(dflist)
