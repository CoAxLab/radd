#!/usr/local/bin/env python
from __future__ import division
import numpy as np
import pandas as pd
from lmfit import Parameters, Minimizer
from scipy.optimize import minimize
from radd.utils import *
from radd.cRADD import recost, recost_scipy
from radd import RADD

def fit_reactive_model(y, inits={}, depends=['xx'], wts=None, model='radd', ntrials=5000, maxfun=5000, ftol=1.e-3, xtol=1.e-3, all_params=0, disp=False):

      if 'pGo' in inits.keys(): del inits['pGo']
      if 'ssd' in inits.keys(): del inits['ssd']

      if model in ['radd', 'ipb', 'abias']:
            inits['ssv']=-abs(inits['ssv'])
      else:
            inits['ssv']=abs(inits['ssv'])

      if use_lmfit:
              p=Parameters()
              for key, val in inits.items():
                    if key in depends:
                          p.add(key, value=val, vary=1)
                          continue
                    p.add(key, value=val, vary=all_params)

              popt = Minimizer(recost, p, fcn_args=(y, ntrials), fcn_kws={'wts':wts}, method='Nelder-Mead')
              popt.fmin(maxfun=maxfun, ftol=ftol, xtol=xtol, full_output=True, disp=disp)
              params={k: p[k].value for k in p.keys()}
              params['chi']=popt.chisqr
              resid=popt.residual; yhat=y+resid

      else:
              a, tr, v, ssv, z  = theta['a'], theta['tr'], theta['v'], -abs(theta['ssv']),  theta['z']
              p=[a, tr, v, ssv, z]
              popt = minimize(recost_nb, p, args=(y, wts, ntrials), method='Nelder-Mead', options={'disp':True, 'xtol':xtol, 'ftol':ftol, 'maxfev': maxfun})
              a, tr, v, ssv, z  = theta['a'], theta['tr'], theta['v'], -abs(theta['ssv']),  theta['z']
              x0=[a, tr, v, ssv, z]
              fit = mina(recost_scipy, x0, args=(y, wts, ntrials), method='Nelder-Mead', options={'disp':True, 'xtol':xtol, 'ftol':ftol, 'maxfev': maxfun})

      return params, yhat


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
