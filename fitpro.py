#/usr/local/bin/env python
from __future__ import division
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
from lmfit import Parameters, Minimizer
from scipy.stats.mstats import mquantiles
from radd import utils, RADD

def fit_proactive_model(y, inits={}, depends=['xx'], model='radd', SSD=.450, tb=.560, rt_cutoff=.54502, ntrials=2000, maxfev=500, ftol=1.e-3, xtol=1.e-3, all_params=0, disp=False):

	if 'pGo' in inits.keys():
		PGO = inits['pGo']
		del inits['pGo']
	if 'ssd' in inits.keys():
		SSD = inits['ssd']
		del inits['ssd']
	if 'ssv' not in inits.keys():
		inits['ssv'] = -1.

	p=Parameters()
	for key, val in inits.items():
		if key in depends:
			p.add(key, value=val, vary=1)
			continue
		p.add(key, value=val, vary=all_params)

	popt = Minimizer(sspro_minfunc, p, fcn_args=(y, ntrials, PGO, SSD), fcn_kws={'model':model, 'tb':tb, 'rt_cutoff':rt_cutoff}, method='Nelder-Mead')
	popt.fmin(maxfev=maxfev, ftol=ftol, xtol=xtol, full_output=True, disp=disp)
	params={k: p[k].value for k in p.keys()}
	params['chi']=popt.chisqr
	resid=popt.residual; yhat=y+resid

	return params, yhat

def sspro_minfunc(p, y, ntrials, PGO, SSD, model='radd', tb=.560, rt_cutoff=.54502):

	try:
		theta={k:p[k].value for k in p.keys()}
	except Exception:
		theta=p.copy()

	theta['pGo']=PGO; theta['ssd']=SSD

	dvg, dvs = RADD.run(theta, ntrials=ntrials, tb=tb)
	simdf = gen_prosim_df(dvg, theta, tb=tb)
	yhat = utils.rangl_pro(simdf, tb=tb, rt_cutoff=rt_cutoff)

	return y-yhat

def gen_prosim_df(DVg, theta, tb=.560, dt=.0005):

	tr=theta['tr']; a=theta['a']; ssd=.450

	upper_rt = lambda x, DV: np.array([tr + np.argmax(DVi>=x)*dt if np.any(DVi>=x) else 999 for DVi in DV])

	#get vector of response times
	rt = upper_rt(a, DVg)
	response = np.where(rt<=tb, 1, 0)
	choice = np.where(response==1, 'go', 'stop')

	simdf = pd.DataFrame({'response':response, 'choice':choice, 'rt':rt})
	simdf.rt.replace(999, np.nan, inplace=True)
	return simdf


def simple_prosim(theta, bias_vals, bias='v', pgo=np.arange(0, 1.2, .2)):

	nogo_list, rt_list = [], []
	for pg, val in zip(pgo, bias_vals):
		#update P(Go)
		theta['pGo'] = pg
		#update bias param
		theta[bias] = val

		# run simulation, outputs simulated go-nogo (dvg) and
		# stop (dvs) process vectors for each trial (n=2000)
		# NOTE: in the proactive task SSD occurs too late (450ms)
		# for the stop-process to affect model output
		dvg = RADD.run(theta, ntrials=ntrials, tb=tb, kind='proactive')
		# extract no-go probability and go RT from
		nogo, rt = gen_prosim_df(dvg, theta, tb=tb, rt_cutoff=rt_cutoff)
		nogo_list.append(nogo); rt_list.append(rt)

	return np.array(nogo_list), np.array(rt_list[1:])*1000
