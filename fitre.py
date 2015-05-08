#!/usr/local/bin/env python
from __future__ import division
import numpy as np
import pandas as pd
from lmfit import Parameters, Minimizer
from radd_demo import utils, vis, RADD
from radd_demo.utils import *

def fit_reactive_data(data, inits={}, model='radd', depends=['v'], all_params=1, ntrials=2000, maxfun=5000, save_path="./", ftol=1.e-3, xtol=1.e-3, **kwargs):

	ssdlist=[200,250,300,350,400,'rt']
	sxlist=data.idx.unique()

	for k in inits.keys():
		if k=='pGo' or k=='ssd':
			del inits[k]

	poptdf=pd.DataFrame(columns=['a', 't', 'v', 'z', 'ssv'], index=sxlist)

	ssdlist=[200,250,300,350,400,'rt']
	sxpred_list=[]
	pb=utils.PBinJ(len(sxlist), color="#009B76"); sx_n=0

	for sx, sxdf in data.groupby('idx'):

		y=utils.rangl_data(sxdf)
		sx_n+=1

		#check if nested dictionary
		#(for when sx have unique init values)
		if type(inits.values()[0]) is dict:
			init_theta = inits[sx]
		else:
			init_theta = inits

		sxpred, popt=run_reactive_model(y, init_theta, depends=depends, ntrials=ntrials, all_params=all_params, maxfun=maxfun, ftol=ftol, xtol=xtol, model=model)
		sxpred['idx']=[sx]*len(sxpred)

		pb.update_i(sx_n)
		sxpred_list.append(sxpred)

		for k in popt.keys():
			poptdf.loc[sx, k]=popt[k]

		poptdf.to_csv(save_path+"popt_all.csv")
		sxpred.to_csv(save_path+"fits_sx"+str(sx)+".csv")

	if len(sxlist)>1:
		allpred = pd.concat(sxpred_list)
		allpred.to_csv(save_path+"allsx_fits.csv")
		return allpred

	else:
		return sxpred


def run_reactive_model(y, inits={}, model='radd', depends=['v'], ntrials=5000, maxfun=5000, ftol=1.e-3, xtol=1.e-3, all_params=1, ssdlist=[200,250,300,350,400,'rt'], **kwargs):

	p=Parameters()

	if all_params: vary=1
	else: vary=0

	for key, val in inits.items():
		if key in ['pGo', 'ssd'] or not all_params:
			vary=0
		else:
			vary=1
		if key in depends:
			p.add(key, value=val, vary=1)
		else:
			p.add(key, value=val, vary=vary)

	popt = Minimizer(ssre_minfunc, p, fcn_args=(y, ntrials),
		fcn_kws={'model':model}, method='Nelder-Mead')
	popt.fmin(maxfun=maxfun, ftol=ftol, xtol=xtol, full_output=True, disp=False)

	params=pd.Series({k:p[k].value for k in p.keys()})
	res=popt.residual
	res[-1]=res[-1]/10; y[-1]=y[-1]/10; yhat=y+res

	pred=pd.DataFrame.from_dict({'ssdlist':ssdlist, 'ydata':y, 'residuals':res,
		'yhat':yhat, 'chi':popt.chisqr}, orient='columns')

	return pred, params


def ssre_minfunc(p, ydata, ntrials, model='radd', tb=.650, intervar=False):

	theta={k:p[k].value for k in p.keys()}
	theta['pGo']=.5

	delays=np.arange(0.200, 0.450, .05)
	accvec=[]; rtvec=[]
	y=np.array(ydata)

	for ssd in delays:
		theta['ssd']=ssd
		sacc, grt = simulate(theta, ntrials, tb=tb, model=model, intervar=intervar)
		accvec.append(sacc)
		rtvec.append(grt)

	accvec.append(np.mean(rtvec))
	yhat=np.array(accvec)

	return yhat-y


def simulate(theta, ntrials=20000, tb=.650, model='radd', intervar=False, return_all=False, return_traces=False, **kwargs):

	if intervar:
		theta=utils.get_intervar_ranges(theta)

	dvg, dvs = RADD.run(theta, ntrials=ntrials, model=model, tb=tb)

	if return_traces:
		return dvg, dvs

	sacc, rt = analyze_reactive_trials(dvg, dvs, theta, model=model, return_all=return_all)

	if not return_all:
		rt=rt*10

	return sacc, rt


def analyze_reactive_trials(DVg, DVs, theta, model='radd', tb=.650, return_all=False):

	ngo_trials=len(DVg)
	nss_trials=len(DVs)

	theta=utils.update_params(theta)
	tr=theta['tt'];	a=theta['a']; ssd=theta['ssd']

	#get individual trial go and ss rt's
	gos_i = [tr + np.argmax(DVgn>=a)*.0005 if np.any(DVgn>=a) else 999 for i, DVgn in enumerate(DVg)]

        if model in ['radd', 'ipb']:
                stops_i = [ssd + np.argmax(DVsn<=0)*.0005 if np.any(DVsn<=0) else 999 for i, DVsn in enumerate(DVs)]
        else:
                stops_i = [ssd + np.argmax(DVsn>=a)*.0005 if np.any(DVsn>=a) else 999 for i, DVsn in enumerate(DVs)]

        #calculate stop accuracy
	stops = [1 if gos_i[i]>si else 0 for i, si in enumerate(stops_i)]
	sacc=np.mean(stops)

	#calculate mean go rt
	go_trial_array=np.array(gos_i)
	rt=np.mean(go_trial_array[go_trial_array<tb])

	if return_all:
		return stops, go_trial_array[go_trial_array<tb]
	else:
		return sacc, rt
