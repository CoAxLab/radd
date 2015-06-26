#!/usr/local/bin/env python
from __future__ import division
import numpy as np
import pandas as pd
from lmfit import Parameters, Minimizer
from radd.utils import *
from radd import RADD

def fit_reactive_data(data, inits={}, model='radd', depends=['v'], all_params=0, ntrials=2000, maxfun=5000, save_path="./", ftol=1.e-3, xtol=1.e-3, **kwargs):

	ssdlist=[200,250,300,350,400,'rt']
	sxlist=data.idx.unique()

	for k in inits.keys():
		if k=='pGo' or k=='ssd':
			del inits[k]

	poptdf=pd.DataFrame(columns=['a', 't', 'v', 'z', 'ssv'], index=sxlist)

	ssdlist=[200,250,300,350,400,'rt']
	sxpred_list=[]
	#pb=utils.PBinJ(len(sxlist), color="#009B76"); sx_n=0

	for sx, sxdf in data.groupby('idx'):

		y=rangl_data(sxdf)
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


def run_reactive_model(y, inits={}, model='radd', depends=['v'], ntrials=5000, maxfun=5000, ftol=1.e-3, xtol=1.e-3, all_params=0, ssdlist=[200,250,300,350,400,'rt'], **kwargs):

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


def ssre_minfunc(p, y, ntrials, model='radd', tb=.650, intervar=False):

	try:
		theta={k:p[k].value for k in p.keys()}
	except Exception:
		theta=p.copy()
	theta['pGo']=.5

	delays=np.arange(0.200, 0.450, .05)
	dflist=[]
	#y=np.array(ydata)

	for ssd in delays:
		theta['ssd']=ssd
		df = simulate(theta, ntrials, tb=tb, model=model, intervar=intervar)
		dflist.append(df)
		#accvec.append(sacc)
		#rtvec.append(grt)
	#accvec.append(np.mean(rtvec))
	#yhat=np.array(accvec)

	return rangl_data(pd.concat(dflist))-y


def simulate(theta, ntrials=20000, tb=.650, model='radd', quant=True, intervar=False, return_traces=False):

	if intervar:
		theta=get_intervar_ranges(theta)

	dvg, dvs = RADD.run(theta, ntrials=ntrials, model=model, tb=tb)

	if return_traces:
		return dvg, dvs

	df = analyze_reactive_trials(dvg, dvs, theta, model=model)
	
	if not quant:
		sacc = df.query('trial_type=="stop"').acc.mean()
		rt = df.query('response==1 and acc==1').rt.mean()*10
		return sacc, rt
	else:
		 return df

def analyze_reactive_trials(DVg, DVs, theta, model='radd', tb=.650):

	ngo=len(DVg)
	nss=len(DVs)
	
	theta=update_params(theta)
	tr=theta['tt'];	a=theta['a']; ssd=theta['ssd']

	#check for and record go trial RTs
	grt = np.array([tr + np.argmax(DVgn>=a)*.0005 if np.any(DVgn>=a) else 999 for DVgn in DVg[nss:, :]])	
	if model=='abias':
		ab=a+theta['ab']
		delay = np.ceil((tb-tr)/.0005) - np.ceil((tb-ssd)/.0005)		
		#check for and record SS-Respond RTs that occur before boundary shift
		ert_pre = np.array([tr + np.argmax(vec>=a)*.0005 if np.any(vec>=a) else 999 for vec in DVg[:nss, :delay]])
		#check for and record SS-Respond RTs that occur POST boundary shift (t=ssd)
		ert_post = np.array([tr + np.argmax(vec>=ab)*.0005 if np.any(vec>=ab) else 999 for vec in DVg[:nss, delay:]])
		#SS-Respond RT equals the smallest value
		ert = np.fmin(ert_pre, ert_post)
	else:
		#check for and record SS-Respond RTs
		ert = np.array([tr + np.argmax(DVgn>=a)*.0005 if np.any(DVgn>=a) else 999 for DVgn in DVg[:nss, :]])
		
        if model in ['radd', 'ipb','abias']:
                ssrt = [ssd + np.argmax(DVsn<=0)*.0005 if np.any(DVsn<=0) else 999 for i, DVsn in enumerate(DVs)]
        else:
                ssrt = [ssd + np.argmax(DVsn>=a)*.0005 if np.any(DVsn>=a) else 999 for i, DVsn in enumerate(DVs)]
	
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
	d = {'response':response, 'choice':choice, 'rt':rt, 'ssd':ssdlist, 'trial_type':ttypes}
	df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.iteritems()]))
	# calculate accuracy for both trial_types
	df['acc'] = np.where(df['trial_type']==df['choice'], 1, 0)
	return df

def simple_resim(theta, ssdlist=range(200, 450, 50), ntrials=2000, return_all=True):
	ssdlist = np.array(ssdlist)*.001
	dflist = []
	theta['pGo']=.5
	for ssd in ssdlist:
		theta['ssd'] = ssd
		dvg, dvs = RADD.run(theta, tb=.650, ntrials=ntrials)
		df = analyze_reactive_trials(dvg, dvs, theta)
		dflist.append(df)
	
	return pd.concat(dflist)
