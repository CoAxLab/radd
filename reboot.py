#!/usr/local/bin/env python
from __future__ import division
import numpy as np
import pandas as pd
from radd_demo import utils, fitre

def Reboot(data, inits, cond, depends=['v'], niter=150, ntrials=2000, method='rwr', model='radd', maxfun=500, ftol=1.e-3, xtol=1.e-3, savepth='./', **kwargs):

	prepend=savepth+cond+"_"+method+"_"+model
	# initialize data storage objects
	fit_results=list(); cols=[200, 250, 300, 350, 400, 'rt']
	y_pred=pd.DataFrame(columns=cols, index=np.arange(niter))
	poptdf=pd.DataFrame(columns=['a', 't', 'v', 'z', 'ssv'], index=np.arange(niter))

	for i in range(niter):

		if method=='subsample':
			yxdf=pd.DataFrame(columns=cols, index=data.subj_idx.unique())
			# sample proportionally given experimental trial counts
			# like generating synthetic sx THEN averging
			for nsample in yxdf.index.values:
				bxY = resample_reactive(data, method='subsample')
				yxdf.loc[nsample] = bxY
			sample_y=yxdf.mean().values

		elif method=='rwr':
			sample_y = resample_reactive(data, method='rwr')

		fits_i, params = fitre.run_reactive_model(sample_y, inits=inits, nx=i, ntrials=ntrials,
			model=model, depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol)

		for name in poptdf.columns:
			poptdf.loc[i, name]=params[name]

		fit_results.append(fits_i)
		y_pred.loc[i,:]=fits_i['yhat'].values
		y_pred.to_csv(prepend+"_fits"+".csv", index=False)

	fit_df=pd.concat(fit_results)
	fit_df.to_csv(prepend+"_fitinfo"+model+".csv")
	poptdf.to_csv(prepend+"_popt.csv")

	return y_pred

def resample_reactive(data, n=None, method='rwr'):

	bootlist=list()
	df=data.copy(); i=0

	if n==None and method=='rwr':
		nlist=[len(df)]*len(df.ssd.unique())
	elif n==None and method=='subsample':
		nlist=[len(ssdf) for ssd, ssdf in df.groupby('ssd')]

	for ssd, ssdf in df.groupby('ssd'):

		boots = ssdf.reset_index(drop=True)
		orig_ix=np.asarray(boots.index[:])

		resampled_ix=utils.rwr(orig_ix, get_index=True, n=nlist[i])
		bootdf=ssdf.irow(resampled_ix)
		bootlist.append(bootdf)
		i+=1

	#concatenate all resampled conditions
	bootdf=pd.concat(bootlist)
	sample_y=list(bootdf[bootdf['trial_type']=='stop'].groupby('ssd').mean()['acc'].values)
	mrt=bootdf[(bootdf['acc']==1)&(bootdf['response']==1)].groupby('ssd').mean()['rt'].mean()*10
	sample_y.append(mrt)

	return sample_y
