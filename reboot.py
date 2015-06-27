#!/usr/local/bin/env python
from __future__ import division
import numpy as np
import pandas as pd
from radd import utils, fitre

def Reboot(data, inits, depends_on={'v':'Cond'}, niter=150, ntrials=2000, all_params=0, model='radd', maxfun=500, ftol=1.e-3, xtol=1.e-3, savepth='./', **kwargs):

	#TODO: need to create a parent function that runs Reboot (&run_reactive_model)
	#	for multiple conditions (i.e. bsl/pnl), similar to how proactive
	#	parameter dependencies are handled
	#		* liberal-ish optimization of all parameters collapsing across conditions
	#		* re-run allowing only depends_on keys to vary
	#	RELEVANT OBJECTS:
	#		* depends
	#		* all_params ()


	prepend=savepth+model
	# initialize data storage objects
	poptdf=pd.DataFrame(columns=inits.keys(), index=sxlist)
	cond = depends_on.values()
	qp_cols = [cond, 'ttype', '5q', '25q', '50q', '75q', '95q', 'presp']
	pstop_cols = data.ssd.unique()
	np.array([gq*10, wcor, eq*10, werr, pstop]).flatten()
	'measure', 'resptype', depends_on.values()

	fit_results=list();
	qpdf=pd.DataFrame(columns=cols, index=np.arange(niter*2))
	pstopdf=pd.DataFrame(columns=cols, index=np.arange(niter))
	poptdf=pd.DataFrame(columns=inits.keys(), index=np.arange(niter))

	for i in range(niter):

		sample_y = resample_reactive(data)

		outi, params = fitre.run_reactive_model(sample_y, inits=inits, nx=i, ntrials=ntrials,
			model=model, depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=all_params)

		for name in poptdf.columns:
			poptdf.loc[i, name]=params[name]

		fit_results.append(outi)

		yhat = outi['yhat']
		# fill df: [cond] [go/ssgo], [cor/err rt quantiles], [prob corr/err response]
		qpdf.loc[i,'ttype':'presp'] = [cond, 'go', yhat[:5], yhat[9]+.05*yhat[9]]
		qpdf.loc[i+1,'ttype':'presp'] = [cond, 'ssgo', yhat[10:15], yhat[19]+.05*yhat[19]]
		pstopdf.loc[i, :] = yhat[20:25]]

		qpdf.to_csv(prepend+"_qpfits.csv", index=False)
		pstopdf.to_csv(prepend+"_scurve.csv", index=False)

	fit_df=pd.concat(fit_results)
	fit_df.to_csv(prepend+"_boot_info.csv")
	poptdf.to_csv(prepend+"_boot_popt.csv")

	return y_pred

def resample_reactive(data, n=None):

	bootlist=list()
	df=data.copy()

	if n==None:
		n=len(data)
	for ssd, ssdf in df.groupby('ssd'):

		boots = ssdf.reset_index(drop=True)
		orig_ix=np.asarray(boots.index[:])

		resampled_ix=utils.rwr(orig_ix, get_index=True, n=n)
		bootdf=ssdf.irow(resampled_ix)
		bootlist.append(bootdf)

	#concatenate all resampled conditions
	bootdf=pd.concat(bootlist)
	return utils.rangl_data(bootdf, quant=True)
