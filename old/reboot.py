#!/usr/local/bin/env python
from __future__ import division
import numpy as np
import pandas as pd
from radd import utils, fitre

def Reactive(data, inits, depends_on={'xx':'XX'}, fit='boot', opt_global=False, nt=2000, mf=500, ftol=1.e-3, xtol=1.e-3, savepth='./', model='radd', niter=500):

	if opt_global:
		y = utils.rangl_re(data)
		out, inits = fitre.run_reactive_model(y, inits, model=model, all_params=1, depends=depends_on.keys(), ntrials=nt, maxfev=mf, ftol=ftol, xtol=xtol)

	# initialize data storage objects
	if fit=='bootstrap':
		indx = niter
	elif fit=='subject':
		indx= len(data.idx.unique())

	cond = depends_on.values(); nc=len(data[cond].unique());
	qp_cols = ['ttype', '5q', '25q', '50q', '75q', '95q', 'presp']
	qp_df=pd.DataFrame(columns=qp_cols, index=np.arange(indx*2*nc))
	pstop_df=pd.DataFrame(columns=data.ssd.unique(), index=np.arange(indx*nc))
	popt_df=pd.DataFrame(columns=inits.keys(), index=np.arange(indx*nc))

	for i, (c, cdf) in enumerate(data.groupby(cond)):
		qp_df, pstop_df, popt_df, fit_info = fit_indx(cdf, inits, qp_df, pstop_df, popt_df, depends=depends_on.keys(), model=model, niter=indx, ntrials=ntrials, all_params=0, maxfev=maxfev, ftol=ftol, xtol=xtol)


	#qp_df.to_csv(prepend+"_qpfits.csv", index=False)
	#pstop_df.to_csv(prepend+"_scurve.csv", index=False)
	#popt_df.to_csv(prepend+"_boot_popt.csv")

def resample_reactive(data, n=None):

	df=data.copy(); bootlist=list()
	if n==None: n=len(data)

	for ssd, ssdf in df.groupby('ssd'):
		boots = ssdf.reset_index(drop=True)
		orig_ix = np.asarray(boots.index[:])
		resampled_ix = rwr(orig_ix, get_index=True, n=n)
		bootdf = ssdf.irow(resampled_ix)
		bootlist.append(bootdf)

	#concatenate and return all resampled conditions
	return rangl_re(pd.concat(bootlist))
