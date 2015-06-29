#!/usr/local/bin/env python
from __future__ import division
import numpy as np
import pandas as pd
from radd import utils, fitre

def Reactive(data, inits, depends_on={'xx':'XX'}, fit='bootstrap', opt_global=False, ntrials=2000, maxfun=500, ftol=1.e-3, xtol=1.e-3, savepth='./', model='radd', niter=500):

	if global_opt:
		inits, yhat = run_reactive_model(y, inits=inits, ntrials=ntrials, model=model,
                    depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=1)

	# initialize data storage objects
	if fit=='bootstrap': indx = range(niter)
	elif fit=='subjects': indx = data.idx.unique()

	qplist = []; sclist; plist=[]
	for i, (c, cond_df) in enumerate(data.groupby(cond)):

		qp_df, pstop_df, popt_df = fit_indx(cond_df, inits, c, depends=depends_on.keys(), model=model, indx=indx, ntrials=ntrials, all_params=0, maxfun=maxfun, ftol=ftol, xtol=xtol)

		qplist.append(qp_df)
		sclist.append(pstop_df)
		plist.append(popt_df)

	qp_df.to_csv(savepth+model+"_qpfits.csv", index=False)
	pstop_df.to_csv(savepth+model+"_scurve.csv", index=False)
	popt_df.to_csv(savepth+model+"_boot_popt.csv")

	return qp_df, pstop_df, popt_df



def Proactive(data, inits, niter=150, depends_on={'v':'Cond'}, save_path="./", ntrials=2000, maxfun=500, ftol=1.e-3, xtol=1.e-3, tb=.560, disp=False, filt_rts=True, **kwargs):

	fit_results=list()
	pgolist=data[depends_on.values()].unique()
	ps_pred=pd.DataFrame(columns=pgolist, index=np.arange(niter))
	rt_pred=pd.DataFrame(columns=pgolist, index=np.arange(niter))

	prepend=savepth+model
	# initialize data storage objects
	cond = depends_on.values(); popt_cols = inits.keys()
	qp_cols = [cond, 'ttype', '5q', '25q', '50q', '75q', '95q', 'presp']
	pstop_cols = data.ssd.unique()

	fit_results=list();
	qp_df=pd.DataFrame(columns=qp_cols, index=np.arange(niter*2))
	pstop_df=pd.DataFrame(columns=pstop_cols, index=np.arange(niter))
	popt_df=pd.DataFrame(columns=popt_cols, index=np.arange(niter))

	for i in range(niter):

		bx_data = resample_proactive(data, method='rwr', filt_rts=filt_rts)

		fits_i, rts_i, ps_i = fitpro.run_proactive_model(pstop, rt, inits, filt_rts=filt_rts, ntrials=ntrials, depends_on=depends_on, nx=i, simfx=simfx, tb=tb,maxfun=maxfun, ftol=ftol, xtol=xtol, pgolist=pgolist, disp=disp)

		ps_pred.loc[i,:]=np.array(ps_i)
		rt_pred.loc[i,:]=np.array(rts_i)
		fit_results.append(fits_i)


		rt_pred.to_csv(save_path+method+"_pro_qpfits.csv", index=False)
		ps_pred.to_csv(save_path+method+"_pro_scurve.csv", index=False)


	fit_df=pd.concat(fit_results)
	fit_df.to_csv(save_path+method+"_pro_fitinfo.csv", index=False)

	return ps_pred, rt_pred, fit_df
