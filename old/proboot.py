#!/usr/local/bin/env python
from __future__ import division
import numpy as np
import pandas as pd
from radd import utils, fitpro


def Proboot(data, inits, niter=150, depends_on={'v':'Cond'}, save_path="./", ntrials=2000, maxfev=500, ftol=1.e-3, xtol=1.e-3, tb=.560, disp=False, filt_rts=True, **kwargs):

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

		fits_i, rts_i, ps_i = fitpro.run_proactive_model(pstop, rt, inits, filt_rts=filt_rts,					ntrials=ntrials, depends_on=depends_on, nx=i, simfx=simfx, tb=tb,maxfev=maxfev, ftol=ftol, xtol=xtol, pgolist=pgolist, disp=disp)

		ps_pred.loc[i,:]=(ps_i)
		rt_pred.loc[i,:]=(rts_i)
		fit_results.append(fits_i)


		rt_pred.to_csv(save_path+method+"_pro_qpfits.csv", index=False)
		ps_pred.to_csv(save_path+method+"_pro_scurve.csv", index=False)


	fit_df=pd.concat(fit_results)
	fit_df.to_csv(save_path+method+"_pro_fitinfo.csv", index=False)

	return ps_pred, rt_pred, fit_df


def resample_proactive(df, n=None, method='rwr', filt_rts=True):

	pvec=list(); rtvec=list(); df=df.copy(); i=0
	bootdf_list=list()
	if n==None: nlist=[len(df)]*6

	for pg, pgdf in df.groupby('pGo'):
		boots = pgdf.reset_index(drop=True)
		orig_ix = np.asarray(boots.index[:])
		resampled_ix = rwr(orig_ix, get_index=True, n=nlist[i])
		bootdf=pgdf.irow(resampled_ix)
		bootdf_list.append(bootdf)
		i+=1

	return pd.concat(bootdf_list)
