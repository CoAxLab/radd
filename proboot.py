#!/usr/local/bin/env python
from __future__ import division
import numpy as np
import pandas as pd
from radd import utils, fitpro


def Proboot(data, inits, niter=150, depends_on={'v':'pGo'}, save_path="./", simfx='vector', method='rwr', ntrials=2000, maxfun=500, ftol=1.e-3, xtol=1.e-3, tb=.560, disp=False, filt_rts=True, rtfunc='mean', **kwargs):

	if 'v' in depends_on.keys():
		model='radd'
	elif 't' in depends_on.keys():
		model='onset'

	fit_results=list()
	#dpstop, drt = utils.pstop_quants(data, filt_rts=filt_rts)
	#print "data pstop: ", dpstop
	#print "data rt: ", drt
	pgolist=data[depends_on.values()].unique()
	ps_pred=pd.DataFrame(columns=pgolist, index=np.arange(niter))
	rt_pred=pd.DataFrame(columns=pgolist, index=np.arange(niter))

	#tb_mu=np.array([ 0.53313075,  0.54314161,  0.54303763,  0.54248767,  0.53333002, 0.52904703])
	#tb_sd=np.array([ 0.04245078,  0.02256533,  0.0191511 ,  0.01667792,  0.02619839, 0.03068832])
	#tb=[utils.sample_tb(mu, sd) for mu, sd in zip(tb_mu, tb_sd)]

	for i in range(niter):

		elif method=='rwr':
			bx_data = resample_proactive(data, method='rwr', filt_rts=filt_rts)

		if rtfunc=='quants':
			pstop, rt = utils.pstop_quants(bx_data, filt_rts=filt_rts)
		elif rtfunc=='mquant':
			pstop, rt, = utils.pstop_mquant(bx_data, filt_rts=filt_rts)
		else:
			pstop, rt = utils.pstop_meanrt(bx_data, filt_rts=filt_rts)

		fits_i, rts_i, ps_i = fitpro.run_proactive_model(pstop, rt, inits, filt_rts=filt_rts,ntrials=ntrials, depends_on=depends_on, nx=i, simfx=simfx, tb=tb,maxfun=maxfun, ftol=ftol, xtol=xtol, pgolist=pgolist, disp=disp)

		ps_pred.loc[i,:]=np.array(ps_i)
		rt_pred.loc[i,:]=np.array(rts_i)
		fit_results.append(fits_i)

		ps_pred.to_csv(save_path+method+"_pro_pstop_"+model+"_"+rtfunc+".csv", index=False)
		rt_pred.to_csv(save_path+method+"_pro_rt_"+model+"_"+rtfunc+".csv", index=False)

	fit_df=pd.concat(fit_results)
	fit_df.to_csv(save_path+method+"_pro_fitinfo_"+model+"_"+rtfunc+".csv", index=False)

	return ps_pred, rt_pred, fit_df


def resample_proactive(df, n=None, method='rwr', filt_rts=True):

	pvec=list(); rtvec=list(); df=df.copy(); i=0
	bootdf_list=list()
	if n==None and method=='rwr':
		nlist=[len(df)]*6
	elif n==None and method=='subsample':
		nlist=[80]*5
		nlist.append(100)
	else:
		nlist=[len(df)]*6

	for pg, pgdf in df.groupby('pGo'):
		boots = pgdf.reset_index(drop=True)
		orig_ix=np.asarray(boots.index[:])
		resampled_ix=utils.rwr(orig_ix, get_index=True, n=nlist[i])
		bootdf=pgdf.irow(resampled_ix)
		bootdf_list.append(bootdf)
		i+=1

	return pd.concat(bootdf_list)
