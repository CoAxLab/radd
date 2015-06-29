#/usr/local/bin/env python
from __future__ import division
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
from lmfit import Parameters, Minimizer
from scipy.stats.mstats import mquantiles
from radd import utils, vis, boldfx, RADD

def fit_proactive_data(data, inits={}, est_global=True, depends_on={'v':'pGo'}, ntrials=2000, maxfun=5000, ftol=1.e-3, xtol=1.e-3, filt_rts=True, tb=.560, disp=False, rtfunc='mean', **kwargs):

	optall={name:inits[name] for name in pnames}
	dep=depends_on.keys()

	if 'ssv' in inits.keys(): del inits['ssv']
	inits['pGo']=.5; inits['ssd']=.450;

	if est_global:
		y = utils.rangl_pro(data)
		p=Parameters()
		for k, v in optall.items():
			if k in ['pGo', 'ssd']: vary=0
			else: vary=1
			p.add(k, value=v, min=0.0, vary=vary)

		popt = Minimizer(sspro_minfunc, p, fcn_args=(y, ntrials), fcn_kws={'rtfunc':rtfunc, 'filt_rts':filt_rts}, method='Nelder-Mead')
		popt.fmin(maxfun=maxfun, ftol=ftol, xtol=xtol, full_output=True, disp=disp)

		# globally optimized parameter set
		global_params={k:p[k].value for k in p.keys()}
		if hilo:
			global_params[dep+'(Lo)']=inits[dep+'(Lo)']; global_params[dep+'(Hi)']=inits[dep+'(Hi)']
		pd.Series(global_params).to_csv("global_opt_theta.csv")
	else:
		global_params=inits
	# fix optimized parameters and refit
	# allowing depends param to vary across P(Go)
	fits=pd.DataFrame({'psdata':pstop, 'rtdata': rt/10}, index=pgolist)
	fits_all, rt_sim, ps_sim = run_proactive_model(pstop, rt, inits=global_params, depends_on=depends_on, pnames=pnames, ntrials=ntrials, ftol=ftol, xtol=xtol, maxfun=maxfun, tb=tb, pgolist=pgolist, filt_rts=filt_rts)
	# frame and save fit results
	fits['psradd']=np.array(ps_sim)
	fits['rtradd']=np.array(rt_sim)

	fits.to_csv("flat_fits.csv")
	fits_all.to_csv("flat_fits_info.csv")
	dep_names=[depends_on.keys()[0] + str(pg) for pg in pgolist]

	popt=global_params.copy()
	for i, dn in enumerate(dep_names):
		popt[dn]=fits_all[dep].values[i]
	pd.Series(popt).to_csv("flat_opt_theta.csv")

	return fits, fits_all, pd.Series(popt)


def run_proactive_model(pstop, rt, inits={}, depends_on={'v':'pGo'}, ntrials=2000, nx=0, maxfun=5000, ftol=1.e-3, xtol=1.e-3, rtfunc='mean', disp=False, filt_rts=True, **kwargs):

	inits['ssd']=.450
	fit_results=pd.DataFrame(columns=['n','pGo', dep, 'psdata', 'psradd', 'rtdata', 'rtradd', 'chi', 'redchi'], index=pgo)

	for i, pg in enumerate(pgo):

		inits['pGo']=pg*.01
		dep_init = init_list[i]
		y=np.array([pstop[i], rt[i]])

		p=Parameters()
		for param in pnames:
			p.add(param, value=inits[param], vary=0)
		p.add(dep, value=dep_init, min=0.0, vary=1)

		popt = Minimizer(sspro_minfunc, p, fcn_args=(y, ntrials), fcn_kws={'rtfunc':rtfunc, 'filt_rts':filt_rts}, method='Nelder-Mead')
		popt.fmin(maxfun=maxfun, ftol=ftol, xtol=xtol, full_output=True, disp=disp)

		if rtfunc=="quants":
			rtsim=(y[1:]+popt.residual[1:])/10
			rtdata=y[1:]/10
		else:
			rtsim=(y[1]+popt.residual[1])/10
			rtdata=y[1]/10

		psim=y[0]+popt.residual[0]
		psdata=y[0]
		x2=popt.chisqr
		redx2=popt.redchi

		fit_results.loc[pg,:]=pd.Series(OrderedDict([('n',int(nx)),('pGo',int(pg)),
		(dep,p[dep].value), ('psdata',psdata), ('psradd',psim), ('rtdata',rtdata),
		('rtradd',rtsim), ('chi',x2), ('redchi',redx2)]))

		rts_i.append(rtsim); ps_i.append(psim)

	return fit_results, rts_i, ps_i


def sspro_minfunc(p, ydata, ntrials, theta={}, filt_rts=True, rtfunc='mean', tb=.560):

	theta={k:p[k].value for k in p.keys()}

	psim, rtsim = simulate(theta, ntrials=ntrials, filt_rts=filt_rts, rtfunc=rtfunc, tb=tb)

	for i, val in enumerate(ydata):
		if hasattr(val, 'append'):
			ydata[i]=val[0]

	pres=psim-ydata[0]

        if rtfunc=='quants':
        	rtres=rtsim-ydata[1:]
		if theta['pGo']==0.0:
			residuals=np.append(pres, np.zeros_like(rtres))
        	else:
			residuals=np.append(pres, rtres)
	else:
		rtres=rtsim-ydata[1]
		if theta['pGo']==0.0:
			residuals=np.array([pres, 0.00001])
		else:
			residuals=np.array([pres, rtres])

	pres=psim-ydata[0]

	return residuals


def simulate(theta, ntrials=2000, tb=.560, filt_rts=True, filtr=.5451, rtfunc='mean', intervar=False, **kwargs):

	if intervar:
		theta=fitfx.get_intervar_ranges(theta)

	dvg, dvs = RADD.run(theta, ntrials=ntrials)

	responses, rt = analyze_proactive_trials(dvg, dvs, theta, filt_rts=filt_rts, filtr=filtr, tb=tb, return_all=True)

	if rtfunc=='quants':
		rt=mquantiles(rt, prob=[.1,.3,.5,.7,.9])*10
	elif rtfunc=='mquant':
		rt=np.mean(mquantiles(rt, prob=[.1,.3,.5,.7,.9]))*10
	else:
		rt=np.mean(rt)*10

	pstop=1-np.mean(responses)

	return pstop, rt


def analyze_proactive_trials(DVg, DVs, theta, filt_rts=True, tb=.560, filtr=.5451, rtfunc='mean', return_all=False):

	theta=utils.update_params(theta)
	tr=theta['tt'];	a=theta['a']; ssd=theta['ssd']

	# what happened on go trials (with no stop signal occuring)
	responses=[1 if np.any(DVgn>=a) else 0 for DVgn in DVg]
	# what happened on trials with a stop signal
	ss_out=[0 if np.any(DVsn<=0) else 1 for DVsn in DVs]

	# get vector of trial outcomes (1:go, 0:nogo)
	# append 0s for any no-response trials
	# caused by stop-signal reaching 0 bound
	if len(ss_out)>0:
		responses=np.append(responses, ss_out[ss_out<1])
	pstop = 1 - np.mean(responses)

	#get vector of response times
	go_traces = [trace for trace in DVg if np.any(trace>=a)]
	rts = np.array([tr + np.argmax(trace>=a)*.0005 for trace in go_traces])
	if filt_rts:
		rts=rts[rts<filtr]
	#return endt
	if return_all:
		return responses, rts
	else:
		rt=np.mean(rts[rts<tb])

	return pstop, rt

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
		dvg, dvs = RADD.run(theta)
		# extract no-go probability and go RT from
		nogo, rt = analyze_proactive_trials(dvg, dvs, theta)
		nogo_list.append(nogo); rt_list.append(rt)

	return np.array(nogo_list), np.array(rt_list[1:])*1000
