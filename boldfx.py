#!/usr/local/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from radd.utils import update_params

def probold_bias(theta, bias={}, pgo=np.arange(0, 1.25, .25), tb=.560, ntrials=150, return_all=False, visual=False, ssv_decay=False):

        alldf=[]
        for i, pg in enumerate(pgo):

                theta['pGo']=pg
                for k, v in bias.items():
			theta[k]=v[i]

                df=simulate(theta, model='radd', ntrials=ntrials, return_all=return_all,
                                simbold=True, ssv_decay=ssv_decay, visual=visual, tb=tb, bias=k)
                alldf.append(df)

        prosims=pd.concat(alldf)

        return prosims

def trialwise_integrated_BOLD(df=pd.DataFrame, outcomes=['s','g','ng']):

	thal=df[['thalamus', 'pGo', 'outcome']]
	dfdict={out: thal[thal['outcome']==out].copy() for out in outcomes}

        for dfi in dfdict.values():
		dfi['mag']=[np.cumsum(np.array(i))[-1] for i in dfi.thalamus.values]

        return dfdict


def simulate(theta, ntrials=2000, model='radd', tb=0.560, return_all=False, visual=False, intervar=False, simbold=False, ssv_decay=False, bias='v', filt_rts=True, **kwargs):

	"""

	Mostly deprecated, need to refactor as exclusive bold simulation function

	"""

	pStop=1-theta['pGo']

	if visual:
		columns=["rt","choice","acc", "response", "go_tsteps", "go_paths", "ss_tsteps",
			"ss_paths", "len_go_tsteps","len_ss_tsteps","trial_type", 'ssd', "pGo"]
		mfx = radd_visfx
	else:
		columns = ['rt', 'choice', 'response', 'acc', 'trial_type', 'ssd', 'pGo']
		mfx = radd_fitfx

	if simbold:
		mfx = boldfx.rise
		columns.append('thalamus')
		columns.append('outcome')
		#pStop=0

	df = pd.DataFrame(columns=columns, index=np.arange(0,ntrials))

	for i in range(ntrials):
		ttype='go'
		if np.random.random_sample()<=pStop:
			ttype='stop'
		sim_out = mfx(theta, model=model, tb=tb, ttype=ttype, ssv_decay=ssv_decay, bias=bias)
		df.loc[i]=pd.Series({c:sim_out[c] for c in df.columns})

	df[['rt', 'acc', 'response']]=df[['rt', 'acc', 'response']].astype(float)

	if visual:
		vis.plot_traces(df, theta=theta, pGo=theta['pGo'], ssd=.450, tb=tb, task='ssPro')

	if return_all:
		return df
	else:
		pstop=1-df['response'].mean()
		if filt_rts:
			rt=df.query("response==1 and rt<=.5451").mean()['rt']*10
		else:
			rt=df.query("response==1").mean()['rt']*10

		return pstop, rt


def rise(theta, model='radd', tb=.560, ttype='go', tau=.0005, si=.01, task='pro', ssv_decay=False, bias='v', **kwargs):

	if ttype=='stop':
		sstrial=1
	else:
		sstrial=0

	theta=update_params(theta)

	tr=theta['tr']; mu=theta['v'];
	a=theta['a']; z=theta['z'];
	ssd=theta['ssd']; ssv=-abs(theta['ssv'])

	a=a-z
	lower=-np.copy([z])[0]
	ssdecision = lambda x: x <= lower
	z=0

	if tr>ssd and sstrial:
		t=ssd # start the time at ssd
	else:
		t=tr  # start the time at tr

	dx=np.sqrt(si*tau)  		# dx is the step size
	e, ess, ithalamus = z, z, z	# starting point
	ss_started=False; no_choice_yet=True
	elist=[]; tlist=[]; elist_ss=[]; tlist_ss=[]; thalamus=[]

	p=0.5*(1 + mu*dx/si)
	p_ss=0.5*(1 + ssv*dx/si)

	choice = 'stop'
	outcome='ng'
	rt=tb
	response=0
	decayed=False

	while not decayed:

		t += tau

		if t>=tr:

			if t>=tb and no_choice_yet:
				choice='stop'
				rt=tb
				outcome='ng'
				no_choice_yet=False
				response=0
				decayed=True
				break

			# r is between 0 and 1
			r=np.random.random_sample()
			if r < p:
				e += dx
				ithalamus = ithalamus + dx
			else:
				e += -dx
				ithalamus = ithalamus - dx

			if e>=a and no_choice_yet:
				choice='go'
				outcome='g'
				rt=t
				no_choice_yet=False
				response=1
				decayed=True
				break

		if 're' in task and t>=ssd and sstrial and no_choice_yet:

			r_ss=np.random.random_sample()
			if r_ss < p_ss:
				ess+=dx
				ithalamus = ithalamus + dx
			else:
				ess+=-dx
				ithalamus = ithalamus - dx

			if ssdecision(ess) and no_choice_yet:
				choice='stop'
				outcome='s'
				no_choice_yet=False
				response=0;
				decayed=True
				break

		thalamus.append(ithalamus)

	thalamus=decay(thalamus, ithalamus, baseline=z, t=t, mu=mu, ssv=ssv, task=task, ssv_decay=ssv_decay, bias=bias)

	if choice==ttype:
		acc=1.00
	else:
		acc=0.00

	return {'rt':rt, 'choice':choice, 'response':response, 'thalamus':thalamus, 'trial_type':ttype,	'acc':acc, 'ssd':ssd, 'pGo':theta['pGo'], 'outcome':outcome}


def decay(thalamus, ithalamus, baseline, t, mu, ssv, decay_coeff=-1.6587, si=.01, tau=.0005, task='Pro', bias='v', ssv_decay=False):

	#1.6484
	dx=np.sqrt(si*tau)
	decay_coeff=-1.8; #1.96919679429205063
	if 'e' in task or bias=='tr':
		decay_coeff=-abs(mu)
		#ssv_decay=True

	#decay_coeff is the negative
	#average drift-rate across
	#probability of go conditions
	p=0.5*(1 + decay_coeff*dx/si)
	p_ss=0.5*(1 + ssv*dx/si)

	while t<.9:#ithalamus>baseline:
		if ithalamus<=baseline:
			break
		t+=tau

		r=np.random.random_sample()
		if r < p:
			ithalamus = ithalamus + dx
		else:
			ithalamus = ithalamus - dx

		if ssv_decay:
			r_ss=np.random.random_sample()
			if r_ss < p_ss:
				ithalamus = ithalamus + dx
			else:
				ithalamus = ithalamus - dx

		thalamus.append(ithalamus)

	return thalamus
