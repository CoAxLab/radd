#!/usr/local/bin/env python
from __future__ import division
import os
import pandas as pd
from radd_demo import utils
import numpy as np
from lmfit import Parameters, Minimizer


def radd_fitfx(theta, model='radd', tb=.560, ttype='go', tau=.0005, si=.01, **kwargs):

	if ttype=='stop':sstrial=1
	else: sstrial=0

	theta=utils.update_params(theta)

	tr=theta['tt']; mu=theta['vv'];
	a=theta['a']; z=theta['zz'];
	ssd=theta['ssd']

	if tr>ssd and sstrial:
		t=ssd # start the time at ssd
	else:
		t=tr  # start the time at tr

	if model=='ipa':
		a=a-z; z=0
		ssdecision = lambda x: x >= a
		ssv=abs(theta['ssv'])

	elif model in ['ipb', 'radd']:
		ssdecision = lambda x: x <= 0
		ssv=-abs(theta['ssv'])

	ss_started=False
	no_choice_yet=True
	dx=np.sqrt(si*tau)
	e=z
	e_ss=z
	response=0

	p=0.5*(1 + mu*dx/si)
	p_ss=0.5*(1 + ssv*dx/si)

	while no_choice_yet:

		t += tau
		if t>=tb and no_choice_yet:
			choice='stop'
			rt=tb
			no_choice_yet=False

		if t>=tr:
			r=np.random.random_sample()
			if r < p:
				e = e + dx
			else:
				e = e - dx
			if e>=a and no_choice_yet:
				choice='go'
				response=1
				rt=t
				no_choice_yet=False

		if sstrial and t>=ssd:
			r_ss=np.random.random_sample()
			if not ss_started and model=='radd':
				ss_started=True
				e_ss=e
			else:
				if r_ss < p_ss:
					e_ss = e_ss + dx
				else:
					e_ss = e_ss - dx

			if ssdecision(e_ss) and no_choice_yet:
				choice='stop'
				rt=t
				no_choice_yet=False
				response=0
	if choice==ttype:
		acc=1.00
	else:
		acc=0.00

	return {'rt':rt, 'choice':choice, 'trial_type':ttype,
		'acc':acc, 'ssd':ssd, 'pGo':theta['pGo'], 'response':response}
