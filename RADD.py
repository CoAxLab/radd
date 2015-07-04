#!/usr/local/bin/env python
from __future__ import division
import os
from radd.utils import update_params
import numpy as np


def run(theta, ntrials=2000, tb=0.650, tau=.0005, si=.01, model='radd', kind='reactive'):

	"""
	DVg is instantiated for all trials. DVs contains traces for a subset of those trials in which a SS occurs (proportional to pGo provided in theta).

	"""

	tr=theta['tr']; mu=theta['v']; a=theta['a'];
	z=theta['z']; ssd=theta['ssd'];
	if model=='ipa':
		ssv=abs(theta['ssv'])
	else:
		ssv=-abs(theta['ssv'])

	dx=np.sqrt(si*tau)

	Pg=0.5*(1 + mu*dx/si)
	Tg=np.ceil((tb-tr)/tau)

	#generate random probability vectors [nGo Trials x nTimesteps TR -> TB]
	trials=np.random.random_sample((ntrials, Tg))
	#simulate all go signal paths
	DVg = z + np.cumsum(np.where(trials<Pg, dx, -dx), axis=1)

	if kind=='proactive':
		return DVg, np.array([999])

	nSS=int(ntrials*(1-theta['pGo']))
	Ps=0.5*(1 + ssv*dx/si)
	Ts=np.ceil((tb-ssd)/tau)

	if tr<ssd:
		IXs = Tg - Ts
		init_ss = DVg[:nSS, IXs]
		init_ss[init_ss>a] = np.nan
	else:
		init_ss = np.array([z]*nSS)

	#generate random probability vectors [nSS Trials x nTimesteps SSD -> TB]
	sstrials = np.random.random_sample((nSS, Ts))
	#simulate all ss signal paths
	DVs = init_ss[:, None] + np.cumsum(np.where(sstrials<Ps, dx, -dx), axis=1)

	return DVg, DVs
