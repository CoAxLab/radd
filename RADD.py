#!/usr/local/bin/env python
from __future__ import division
import os
from radd.utils import update_params
import numpy as np


def run(theta, ntrials=2000, tb=0.560, tau=.0005, si=.01, model='radd'):

	"""
	DVg is instantiated for all trials. DVs contains traces for a subset of those trials in which a SS occurs (proportional to pGo provided in theta).

	"""

	theta=update_params(theta)
	tr=theta['tt']; mu=theta['vv']; a=theta['a'];
	z=theta['zz']; ssd=theta['ssd']

	if model in ['radd', 'ipb', 'abias']:
		ssv=-abs(theta['ssv'])
        else:
                ssv=abs(theta['ssv'])

	if 'si' in theta.keys():
		si=theta['si']

	nSS=int(ntrials*(1-theta['pGo']))

	dx=np.sqrt(si*tau)
	Pg=0.5*(1 + mu*dx/si)
	Ps=0.5*(1 + ssv*dx/si)

	Tg=np.ceil((tb-tr)/tau)
	Ts=np.ceil((tb-ssd)/tau)

	#generate random probability vectors [nGo Trials x nTimesteps TR -> TB]
	trials=np.random.random_sample((ntrials, Tg))
	#simulate all go signal paths
	DVg = z + np.cumsum(np.where(trials<Pg, dx, -dx), axis=1)

	if tr<ssd and model in ['abias', 'radd']:
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
