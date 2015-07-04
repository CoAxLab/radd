#!/usr/local/bin/env python
from __future__ import division
#from radd.utils import update_params
from numpy import cumsum, where, ceil, array, sqrt, nan
from numpy.random import random_sample as rs

def run(theta, ntrials=2000, tb=0.650, tau=.0005, si=.01, model='radd', kind='reactive'):

	"""
	DVg is instantiated for all trials. DVs contains traces for a subset of
	those trials in which a SS occurs (proportional to pGo provided in theta).
	"""

	tr=theta['tr']; mu=theta['v']; a=theta['a'];
	z=theta['z']; ssd=theta['ssd'];
	if model=='ipa':
		ssv=abs(theta['ssv'])
	else:
		ssv=-abs(theta['ssv'])

	dx = sqrt(si*tau)

	Pg = 0.5*(1 + mu*dx/si)
	Tg = ceil((tb-tr)/tau)

	#generate random probability vectors [nGo Trials x nTimesteps TR -> TB]
	trials = rs((ntrials, Tg))
	#simulate all go signal paths
	DVg = z + cumsum(where(trials<Pg, dx, -dx), axis=1)

	if kind=='proactive':
		return DVg, array([999])

	nSS = int(ntrials*(1-theta['pGo']))
	Ps = 0.5*(1 + ssv*dx/si)
	Ts = ceil((tb-ssd)/tau)

	if tr<ssd:
		IXs = Tg - Ts
		init_ss = DVg[:nSS, IXs]
		init_ss[init_ss>a] = nan
	else:
		init_ss = array([z]*nSS)

	#generate random probability vectors [nSS Trials x nTimesteps SSD -> TB]
	sstrials =  rs((nSS, Ts))
	#simulate all ss signal paths
	DVs = init_ss[:, None] + cumsum(where(sstrials<Ps, dx, -dx), axis=1)

	return DVg, DVs


def run_reactive(a, tr, v, ssv, z, ssd, pGo=.5, ntrials=2000, tb=0.650, tau=.0005, si=.01):

	"""
	DVg is instantiated for all trials. DVs contains traces for a subset of
	those trials in which a SS occurs (proportional to pGo provided in theta).
	"""

	nSS=int(ntrials*(1-pGo))


	dx=sqrt(si*tau)
	#Ps = 0.5*(1 + ssv*dx/si)
	#Pg = 0.5*(1 + mu*dx/si)
	#Tg = ceil((tb-tr)/tau)
	#Ts = ceil((tb-ssd)/tau)

	#generate random probability vectors [nGo Trials x nTimesteps TR -> TB]
	#trials= random.random_sample((ntrials,  ceil((tb-ssd)/tau)))
	#simulate all go signal paths
	DVg = z + cumsum(where(rs((ntrials, ceil((tb-ssd)/tau)))<(0.5*(1 + mu*dx/si)), dx, -dx), axis=1)
	if tr<ssd: init_ss = DVg[:int(ntrials*(1-pGo)), ceil((tb-tr)/tau) - ceil((tb-ssd)/tau)]
	else: init_ss = array([z]*int(ntrials*(1-pGo)))
	DVs = init_ss[:, None] + cumsum(where(rs((nSS, ceil((tb-ssd)/tau)))<(0.5*(1 + ssv*dx/si)), dx, -dx), axis=1)
	#IXs = Tg - Ts
	#init_ss[init_ss>a] =  nan


	#generate random probability vectors [nSS Trials x nTimesteps SSD -> TB]
	#sstrials = rs((nSS, Ts))
	#simulate all ss signal paths

	return DVg, DVs
