#!usr/bin/env python
from __future__ import division
import os, sys, time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles


def update_params(theta):

	if 't_hi' in theta.keys():
		theta['tt'] = theta['t_lo'] + np.random.uniform() * (theta['t_hi'] - theta['t_lo'])
	else:
		theta['tt']=theta['t']

	if 'z_hi' in theta.keys():
		theta['zz'] = theta['z_lo'] + np.random.uniform() * (theta['z_hi'] - theta['z_lo'])
	else:
		theta['zz']=theta['z']

	if 'sv' in theta.keys():
		theta['vv'] = theta['sv'] * np.random.randn() + theta['v']
	else:
		theta['vv']=theta['v']

	return theta


def get_intervar_ranges(theta):
	"""
	:args:
		parameters (dict):	dictionary of theta (Go/NoGo Signal Parameters)
					and sp (Stop Signal Parameters)
	"""
	if 'st' in theta.keys():
		theta['t_lo'] = theta['t'] - theta['st']/2
		theta['t_hi'] = theta['t'] + theta['st']/2
	if 'sz' in theta.keys():
		theta['z_lo'] = theta['z'] - theta['sz']/2
		theta['z_hi'] = theta['z'] + theta['sz']/2
	return theta


def pstop_mquant(df, filt_rts=True, quantp=[.1,.3,.5,.7,.9]):

        pstop=1-df.groupby('pGo').mean()['response'].values
        if filt_rts:
                godf=df.query('choice=="go" and rt<=.5451')
        else:
                godf=df.query('choice=="go"')
        mquant=[np.mean(mquantiles(pgdf.rt.values, prob=quantp))*10 for pg, pgdf in godf.groupby('pGo')]
        return pstop, mquant


def pstop_quants(df, filt_rts=True, quantp=[.1,.3,.5,.7,.9]):

        pstop=1-df.groupby('pGo').mean()['response'].values
        if filt_rts:
                godf=df.query('choice=="go" and rt<=.5451')
        else:
                godf=df.query('choice=="go"')
        rtquants=[mquantiles(pgdf.rt.values, prob=quantp)*10 for pg, pgdf in godf.groupby('pGo')]
        return pstop, rtquants


def pstop_meanrt(df, filt_rts=True):

        pstop=1-df.groupby('pGo').mean()['response'].values
        if filt_rts:
        	godf=df.query("response==1 and rt<=.5451")
        else:
                godf=df.query('response==1')
        go_rt=godf.groupby('pGo').mean()['rt'].values

        return pstop, go_rt*10


def get_params_from_flatp(theta, dep='v', pgo=np.arange(0,120,20)):

	if not type(theta)==dict:
		theta=theta.to_dict()['mean']

	keep=['a', 'z', 'v', 't', 'ssv', 'ssd', 'pGo']
	keep.pop(keep.index(dep))

	plist=[theta[dep+str(pg)] for pg in pgo]

	for k in theta.keys():
		if k not in keep:
			theta.pop(k)

	return theta, plist


def rangl_data(data, cutoff=.650):

	gocor=data[(data['trial_type']=='go')&(data['acc']==1)&(data['rt']<cutoff)]
	rts=gocor.rt.mean()*10
	ydata=data[data['trial_type']=='stop'].groupby('ssd').mean()['acc'].values

	return np.append(ydata, rts)


def rwr(X, get_index=False, n=None):
	"""
	Modified from http://nbviewer.ipython.org/gist/aflaxman/6871948
	"""

	if isinstance(X, pd.Series):
		X = X.copy()
		X.index = range(len(X.index))
	if n == None:
		n = len(X)

	resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
	X_resample = np.array(X[resample_i])

	if get_index:
		return resample_i
	else:
		return X_resample
