#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
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


def remove_outliers(df, sd=1.95):

	print "len(df) = %s \n\n" % (str(len(df)))

	df_ss=df[df['choice']=='stop']
	df_go=df[df['choice']=='go']
	cutoff_go=df_go['rt'].std()*sd + (df_go['rt'].mean())
	df_go_new=df_go[df_go['rt']<cutoff_go]

	df_trimmed=pd.concat([df_go_new, df_ss])
	df_trimmed.sort('trial', inplace=True)

	print "cutoff_go = %s \nlen(df_go) = %i\n len(df_go_new) = %i\n" % (str(cutoff_go), len(df_go), len(df_go))

	return df_trimmed

def flatui():

        return { "t1":"#1abc9c","t2":"#16a085","g1":"#2ecc71","g2":"#27ae60",
                "b1":"#2980b9","b2":"#4168B7","p1":"#9B59B6","p2":"#674172",
                "m1":"#34495e","m2":"#2c3e50","y1":"#f1c40f","y2":"#f39c12",
                "o1":"#e67e22","o2":"#d35400","r1":"#e74c3c","r2":"#c0392b",
                "gr1":"#ecf0f1", "gr2":"#bdc3c7","a1":"#95a5a6","a2":"#7f8c8d" }

def get_hsv(hexrgb):
    hexrgb = hexrgb.lstrip("#")   # in case you have Web color specs
    r, g, b = (int(hexrgb[i:i+2], 16) / 255.0 for i in xrange(0,5,2))

    return colorsys.rgb_to_hsv(r, g, b)

def style_params(style='ticks', context='paper'):

	if style=='ticks':
		rcdict={'axes.axisbelow': True,'axes.edgecolor': '.15','axes.facecolor': 'white','axes.grid': False,'axes.labelcolor': '.15',
		'axes.linewidth': 1.2,'font.family': 'Helvetica','grid.color': '.8','grid.linestyle': '-','image.cmap': 'Greys',
		'legend.frameon': False,'legend.numpoints': 1,'legend.scatterpoints': 1,'lines.solid_capstyle': 'round','pdf.fonttype': 42,
		'text.color': '.15','xtick.color': '.15','xtick.direction': 'out','xtick.major.size': 6,'xtick.minor.size': 3,'ytick.color': '.15',
		'ytick.direction': 'out','ytick.major.size': 6,'ytick.minor.size': 3}
	if context=='paper':
		cdict={'axes.labelsize': 16,'axes.titlesize': 17.28,'figure.figsize': np.array([ 5,  5]), 'grid.linewidth': 0.8,
		'legend.fontsize': 14.,'lines.linewidth': 3.0,'lines.markeredgewidth': 0.0, 'lines.markersize': 6.,'patch.linewidth': 0.24,
		'xtick.labelsize': 14.,'xtick.major.pad': 5.6, 'xtick.major.width': 0.8,'xtick.minor.width': 0.4,'ytick.labelsize': 14.,
		'ytick.major.pad': 5.6,'ytick.major.width': 0.8,'ytick.minor.width': 0.4}

	colors=['ghostwhite', '#95A5A6', '#6C7A89',
	'#3498db', '#4168B7', '#5C97BF', '#34495e', '#3A539B', '#4B77BE',
	(0.21568627450980393, 0.47058823529411764, 0.7490196078431373),
 	(0.23137254901960785, 0.3568627450980392, 0.5725490196078431),
	'#2ecc71', '#009B76', '#00B16A',"mediumseagreen", '#16A085', '#019875',
	(0.5098039215686274, 0.37254901960784315, 0.5294117647058824),
	'#674172', '#9B59B6', '#8E44AD', '#BF55EC', '#663399', '#9A12B3',
	(0.996078431372549, 0.7019607843137254, 0.03137254901960784),
	'#F27935','#E26A6A', '#F62459',
	(0.8509803921568627, 0.32941176470588235, 0.30196078431372547),
	'#D91E18', '#F64747', '#e74c3c','#CF000F']

	return {'style':rcdict, 'context':cdict, 'colors':colors, 'reds':colors[-9:],
		'purples':colors[-16:-9], 'greens':colors[-22:-16], 'grays':colors[:3],'blues':colors[3:11]}
