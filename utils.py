#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from scipy.stats.mstats import mquantiles as mq
from scipy import optimize
from scipy.io import loadmat
import os, re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors.kde import KernelDensity


def rangl_re(data, cutoff=.650, prob=np.array([.1, .3, .5, .7, .9])):

	gotrials = data.query('response==1 & acc==1')
	sigresp = data.query('response==1 & acc==0')

	pg_cor, pg_err = data.groupby('trial_type').mean()['response'].values
	wcor = prob*pg_cor
	werr = prob*pg_err

	gq = mq(gotrials.rt.values, prob=prob)#wcor)
	eq = mq(sigresp.rt.values, prob=prob)#werr)

	return np.hstack([gq, pg_cor, eq, pg_err, pstop])



def rangl_pro(data, tb=.560, rt_cutoff=.54502, prob=np.array([1, 3, 5, 7, 9])):

	godf = data.query('response==1')
	gotrials=godf[godf.rt<=rt_cutoff]
	pgo = data.response.mean()
	gp = pgo*prob
	gq = mq(gotrials.rt, prob=gp)
	gmu = gotrials.rt.mean()
	return np.hstack([gq*10, gp, gmu, pgo])


def kde_fit_quantiles(rtquants, nsamples=1000):
	"""
	takes quantile estimates and fits cumulative density function
	returns samples to pass to sns.kdeplot()
	"""

	kdefit = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(rtquants)

	samples = kdefit.sample(n_samples=nsamples).flatten()

	return samples



def aic(model):
	k = len(model.get_stochasticts())
	logp = sum([x.logp for x in model.get_observeds()['node']])
	return 2 * k - 2 * logp


def bic(model):
	k = len(model.get_stochastics())
	n = len(model.data)
	logp = sum([x.logp for x in model.get_observeds()['node']])
	return -2 * logp + k * np.log(n)


def resample_reactive(data, n=120):

	df=data.copy(); bootlist=list()
	if n==None: n=len(df)

	for ssd, ssdf in df.groupby('ssd'):
		boots = ssdf.reset_index(drop=True)
		orig_ix = np.asarray(boots.index[:])
		resampled_ix = rwr(orig_ix, get_index=True, n=n)
		bootdf = ssdf.irow(resampled_ix)
		bootlist.append(bootdf)

	#concatenate and return all resampled conditions
	return rangl_re(pd.concat(bootlist))

def resample_proactive(data, n=120, rt_cutoff=.54502):

	df=data.copy(); bootdf_list=list()
	if n==None: nlist=len(df)

	boots = df.reset_index(drop=True)
	orig_ix = np.asarray(boots.index[:])
	resampled_ix = rwr(orig_ix, get_index=True, n=n)
	bootdf = df.irow(resampled_ix)
	bootdf_list.append(bootdf)

	return rangl_pro(pd.concat(bootdf_list), rt_cutoff=rt_cutoff)

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

def get_proactive_params(theta, dep='v', pgo=np.arange(0,120,20)):

	if not type(theta)==dict:
		theta=theta.to_dict()['mean']

	keep=['a', 'z', 'v', 'tr', 'ssv', 'ssd']
	keep.pop(keep.index(dep))

	pdict={pg:theta[dep+str(pg)] for pg in pgo}

	for k in theta.keys():
		if k not in keep:
			theta.pop(k)

	return theta, pdict


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

def update_params(theta):

	if 't_hi' in theta.keys():
		theta['tr'] = theta['t_lo'] + np.random.uniform() * (theta['t_hi'] - theta['t_lo'])
	else:
		theta['tr']=theta['tr']

	if 'z_hi' in theta.keys():
		theta['z'] = theta['z_lo'] + np.random.uniform() * (theta['z_hi'] - theta['z_lo'])
	else:
		theta['z']=theta['z']

	if 'sv' in theta.keys():
		theta['v'] = theta['sv'] * np.random.randn() + theta['v']
	else:
		theta['v']=theta['v']

	return theta


def get_intervar_ranges(theta):
	"""
	:args:
		parameters (dict):	dictionary of theta (Go/NoGo Signal Parameters)
					and sp (Stop Signal Parameters)
	"""
	if 'st' in theta.keys():
		theta['t_lo'] = theta['tr'] - theta['st']/2
		theta['t_hi'] = theta['tr'] + theta['st']/2
	if 'sz' in theta.keys():
		theta['z_lo'] = theta['z'] - theta['sz']/2
		theta['z_hi'] = theta['z'] + theta['sz']/2
	return theta


def sigmoid(p,x):
        x0,y0,c,k=p
        y = c / (1 + np.exp(k*(x-x0))) + y0
        return y

def residuals(p,x,y):
        return y - sigmoid(p,x)

def res(arr,lower=0.0,upper=1.0):
        arr=arr.copy()
        if lower>upper: lower,upper=upper,lower
        arr -= arr.min()
        arr *= (upper-lower)/arr.max()
        arr += lower
        return arr

def get_intersection(iter1, iter2):

	intersect_set = set(iter1).intersection(set(iter2))

	return np.array([i for i in intersect_set])

def ssrt_calc(df, avgrt=.3):

	dfstp = df.query('trial_type=="stop"')
	dfgo = df.query('choice=="go"')

	pGoErr = np.array([idf.response.mean() for ix, idf in dfstp.groupby('idx')])
	nlist = [int(pGoErr[i]*len(idf)) for i, (ix, idf) in enumerate(df.groupby('idx'))]

	GoRTs = np.array([idf.rt.sort(inplace=False).values for ix, idf in dfgo.groupby('idx')])
	ssrt_list = np.array([GoRTs[i][nlist[i]] for i in np.arange(len(nlist))]) - avgrt

	return ssrt_list


def scurves(lines=[], task='ssRe', pstop=.5, ax=None, linestyles=None, colors=None, labels=None):

        if len(lines[0])==6:
                task='pro'
	if ax is None:
		f, ax = plt.subplots(1)
	if colors is None:
		colors=sns.color_palette('husl', n_colors=len(lines))
		labels=['']*len(lines)
		linestyles = ['-']*len(lines)

        lines=[np.array(line) if type(line)==list else line for line in lines]
	pse=[];

	if 'Re' in task:
		x=np.array([400, 350, 300, 250, 200], dtype='float')
		xtls=x.copy()[::-1]; xsim=np.linspace(15, 50, 10000);
		yylabel='P(Stop)'; scale_factor=100; xxlabel='SSD'; xxlim=(18,42)
	else:
		x=np.array([100, 80, 60, 40, 20, 0], dtype='float')
		xsim=np.linspace(-5, 11, 10000)
		scale_factor=10

	x=res(-x,lower=x[-1]/10, upper=x[0]/10)
	for i, yi in enumerate(lines):

		y=res(yi, lower=yi[-1], upper=yi[0])
		p_guess=(np.mean(x),np.mean(y),.5,.5)
		p, cov, infodict, mesg, ier = optimize.leastsq(residuals, p_guess, args=(x,y), full_output=1, maxfev=5000, ftol=1.e-20)
		x0,y0,c,k=p
		xp = xsim
		pxp=sigmoid(p,xp)
		idx = (np.abs(pxp - pstop)).argmin()

		pse.append(xp[idx]/scale_factor)

		# Plot the results
		ax.plot(xp, pxp, linestyle=linestyles[i], lw=3.5, color=colors[i], label=labels[i])
		pse.append(xp[idx]/scale_factor)

	plt.setp(ax, xlim=xxlim, xticks=x, ylim=(-.05, 1.05), yticks=[0, 1])
	ax.set_xticklabels([int(xt) for xt in xtls], fontsize=18); ax.set_yticklabels([0.0, 1.0], fontsize=18)
	ax.set_xlabel(xxlabel, fontsize=18); ax.set_ylabel(yylabel, fontsize=18)
	ax.legend(loc=0)
	return np.array(pse)



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


def mat2py(indir, outdir=None, droplist=None):

	if droplist is None:
		droplist = ['dt_vec', 'Speed', 'state', 'time', 'probe_trial', 'ypos', 'fill_pos', 't_vec', 'Y', 'pos']

	flist = filter(lambda x: 'SS' in x and '_fMRI_Proactive' in x and 'run' in x, os.listdir(indir))
	dflist = []
	noresp_group = []
	os.chdir(indir)
	for name in flist:

		idx, run = re.split('_|-', name)[:2]
	        date = '-'.join(re.split('_|-', name)[2:5])

		mat = loadmat(name)  # load mat-file
		mdata = mat['Data']  # variable in mat file
	        mdtype = mdata.dtype  # dtypes of structures are "unsized objects"


		columns = [ n for n in mdtype.names]
		columns.insert(0, 'idx')
		columns.insert(1, 'run')
	        columns.insert(2, 'date')

		data = [[vals[0][0] for vals in trial] for trial in mat['Data'][0]]
		for trial in data:
			trial.insert(0,int(idx[2:]))
			trial.insert(1, int(run[-1]))
			trial.insert(2, date)
		df = pd.DataFrame(data, columns=columns)
	        df.rename(columns={'Keypress': 'go', 'Hit':'hit', 'Stop':'nogo', 'DO_STOP':'sstrial', 'GoPoint':'pGo', 'Bonus':'bonus'}, inplace=True)

		df['gotrial']=abs(1-df.sstrial)
		df['ttype']=np.where(df['gotrial']==1, 'go', 'stop')
		df['response']=df['go'].copy()
		df['rt']=df['pos']*df['Speed']
	        df.drop(droplist, axis=1, inplace=True)

		if 'trial_start_time' in columns:
			df.drop('trial_start_time', axis=1, inplace=True)

		if df.response.mean()<.2:
			noresp_group.append(df)
		else:
			df['run'] = df['run'].astype(int)
			df['idx'] = df['idx'].astype(int)
			df['response'] = df['response'].astype(int)
			df['hit'] = df['hit'].astype(int)
			df['nogo'] = df['nogo'].astype(int)
			df['sstrial'] = df['sstrial'].astype(int)
			df['gotrial'] = df['gotrial'].astype(int)
			df['bonus'] = df['bonus'].astype(int)
			df['pGo'] = df['pGo']*100
			df['pGo'] = df['pGo'].astype(int)
			df['rt'] = df['rt'].astype(float)

			if outdir:
				df.to_csv(outdir+'sx%s_proimg_data.csv' % idx, index=False)
			dflist.append(df)

	master=pd.concat(dflist)
	if outdir:
		master.to_csv(outdir+"ProImg_All.csv", index=False)

	return master
