#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os, re



def remove_outliers(df, sd=1.5, verbose=False):

      ssdf=df[df.response==0]
      godf = df[df.response==1]
      bound = godf.rt.std()*sd
      rmslow=godf[godf['rt']<(godf.rt.mean()+bound)]
      clean_go=rmslow[rmslow['rt']>(godf.rt.mean()-bound)]

      clean=pd.concat([clean_go, ssdf])
      if verbose:
            pct_removed = len(clean)*1./len(df)
            print "len(df): %i\nbound: %s \nlen(cleaned): %i\npercent removed: %.5f" % (len(df), str(bound), len(clean), pct_removed)
            
      return clean



def get_default_inits(kind='radd', dynamic='hyp', depends_on={}, include_ss=False, fit_noise=False):

	if 'radd' in kind:
            if len(depends_on.keys())>1:
		      inits = {'a':.45, 'ssv':-0.9473, 'tr': 0.2939, 'v': 1.0919}
            if 'v' in depends_on.keys():
                  inits = {'a':0.4441, 'ssv':-0.9473, 'tr':0.3049, 'v':1.0919, 'z':0.1542}
            elif 'tr' in depends_on.keys():
                  inits = {'a': 0.4550, 'ssv': -0.9788, 'tr':.3392, 'v': 1.1392, 'z': 0.1545}

	elif 'pro' in kind:
		if len(depends_on.keys())>1:
		      inits = {'a':.39, 'tr': 0.2939, 'v': 1.0919}
		if 'tr' in depends_on.keys():
			inits = {'a':0.3267, 'tr':0.3192, 'v': 1.3813}
		elif 'v' in depends_on.keys():
			inits = {'a':0.4748, 'tr':0.2725,'v':1.6961}
		if 'x' in kind and dynamic == 'exp':
			vopt = np.array([1.2628, 1.4304, 1.5705, 1.701, 1.8682, 1.9973])
			inits = {'a':0.4836, 'xb': 1.4604 , 'tr': 0.3375}
			inits['v'] = vopt.mean()
		elif 'xb' in kind and dynamic == 'hyp':
			inits = {'xb': .01, 'a': 0.473022,"tr":0.330223, "v":1.24306}

	elif 'race' in kind:
		inits = {'a':0.3926740, 'ssv':1.1244, 'tr':0.33502, 'v':1.0379,  'z':0.1501}

	return inits

def ensure_numerical_wts(wts, fwts):

	# test inf
	wts[np.isinf(wts)] = np.median(wts[~np.isinf(wts)])
	fwts[np.isinf(fwts)] = np.median(fwts[~np.isinf(fwts)])

	# test nan
	wts[np.isnan(wts)] = np.median(wts[~np.isnan(wts)])
	fwts[np.isnan(fwts)] = np.median(fwts[~np.isnan(fwts)])

	return wts, fwts

def get_header(params=None, data_style='re', labels=[], delays=[], prob=np.array([.1, .3, .5, .7, .9])):

	info = ['nfev','chi','rchi','AIC','BIC','CNVRG']
	if data_style=='re':
		cq = ['c'+str(int(n*100)) for n in prob]
		eq = ['e'+str(int(n*100)) for n in prob]
		qp_cols = ['Go'] + delays + cq + eq
	else:
		hi = ['hi'+str(int(n*100)) for n in prob]
		lo = ['lo'+str(int(n*100)) for n in prob]
		qp_cols = labels + hi + lo

	if params is not None:
		infolabels = params + info
		return [qp_cols, infolabels]
	else:
		return [qp_cols]

def check_inits(inits={}, kind='radd', dynamic='hyp', pro_ss=False, fit_noise=False):

	single_bound_models = ['xirace', 'irace', 'xpro', 'pro']

	for k, val in inits.items():
		if isinstance(val, np.ndarray):
			inits[k]=val[0]
	if 'ssd' in inits.keys():
		del inits['ssd']
	if 'pGo' in inits.keys():
		del inits['pGo']

	if pro_ss and 'ssv' not in inits.keys():
		inits['ssv'] = -0.9976

	if kind in single_bound_models and 'z' in inits.keys():
		z=inits.pop('z')
		inits['a']=inits['a']-z

	if 'race' in kind:
		inits['ssv']=abs(inits['ssv'])
	elif 'radd' in kind:
		inits['ssv']=-abs(inits['ssv'])

	if 'pro' in kind:
		if pro_ss and 'ssv' not in inits.keys():
			inits['ssv'] = -0.9976
		elif not pro_ss and 'ssv' in inits.keys():
			ssv=inits.pop('ssv')

	if 'x' in kind and 'xb' not in inits.keys():
		if dynamic == 'exp':
			inits['xb'] = 2
		elif dynamic == 'hyp':
			inits['xb'] = .02

	if fit_noise and 'si' not in inits.keys():
		inits['si'] = .01

	return inits


def make_proRT_conds(data, split):

	if np.any(data['pGo'].values > 1):
		data['pGo']=data['pGo']*.01
	if np.any(data['rt'].values > 5):
		data['rt']=data['rt']*.001

	if split=='HL':
		data['HL']='x'
		data.ix[data.pGo>.5, 'HL']=1
		data.ix[data.pGo<=.5, 'HL']=2
	return data


def rename_bad_cols(data):

	if 'trial_type' in data.columns:
		data.rename(columns={'trial_type':'ttype'}, inplace=True)

	return data



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
