#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os, re




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



def get_intersection(iter1, iter2):

      intersect_set = set(iter1).intersection(set(iter2))

      return ([i for i in intersect_set])


def old_proactive_weighting_code():

	pass
	#upper = self.data[self.data.isin([.6,.8,1.0])].response.mean()
	#lower = self.data[self.data.pGo.isin([.2,.4,.6])].response.mean()
	#qvar = self.observed.std().iloc[6:].values
	#hi = qvar[:5]; lo = qvar[5:]
	#qwts = np.hstack([upper*(hi[2]/hi), lower*(lo[2]/lo)])

	#pvar = self.data.groupby('pGo').std().response.values
	#psub1 = np.median(pvar[:-1])/pvar[:-1]
	#pwts = np.append(psub1, psub1.max())
	#pwts = np.array([1.5,1,1,1,1,1.5])
	#self.wts = np.hstack([pwts, qwts])
	#qvar = self.observed.std().iloc[6:].values.reshape(nrtc, len(prob))
	#qr = np.median(qvar)/qvar
	#qwts = np.append(upper*qr[:5], lower*qr[5:])
	#wt_hi = upper*sq_ratio[0, :]
	#wt_lo = lower*sq_ratio[1, :]
	#self.wts = np.hstack([pwts, wt_hi, wt_lo])

	#qwts = np.hstack([upper*(hi[2]/hi), lower*(lo[2]/lo)])
	#pwts = np.array([1.5,  1,  1,  1,  2, 2])
	#self.wts = np.hstack([pwts, qwts])

	
