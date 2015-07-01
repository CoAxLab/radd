#!/usr/local/bin/env python
from __future__ import division
import os
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from scipy import optimize
from scipy.io import loadmat
from radd import fitre, fitpro, utils

class Model(object):

      from radd import fitre, fitpro, utils
      from scipy.stats.mstats import mquantiles

      def __init__(self, kind='reactive', model='radd', inits={}, data=pd.DataFrame, depends_on={'xx':'XX'}, fit='bootstrap', niter=50, *args, **kwargs):

            self.model = model
            self.inits = inits
            self.kind = kind
            self.data = data
            self.niter = niter
            self.depends_on = depends_on
            self.depends = self.depends_on.keys()
            self.cond = self.depends_on.values()[0]
            self.fitparams = None
            self.live_update = True
            self.i = 0
            self.fit = fit
            if self.fit=='bootstrap':
                  self.indx=range(niter)
            elif self.fit=='subjects':
                  self.indx=list(self.data.idx.unique())




      def set_fitparams(self, xtol=1.e-3, ftol=1.e-3, maxfun=5000, ntrials=2000, niter=500):

            self.fitparams = {'ntrials':ntrials, 'maxfun':maxfun, 'ftol':ftol, 'xtol':xtol, 'niter':niter}

            if self.fit=='bootstrap':
                  self.indx=range(self.fitparams['niter'])



      def get_fitparams(self):

            if self.fitparams==None:
                  self.set_fitparams()
            fitp = self.fitparams

            return fitp['ntrials'], fitp['ftol'], fitp['xtol'], fitp['maxfun'], fitp['niter']



      def set_rt_cutoff(self, rt_cutoff=None):

            if rt_cutoff==None:
                  if self.kind=='reactive':
                        self.rt_cutoff = .650
                  elif self.kind=='proactive':
                        self.rt_cutoff = .54502
                  else:
                        self.rt_cutoff=self.data.query('response==1').rt.max()
            else:
                  self.rt_cutoff=rt_cutoff



      def global_opt(self):

            inits = self.inits
            ntrials, ftol, xtol, maxfun, niter = self.get_fitparams()

            self.inits, yhat = run_reactive_model(y, inits=inits, ntrials=ntrials, model=model, depends=['xx'], maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=1)



      def prepare_fit(self):

            print "preparing %s model using %s method" % (self.kind, self.fit)

            inits=self.inits; data=self.data; cond=self.cond; indx = self.indx
            labels = data[cond].unique(); nlabels = len(labels);
            index = range(nlabels*len(indx))

            delays = list(data.query('trial_type=="stop"').ssd.unique())
            self.build_stores(data, cond, inits, index, delays)

            if self.fit=='bootstrap':
                  if self.kind=='reactive':
                        ifx = utils.resample_reactive
                  elif self.kind=='proactive':
                        ifx = utils.resample_proactive

                  #CREATE ITERABLE OBJECT CONTAINING NITER of RESAMPLED DATA FOR FITTING
                  self.dat = np.vstack([[ifx(cdf) for i in indx] for c, cdf in self.data.groupby(self.cond)])

            elif self.fit=='subjects':
                  if self.kind=='reactive':
                        ifx = utils.rangl_re
                  elif self.kind=='proactive':
                        ifx = utils.rangl_pro

                  #CREATE ITERABLE OBJECT CONTAINING ALL INDIVIDUAL IDX DATA FOR FITTING
                  self.dat = np.vstack([[ifx(idxdf) for idx, idxdf in cdf.groupby('idx')] for c, cdf in     self.data.groupby(self.cond)])

            # STORE THE DATA OBJECTS IN PANDAS PANEL/DF FOR CONVENIENT FUTURE REFERENCE
            lalist=sum([[lbl]*len(indx) for lbl in labels], [])
            #self.qp.gqp_obs.loc[:, '5q':] = dat[:, :6]
            self.qp.gqp_obs.loc[:,:] = np.hstack([np.array([lalist, indx*nlabels]).T,  self.dat[:, :6]])
            self.qp.eqp_obs.loc[:, :] = np.hstack([np.array([lalist, indx*nlabels]).T,  self.dat[:, 6:12]])
            self.pstop.pstop_obs.loc[:, :] = np.hstack([np.array([lalist, indx*nlabels]).T, self.dat[:, 12:]])
            self.popt.iloc[:, :2] =  np.vstack([lalist, indx*nlabels]).T



      def build_stores(self, data, cond, inits, index, delays):

            popt_cols = np.hstack(['indx', cond, 'chi', inits.keys()])
            qp_cols = ['indx', cond, '5q', '25q', '50q', '75q', '95q', 'presp']
            pstop_cols = np.hstack(['indx', cond, delays])

            qp_df = pd.DataFrame(columns=qp_cols, index=index)
            pstop_df = pd.DataFrame(columns=pstop_cols, index=index)

            if self.kind=='reactive':
                  qpitems = ['gqp_obs', 'eqp_obs', 'gqp_fit', 'eqp_fit']
                  psitems = ['pstop_obs', 'pstop_fit']
                  self.qp = pd.Panel.from_dict({item: qp_df.copy() for item in qpitems}, orient='items')
                  self.pstop = pd.Panel.from_dict({item: pstop_df.copy() for item in psitems}, orient='items')

            elif self.kind=='proactive':
                  qpitems = ['gqp_obs', 'gqp_fit']
                  self.qp = pd.Panel.from_dict({item: qp_df.copy() for item in qpitems}, orient='items')

            self.popt = pd.DataFrame(columns=popt_cols, index=index)
            self.isprepared = True



      def run_model(self, save=False, savepth='./', live_update=True, disp=False, prepare=False, **kwargs):

            if "depends_on" in kwargs.keys():
                  self.depends_on = kwargs['depends_on']
                  self.depends = self.depends_on.keys()
                  self.cond = self.depends_on.values()[0]

            if not self.isprepared:
                  # initialize data storage objects
                  self.prepare_fit()

            for label, cdf in self.data.groupby(self.cond):
                  self.fit_indx(cdata=cdf, indx=indx, label=label, savepth=savepth, live_update=live_update, disp=disp)

            if save:
                  self.qp.gqp_fits.to_csv(savepth+model+"_qpfits.csv", index=False)
                  self.pstop.pstop_fits.to_csv(savepth+self.model+"_scurve.csv", index=False)
                  self.popt.to_csv(savepth+model+"_popt.csv")



      def fit_indx(self, cdata, indx, label, savepth='./', disp=False, live_update=True):

            cdf = cdata.copy(); inits = self.inits; model=self.model; depends=self.depends;
            ntrials, ftol, xtol, maxfun, niter = self.get_fitparams()

            for i, y in enumerate(self.dat):

                  if self.kind=='reactive':
                        params, yhat = fitre.fit_reactive_model(y, inits=inits, ntrials=ntrials, model=model, depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=0, disp=disp)
                        self.store_recost(indx[i], label, params, yhat)

                  elif self.kind=='proactive':
                        inits['pGo']=cdf.pGo.mean()
                        params, yhat = fitpro.fit_proactive_model(y, inits=inits, ntrials=ntrials, model=model, depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=0, disp=disp)
                        self.store_procost(indx[i], label, params, yhat)

                  if live_update:
                        self.qp.gqp_fits.to_csv(savepth+model+"_gqp.csv", index=False)
                        self.popt.to_csv(savepth+model+"_popt.csv", index=False)

                        if self.kind=='reactive':
                              self.qp.eqp_fits.to_csv(savepth+model+"_eqp.csv", index=False)
                              self.pstop.pstop_fits.to_csv(savepth+model+"_pstop.csv", index=False)




      def store_recost(self, indxi, label, params, yhat):
            # get predictions and store optimized parameter set
            popti = pd.Series({k:params[k] for k in self.popt.columns})
            self.popt.iloc[self.i, 2:] = popti

            # fill df: [cond] [go/ssgo], [cor/err rt quantiles], [prob corr/err response]
            self.qp.gqp_fits.iloc[self.i, 2:] = yhat[:6]
            self.qp.eqp_fits.iloc[self.i, 2:] = yhat[6:12]
            self.pstop.pstop_fits.iloc[self.i, 2:] = yhat[12:]
            self.i+=1




      def store_procost(self, indxi, label, params, yhat):
            # get predictions and store optimized parameter set

            popti = pd.Series({k:params[k] for k in self.popt.columns})

            # fill df: [cond] [go/ssgo], [cor/err rt quantiles], [prob corr/err response]
            self.qp.gqp_fits.loc[self.i,:] = yhat[:6]
            self.popt.loc[self.i,:] = popti
            self.i+=1
