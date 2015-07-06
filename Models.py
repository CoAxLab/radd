#!/usr/local/bin/env python
from __future__ import division
import os
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from scipy import optimize
from scipy.io import loadmat
from radd import fitre, fitpro, utils
from numba.decorators import jit


class Model(object):

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

            delays = sorted(data.query('trial_type=="stop"').ssd.unique().astype(np.int))
            #self.build_stores(data, cond, inits, index, delays)

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
                  self.dat = np.vstack([[ifx(idxdf) for idx, idxdf in cdf.groupby('idx')] for c, cdf in self.data.groupby(self.cond)]).astype(np.float32)

            popt_cols = np.hstack(['chi', inits.keys()])
            qp_cols = ['GoAcc']+delays +['c5','c25','c50','c75','c95'] + ['e5','e25','e50','e75','e95']
            ixdf = pd.concat([pd.DataFrame({'indx': indx, 'cond':c}, columns=['indx', 'cond']) for c in labels]).reset_index(drop=True)

            self.observed = pd.concat([ixdf, pd.DataFrame(self.dat, columns=qp_cols)], axis=1)
            self.fit = pd.concat([ixdf, pd.DataFrame(np.zeros_like(self.dat), columns=qp_cols)], axis=1)
            self.popt = pd.concat([ixdf, pd.DataFrame(columns=popt_cols, index=ixdf.index)], axis=1)
            #self.fit = pd.DataFrame(columns=qp_cols, index=index)
            #pstop_df = pd.DataFrame(columns=pstop_cols, index=index)

            #qpitems = ['obs', 'fit']
            # STORE THE DATA OBJECTS IN PANDAS PANEL/DF FOR CONVENIENT FUTURE REFERENCE
            #lalist=sum([[lbl]*len(indx) for lbl in labels], [])
            #self.qp.gqp_obs.loc[:,:] = np.hstack([np.array([lalist, indx*nlabels]).T, self.dat[:, :6]])
            #self.qp.eqp_obs.loc[:, :] = np.hstack([np.array([lalist, indx*nlabels]).T, self.dat[:, 6:12]])
            #self.pstop.pstop_obs.loc[:, :] = np.hstack([np.array([lalist, indx*nlabels]).T, self.dat[:, 12:]])
            #self.popt.iloc[:, :2] =  np.vstack([lalist, indx*nlabels]).T
            #self.isprepared = True

            # STORE THE DATA OBJECTS IN PANDAS PANEL/DF FOR CONVENIENT FUTURE REFERENCE
            #lalist=sum([[lbl]*len(indx) for lbl in labels], [])
            #tmp = pd.DataFrame(np.vstack([indx*nlabels, lalist]).T)
            #self.observed.iloc[:,2:] = self.dat
            #self.observed = pd.DataFrame(self.dat, columns=qp_cols, index=index)
            #self.popt = pd.DataFrame(columns=popt_cols, index=index)
            #self.popt.iloc[:, :2] =  np.vstack([lalist, indx*nlabels]).T
            self.isprepared = True

      def build_stores(self, data, cond, inits, index, delays):

            #popt_cols = np.hstack(['indx', cond, 'chi', inits.keys()])
            #qp_cols = ['indx', cond, '5q', '25q', '50q', '75q', '95q', 'presp']
            #pstop_cols = np.hstack(['indx', cond, delays])
            #qp_cols = ['indx', 'cond', 'GoAcc'] + delays + ['c5', 'c25', 'c50', 'c75', 'c95'] + ['e5', 'e25', 'e50', 'e75', 'e95']
            #qp_df = pd.DataFrame(columns=qp_cols, index=index)
            #pstop_df = pd.DataFrame(columns=pstop_cols, index=index)

            #qpitems = ['obs', 'fit']
            #psitems = ['pstop_obs', 'pstop_fit']
            #self.qp = pd.Panel.from_dict({item: qp_df.copy() for item in qpitems}, orient='items')

            #if self.kind=='reactive':
                  #qpitems = ['gqp_obs', 'eqp_obs', 'gqp_fit', 'eqp_fit']
                  #qpitems = ['qp_obs', 'qp_fit']
                  #psitems = ['pstop_obs', 'pstop_fit']
                  #self.qp = pd.Panel.from_dict({item: qp_df.copy() for item in qpitems}, orient='items')
                  #self.pstop = pd.Panel.from_dict({item: pstop_df.copy() for item in psitems}, orient='items')

            #elif self.kind=='proactive':
                  #qpitems = ['gqp_obs', 'gqp_fit']
                  #self.qp = pd.Panel.from_dict({item: qp_df.copy() for item in qpitems}, orient='items')

            #self.popt = pd.DataFrame(columns=popt_cols, index=index)
            self.isprepared = True



      def run_model(self, save=False, savepth='./', live_update=True, disp=False, prepare=False, **kwargs):

            if "depends_on" in kwargs.keys():
                  self.depends_on = kwargs['depends_on']
                  self.depends = self.depends_on.keys()
                  self.cond = self.depends_on.values()[0]

            inits = self.inits; model=self.model; depends=self.depends;
            ntrials, ftol, xtol, maxfun, niter = self.get_fitparams()

            if not self.isprepared:
                  # initialize data storage objects
                  self.prepare_fit()

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

            if save:
                  self.qp.gqp_fits.to_csv(savepth+model+"_qpfits.csv", index=False)
                  self.pstop.pstop_fits.to_csv(savepth+self.model+"_scurve.csv", index=False)
                  self.popt.to_csv(savepth+model+"_popt.csv")



      def get_observed(df, prob=np.array([10, 30, 50, 70, 90])):

            rt = df.rt

            inter_prob = [prob[0]-0] + [prob[i] - prob[i-1] for i in range(1,len(prob))] + [100 - prob[-1]]
            rtquant = mq(rt, prob=prob*.01)
            observed = np.ceil(np.array(inter_prob)*.01*len(rt)).astype(int)
            n_obs = np.sum(observed)

            return [observed, rtquant, n_obs]


      def get_expected(simdf, obs_quant, n_obs):

            simrt = simdf.rt
            q = obs_quant

            first = np.array([len(simrt[simrt.between(simrt.min(), q[0])])/len(simrt)])*n_obs
            middle = np.array([len(simrt[simrt.between(q[i-1], q[i])])/len(simrt) for i in range(1,len(q))])*n_obs
            last = np.array([len(simrt[simrt.between(q[-1], simrt.max())])/len(simrt)])*n_obs

            expected = np.ceil(np.hstack([first, middle, last]))
            return expected



      @jit
      def store_recost(self, indxi, label, params, yhat):
            # get predictions and store optimized parameter set
            popti = pd.Series({k:params[k] for k in self.popt.columns})
            self.popt.iloc[self.i, 2:] = popti

            # fill df: [cond] [go/ssgo], [cor/err rt quantiles], [prob corr/err response]
            self.qp.gqp_fits.iloc[self.i, 2:] = yhat[:6]
            self.qp.eqp_fits.iloc[self.i, 2:] = yhat[6:12]
            self.pstop.pstop_fits.iloc[self.i, 2:] = yhat[12:]
            self.i+=1



      @jit
      def store_procost(self, indxi, label, params, yhat):
            # get predictions and store optimized parameter set

            popti = pd.Series({k:params[k] for k in self.popt.columns})

            # fill df: [cond] [go/ssgo], [cor/err rt quantiles], [prob corr/err response]
            self.qp.gqp_fits.loc[self.i,:] = yhat[:6]
            self.popt.loc[self.i,:] = popti
            self.i+=1
