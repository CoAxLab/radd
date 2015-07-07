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
            self.prob=np.array([.1, .3, .5, .7, .9])
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



      def global_opt(self, ):

            if not self.isprepared:
                  self.prepare_fit()

            ntrials, ftol, xtol, maxfun, niter = self.get_fitparams()
            inits = self.inits
            y = self.dat.mean(axis=0)

            if self.kind=='reactive':
                  self.inits, yhat = fitre.fit_reactive_model(y, inits=inits, ntrials=ntrials, model=self.model, depends=['xx'], maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=1)
            elif self.kind=='proactive':
                  self.inits, yhat = fitpro.fit_proactive_model(y, inits=inits, ntrials=ntrials, model=model, depends=['xx'], maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=1)



      def prepare_fit(self):

            print "preparing %s model using %s method" % (self.kind, self.fit)

            inits=self.inits; data=self.data; cond=self.cond; indx = self.indx
            labels = data[cond].unique(); nlabels = len(labels);
            index = range(nlabels*len(indx))

            delays = sorted(data.query('trial_type=="stop"').ssd.unique().astype(np.int))

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
            self.fits = pd.concat([ixdf, pd.DataFrame(np.zeros_like(self.dat), columns=qp_cols)], axis=1)
            self.popt = pd.concat([ixdf, pd.DataFrame(columns=popt_cols, index=ixdf.index)], axis=1)
            self.isprepared = True


      def run_model(self, save=False, savepth='./', live_update=True, all_params=1, disp=False, prepare=False, **kwargs):

            if "depends_on" in kwargs.keys():
                  self.depends_on = kwargs['depends_on']
                  self.depends = self.depends_on.keys()
                  self.cond = self.depends_on.values()[0]

            inits = self.inits; model=self.model; depends=self.depends;
            ntrials, ftol, xtol, maxfun, niter = self.get_fitparams()

            if not self.isprepared:
                  # initialize data storage objects
                  self.prepare_fit()


            if fit_global:
                  global_opt()

            for i, y in enumerate(self.dat):

                  if self.kind=='reactive':
                        params, yhat = fitre.fit_reactive_model(y, inits=inits, ntrials=ntrials, model=model, depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=all_params, disp=disp)

                  elif self.kind=='proactive':
                        inits['pGo']=cdf.pGo.mean()
                        params, yhat = fitpro.fit_proactive_model(y, inits=inits, ntrials=ntrials, model=model, depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=0, disp=disp)
                        #self.store_procost(indx[i], label, params, yhat)

                  self.popt.iloc[self.i, 2:] = popti
                  self.fits.iloc[self.i, 2:] = yhat

                  if live_update:
                        self.fits.to_csv(savepth+model+"_fits.csv", index=False)
                        self.popt.to_csv(savepth+model+"_popt.csv", index=False)

            if save:
                  self.fits.to_csv(savepth+model+"_fits.csv", index=False)
                  self.popt.to_csv(savepth+model+"_popt.csv", index=False)
                  self.observed.to_csv(savepth+model+"_data.csv", index=False)



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


      def get_obs_quant_counts(df, prob=np.array([.10, .30, .50, .70, .90])):

            if type(df) == pd.Series:
                  rt=df.copy()
            else:
                  rt=df.rt.copy()

            inter_prob = [prob[0]-0] + [prob[i] - prob[i-1] for i in range(1,len(prob))] + [1.00 - prob[-1]]
            obs_quant = mq(rt, prob=prob)
            observed = np.ceil(np.array(inter_prob)*len(rt)*.94).astype(int)

            return observed, obs_quant



      def get_exp_counts(simdf, obs_quant, n_obs, prob=np.array([.10, .30, .50, .70, .90])):

            if type(simdf) == pd.Series:
                  simrt=simdf.copy()
            else:
                  simrt = simdf.rt.copy()


            exp_quant = mq(simrt, prob); oq = obs_quant
            expected = np.ceil(np.diff([0] + [pscore(simrt, oq_rt)*.01 for oq_rt in oq] + [1]) * n_obs)

            return expected, exp_quant


      def calc_quant_weights(rtvec, quants):

            rt=rtvec.copy()
            q=quants.copy()

            first = rt[rt.between(rt.min(), q[0])].std()
            rest = [rt[rt.between(q[i-1], q[i])].std() for i in range(1,len(q))]
            #last = rt[rt.between(q[-1], rt.max())].std()
            sdrt = np.hstack([first, rest])

            wt = (sdrt[np.ceil(len(sdrt)/2)]/sdrt)**2

            return wt



      @jit
      def store_recost(self, indxi, label, params, yhat):
            # get predictions and store optimized parameter set
            popti = pd.Series({k:params[k] for k in self.popt.columns})
            self.popt.iloc[self.i, 2:] = popti
            self.fits.iloc[self.i, 2:] = yhat

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




      def visualize_fits(self):

            gacc = self.observed['GoAcc'].mean()
            sacc = self.observed.loc[:, 200:400].mean()
            fit_gacc = self.fits['GoAcc'].mean()
            fit_sacc = self.fits.loc[:, 200:400].mean()

            gq = self.observed.loc[:, 'c5':'c95'].mean()
            eq = self.observed.loc[:, 'e5':'e95'].mean()
            fit_gq = self.fits.loc[:, 'c5':'c95'].mean()
            fit_eq = self.fits.loc[:, 'e5':'e95'].mean()

            sns.set_context('notebook', font_scale=1.6)
            f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))

            # Fit RT quantiles to KDE function in radd.utils
            quant_list = [gq, fit_gq, eq, fit_eq]
            kdefits = [utils.kde_fit_quantiles(q) for q in quant_list]

            sns.kdeplot(kdefits[0], cumulative=True, label='data gQP', linestyle='-', color=gpal(2)[0], ax=ax1, linewidth=3.5)

            sns.kdeplot(kdefits[1], cumulative=True, label='model gQP', linestyle='--', color=gpal(2)[1], ax=ax1, linewidth=3.5)

            sns.kdeplot(kdefits[2], cumulative=True, label='data eQP', linestyle='-', color=rpal(2)[0], ax=ax1, linewidth=3.5)

            sns.kdeplot(kdefits[3], cumulative=True, label='model eQP', linestyle='--', color=rpal(2)[1], ax=ax1, linewidth=3.5)

            ax1.set_xlim(4.3, 6.5)
            ax1.set_ylabel('P(RT<t)')
            ax1.set_xlabel('RT (s)')
            ax1.set_ylim(-.05, 1.05)
            ax1.set_xticklabels(ax1.get_xticks()*.1)

            # Plot observed and predicted stop curves
            utils.scurves([y[20:25], yhat[20:25]], labels=['data Stop', 'model Stop'], colors=bpal(2), linestyles=['-','--'], ax=ax2)
            plt.tight_layout()
            sns.despine()
