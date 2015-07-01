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
        ntrials, ftol, xtol, maxfun, niter = self.get_fitparams

        self.inits, self.yhat = run_reactive_model(y, inits=inits, ntrials=ntrials, model=model,
                        depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=1)

    def build_fit_store(self):

        inits=self.inits; data=self.data; cond=self.cond; indx = self.indx

        labels = data[cond].unique()
        nlabels = len(labels)
        index = range(nlabels*len(indx))
        popt_cols=sum([['indx'], [cond], ['chi'], inits.keys()], [])
        qp_cols=['indx', cond, 'ttype', '5q', '25q', '50q', '75q', '95q', 'presp']

        self.gqp_fits = pd.DataFrame(columns=qp_cols, index=index)
        self.popt = pd.DataFrame(columns=popt_cols, index=index)

        if self.kind=='reactive':
            delays = list(data.query('trial_type=="stop"').ssd.unique())
            pstop_cols=sum([['indx'], [cond], delays],[])
            self.eqp_fits = pd.DataFrame(columns=qp_cols, index=index)
            self.pstop_fits = pd.DataFrame(columns=pstop_cols, index=index)



    def run_model(self, save=False, savepth='./', live_update=True, disp=False):

        if self.fit=='bootstrap': self.indx = range(self.niter)
        elif self.fit=='subjects': self.indx = self.data.idx.unique()

        # initialize data storage objects
        self.build_fit_store()
        self.live_update=live_update

        for label, cdf in self.data.groupby(self.cond):
            self.fit_indx(cdata=cdf, label=label, savepth=savepth, disp=disp)

        if save:
            self.gqp_fits.to_csv(savepth+self.model+"_qpfits.csv", index=False)
            #self.pstop_fits.to_csv(savepth+self.model+"_scurve.csv", index=False)
            self.popt.to_csv(savepth+self.model+"_popt.csv")


    def fit_indx(self, cdata, label, savepth='./', disp=False):

        cdf = cdata.copy(); inits = self.inits; model=self.model; depends=self.depends;
        ntrials, ftol, xtol, maxfun, niter = self.get_fitparams()

        for i, indxi in enumerate(self.indx):




            if self.kind=='reactive':
                if 'bootstrap': y = utils.resample_reactive(cdf)
                else: y = utils.rangl_re(cdf[cdf['idx']==indxi])
                params, yhat = fitre.fit_reactive_model(y, inits=inits, ntrials=ntrials, model=model,
                        depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=0, disp=disp)
                self.store_recost(indxi, label, params, yhat)




            elif self.kind=='proactive':
                inits['pGo']=cdf.pGo.mean()
                if 'bootstrap': y = utils.resample_proactive(cdf)
                #else: y = utils.rangl_pro(cdf[cdf['idx']==indxi])
                params, yhat = fitpro.fit_proactive_model(y, inits=inits, ntrials=ntrials, model=model,
                        depends=depends, maxfun=maxfun, ftol=ftol, xtol=xtol, all_params=0, disp=disp)
                self.store_procost(indxi, label, params, yhat)




            if self.live_update:
                self.gqp_fits.to_csv(savepth+self.model+"_gqp.csv", index=False)
                self.popt.to_csv(savepth+self.model+"_popt.csv", index=False)

                if self.kind=='reactive':
                    self.eqp_fits.to_csv(savepth+self.model+"_eqp.csv", index=False)
                    self.pstop_fits.to_csv(savepth+self.model+"_pstop.csv", index=False)




    def store_recost(self, indxi, label, params, yhat):
        # get predictions and store optimized parameter set
        params['indx']=indxi; params[self.cond]=label
        gqpi = sum([[indxi], [label], ['go'], list(yhat[:5]*.1), list(yhat[5])], [])
        eqpi = sum([[indxi], [label], ['stop'], list(yhat[6:10]*.1), list(yhat[10])], [])
        pstopi = sum([[indxi], [label], list(yhat[11:])], [])
        popti = pd.Series({k:params[k] for k in self.popt.columns})

        # fill df: [cond] [go/ssgo], [cor/err rt quantiles], [prob corr/err response]
        self.gqp_fits.loc[self.i,:] = gqpi
        self.eqp_fits.loc[self.i,:] = eqpi
        self.pstop_fits.loc[self.i,:] = pstopi
        self.popt.loc[self.i,:] = popti
        self.i+=1

    def store_procost(self, indxi, label, params, yhat):
        # get predictions and store optimized parameter set
        params['indx']=indxi; params[self.cond]=label
        gqpi = sum([[indxi], [label], ['go'], list(yhat[:5]*.1), yhat[5]], [])
        popti = pd.Series({k:params[k] for k in self.popt.columns})

        # fill df: [cond] [go/ssgo], [cor/err rt quantiles], [prob corr/err response]
        self.gqp_fits.loc[self.i,:] = gqpi
        self.popt.loc[self.i,:] = popti
        self.i+=1
