#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.stats.mstats import mquantiles as mq
from radd import fit, fit_flat
import matplotlib.pyplot as plt
from radd.misc.messages import saygo
from radd.fit import Simulator
from radd.utils import rt_quantiles, rangl_data, resample_data


class Base(object):

      def __init__(self, kind='reactive', inits=None, data=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, style='RADD', *args, **kws):

            self.data = data
            self.inits = inits
            self.kind = kind
            self.style = style
            self.depends_on = depends_on
            self.fit_on = fit_on
            if fit_whole_model:
                  self.fit_flat=True

            if self.depends_on != None:
                  self.cond=depends_on.values()[0]
                  self.labels=data[self.cond].unique()
                  self.ncond=len(self.labels)
            else:
                  self.cond='flat'
                  self.labels=None
                  self.ncond=1

            if self.kind=='reactive':
                  ssd = self.data.query('ttype=="stop"').ssd.unique()
                  self.delays = sorted(ssd.astype(np.int))
                  self.ssd = np.array(self.delays)*.001
            elif self.kind=='proactive':
                  self.pGo = sorted(self.data.pGo.unique())
                  self.ssd=np.array([.450])

            if self.fit_on=='bootstrap':
                  self.indx = range(niter)
                  self.ifx = resample_data
            else:
                  self.indx = list(self.data.idx.unique())
                  self.ifx = rangl_data


      def rangl_data(self, data, re_cut=.650, pro_cut=.54502, kind='reactive', prob=np.array([.1, .3, .5, .7, .9])):

            if kind == 'reactive':
                  gac = data.query('ttype=="go"').acc.mean()
                  sacc = data.query('ttype=="stop"').groupby('ssd').mean()['acc'].values
                  grt = data.query('ttype=="go" & acc==1').rt.values
                  ert = data.query('response==1 & acc==0').rt.values
                  gq = mq(grt, prob=prob)
                  eq = mq(ert, prob=prob)
                  return np.hstack([gac, sacc, gq*10, eq*10])

            elif kind=='proactive':
                  return 1-data.response.mean()


      def rt_quantiles(self, data, cutoff=.560, split='HiLo', prob=np.arange(0.1,1.0,0.2)):
            rtq = []
            godf = data[data.response==1]
            if split=='HiLo' and 'HiLo' not in godf.columns:
                  godf['HiLo']=['x']
                  godf[godf['pGo']<=.5, 'HiLo'] = 'Lo'
                  godf[godf['pGo']>.5, 'HiLo'] = 'Hi'
            if split != None:
                  splitdf = godf.groupby(split)
            else:
                  rts = godf[godf.rt<=cutoff].rt.values
                  return mq(rts, prob=prob)*10

            for c, df in splitdf:
                  rts = df[df.rt<=cutoff].rt.values
                  rtq.append(mq(rts, prob=prob)*10)

            return np.hstack(rtq)


      def resample_data(self, data, n=120, kind='reactive'):

            df=data.copy(); bootlist=list()
            if n==None: n=len(df)

            if kind=='reactive':
                  for ssd, ssdf in df.groupby('ssd'):
                        boots = ssdf.reset_index(drop=True)
                        orig_ix = np.asarray(boots.index[:])
                        resampled_ix = rwr(orig_ix, get_index=True, n=n)
                        bootdf = ssdf.irow(resampled_ix)
                        bootlist.append(bootdf)
                        #concatenate and return all resampled conditions
                        return rangl_re(pd.concat(bootlist))
            else:
                  boots = df.reset_index(drop=True)
                  orig_ix = np.asarray(boots.index[:])
                  resampled_ix = rwr(orig_ix, get_index=True, n=n)
                  bootdf = df.irow(resampled_ix)
                  bootdf_list.append(bootdf)
                  return rangl_pro(pd.concat(bootdf_list), rt_cutoff=rt_cutoff)


      def set_fitparams(self, ntrials=10000, ftol=1.e-4, xtol=1.e-4, maxfev=1000, niter=500, log_fits=True, disp=True, prob=np.array([.1, .3, .5, .7, .9]), get_params=False, tb=None, **kwgs):

            if not hasattr(self, 'fitparams'):
                  self.fitparams={}

            if self.kind=='reactive':
                  fwts = self.wts.reshape(self.ncond, 16).mean(axis=0)
                  if tb is None:
                        self.tb = .650

            elif self.kind=='proactive':
                  nogo = self.wts[:len(self.pGo)].mean()
                  quant = self.wts[len(self.pGo):].reshape(2, len(prob)).mean(axis=0)
                  fwts = np.hstack([nogo, quant])
                  if tb is None:
                        self.tb = .560
            self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'disp':disp, 'ftol':ftol, 'xtol':xtol, 'niter':niter, 'prob':prob, 'log_fits':log_fits, 'ssd':self.ssd, 'tb':self.tb, 'wts':self.wts, 'ncond':self.ncond, 'flat_wts':fwts}

            if get_params:
                  return self.fitparams


      def load_default_inits(self):

            if self.kind=='reactive':
                  self.inits = {'a': 0.44, 'ssv': 0.947, 'tr': 0.3049, 'v': 1.1224, 'z': 0.15}
            elif self.kind=='proactive':
                  self.inits = {}




class Optimizer(Base):

      def __init__(self, dataframes=None, kind='reactive', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, style='RADD', method='nelder', pc_map=None, wts=None, *args, **kws):

            dfs=dict(deepcopy(dataframes))
            self.observed=dfs['observed']
            self.fits=dfs['fits']
            self.fitinfo=dfs['fitinfo']
            self.avg_y=dfs['avg_y']
            self.flat_y=dfs['flat_y']
            self.dat=dfs['dat']
            self.data = dfs['data']
            self.pc_map = pc_map
            self.wts=wts
            self.method=method

            super(Optimizer, self).__init__(kind=kind, data=self.data, fit_on=fit_on, depends_on=depends_on, inits=inits, style=style, fit_whole_model=fit_whole_model, niter=niter)


      def optimize(self, save=True, savepth='./', log_fits=True, disp=True, xtol=1.e-4, ftol=1.e-4, maxfev=1000, ntrials=10000, niter=500, prob=np.array([.1, .3, .5, .7, .9]), tb=None):

            self.set_fitparams(xtol=xtol, ftol=xtol, maxfev=maxfev, ntrials=ntrials, niter=niter, disp=disp, log_fits=log_fits, prob=prob)

            self.simulator = Simulator(fitparams=self.fitparams, kind=self.kind, style=self.style, inits=self.inits, method=self.method, pc_map=self.pc_map)

            if self.fit_on=='average':
                  self.yhat, self.fitinfo, self.popt = self.__opt_routine__(self.avg_y)
                  return self.yhat, self.fitinfo, self.popt
            else:
                  self.__indx_optimize__(save=save, savepth=savepth)
                  return self.fits, self.fitinfo, self.popt


      def __indx_optimize__(self, save=True, savepth='./'):

            ri=0; radv=self.ncond
            pcols=self.fitinfo.columns
            for i, y in enumerate(self.dat):
                  # rejoin grouped y vector, with mean eq
                  # this flattens the full vector, gets
                  # reshaped before weights are applied
                  yhat, finfo, popt = self.__opt_routine__(y)
                  self.fitinfo.iloc[i]=pd.Series({pc: finfo[pc] for pc in pcols})

                  if self.kind=='reactive':
                        self.fits.iloc[ri:ri+radv, radv:] = yhat
                        ri+=radv
                  else:
                        self.fits.iloc[i] = yhat
                  if save:
                        self.fits.to_csv(savepth+"fits.csv")
                        self.fitinfo.to_csv(savepth+"fitinfo.csv")

            self.popt=self.fitinfo.mean()


      def __opt_routine__(self, y):

            p = dict(deepcopy(self.inits))
            if self.fit_flat:
                  to_fit = [self.flat_y, y]
                  flat=[1, 0]
                  ncond=[1, self.ncond]
            else:
                  to_fit = [y]
                  flat=[0]
                  ncond=[self.ncond]

            for i, yy in enumerate(to_fit):
                  self.simulator.ncond=ncond[i]
                  yhat, finfo, popt = self.simulator.optimize_theta(y=yy, inits=p, flat=flat[i])
                  p = deepcopy(popt)

            return yhat, finfo, popt



class Model(Base):


      def __init__(self, data=pd.DataFrame, kind='reactive', inits=None, fit_on='subjects', depends_on=None, niter=50, fit_whole_model=True, style='RADD', weight_presp=True, *args, **kws):

            self.data=data
            self.weight_presp=weight_presp
            self.is_optimized=False

            super(Model, self).__init__(data=self.data, inits=inits, fit_on=fit_on, depends_on=depends_on, niter=niter, fit_whole_model=fit_whole_model, style=style, kind=kind)

            self.prepare_fit()


      def optimize(self, save=True, savepth='./', ntrials=10000, ftol=1.e-4, xtol=1.e-4, maxfev=1000, niter=500, log_fits=True, disp=True, prob=np.array([.1, .3, .5, .7, .9])):

            """
            Main link between Model class and fitting methods in Optimizer class
            see Optimizer method optimize()
            """

            self.opt = Optimizer(kind=self.kind, inits=self.inits, style=self.style, depends_on=self.depends_on, fit_on=self.fit_on, dataframes=self.df_dict, wts=self.wts, pc_map=self.pc_map)

            self.fits, self.fitinfo, self.popt = self.opt.optimize(save=save, savepth=savepth, log_fits=log_fits, disp=disp, xtol=xtol, ftol=ftol, maxfev=maxfev, ntrials=ntrials, niter=niter, prob=prob)

            self.is_optimized=True


      def simulate(self):

            try:
                  theta=self.popt
            except Exception:
                  theta=self.inits
                  self.set_fitparams()

            self.simulator = Simulator(fitparams=self.fitparams, kind=self.kind, style=self.style, inits=self.inits, pc_map=self.pc_map)
            theta = self.simulator.vectorize_params(theta, sim_info=False, as_dict=True)

            if self.kind=='reactive':
                  dvg, dvs = self.simulator.core_radd(theta)
                  yhat = self.simulator.analyze_reactive(dvg, dvs, theta)
            elif self.kind=='proactive':
                  dvg = self.simulator.pro_radd(theta)
                  yhat = self.simulator.analyze_proactive(dvg, theta)

            return yhat


      def prepare_fit(self):

            if 'trial_type' in self.data.columns:
                  self.data.rename(columns={'trial_type':'ttype'}, inplace=True)

            # if numeric, sort first then convert to string
            if not isinstance(self.labels[0], str):
                  ints = sorted([int(l*100) for l in self.labels])
                  self.labels = [str(intl) for intl in ints]
            else:
                  self.labels = sorted(self.labels)

            params = sorted(self.inits.keys())
            self.pc_map = {}
            for d in self.depends_on.keys():
                  params.remove(d)
                  params_dep = ['_'.join([d, l]) for l in self.labels]
                  self.pc_map[d] = params_dep
                  params.extend(params_dep)
            qp_cols = self.__get_header__(params)
            self.__make_dataframes__(qp_cols)
            self.get_wts()
            self.is_prepared=saygo(depends_on=self.depends_on, labels=self.labels, kind=self.kind, fit_on=self.fit_on)


      def __make_dataframes__(self, qp_cols):

            cond = self.cond; ncond = self.ncond
            data = self.data; indx = self.indx
            labels = self.labels

            ic_grp = data.groupby(['idx', cond])
            c_grp = data.groupby([cond])
            i_grp = data.groupby(['idx'])

            if self.fit_on=='bootstrap':
                  self.dat = np.vstack([i_grp.apply(self.ifx, kind=self.kind).values for i in indx]).unstack()

            if self.kind=='proactive':
                  datdf = ic_grp.apply(rangl_data, kind=self.kind).unstack()
                  rtdat = pd.DataFrame(np.vstack(i_grp.apply(rt_quantiles).values), index=indx)
                  rtdat[rtdat<1] = np.nan
                  rts_flat = pd.DataFrame(np.vstack(i_grp.apply(rt_quantiles, split=None).values), index=indx)
                  self.observed = pd.concat([datdf, rtdat], axis=1)
                  self.observed.columns = qp_cols
                  self.avg_y = self.observed.mean().values
                  self.flat_y=np.append(datdf.mean().mean(), rts_flat.mean())
                  dat = self.observed.values.reshape((len(indx), len(qp_cols)))
                  fits = pd.DataFrame(np.zeros_like(dat), columns=qp_cols, index=indx)

            elif self.kind=='reactive':
                  datdf = ic_grp.apply(rangl_data, kind=self.kind).unstack().unstack()
                  indxx = pd.Series(indx*ncond, name='idx')
                  obs = pd.DataFrame(np.vstack(datdf.values), columns=qp_cols, index=indxx)
                  obs[cond]=np.sort(labels*len(indx))
                  self.observed = obs.sort_index().reset_index()
                  self.avg_y = self.observed.groupby(cond).mean().loc[:,qp_cols[0] : qp_cols[-1]].values
                  self.flat_y = self.observed.loc[:, qp_cols[0] : qp_cols[-1]].mean().values
                  dat = self.observed.loc[:,qp_cols[0]:qp_cols[-1]].values.reshape(len(indx),ncond,16)
                  fits = pd.DataFrame(np.zeros((len(indxx),len(qp_cols))), columns=qp_cols, index=indxx)

            fitinfo = pd.DataFrame(columns=self.infolabels, index=indx)
            self.df_dict = {'data':self.data, 'flat_y':self.flat_y, 'avg_y':self.avg_y, 'fitinfo': fitinfo, 'fits': fits, 'observed': self.observed, 'dat':dat}



      def __get_header__(self, params=None, prob=np.array([.1, .3, .5, .7, .9])):

            info = ['nfev','chi','rchi','AIC','BIC','CNVRG']
            cq = ['c'+str(int(n*100)) for n in prob]

            if self.kind == 'reactive':
                  cq = ['c'+str(int(n*100)) for n in prob]
                  eq = ['e'+str(int(n*100)) for n in prob]
                  qp_cols = ['Go'] + self.delays + cq + eq
            else:
                  hi = ['hi'+str(int(n*100)) for n in prob]
                  lo = ['lo'+str(int(n*100)) for n in prob]
                  qp_cols = self.labels + hi + lo

            if params is not None:
                  self.infolabels = params + info

            return qp_cols


      def get_wts(self):
            """
            wtc: weights applied to correct rt quantiles in cost f(x)
                  * P(R | No SSD)j * sdj(.5Qj, ... .95Qj)
            wte: weight applied to error rt quantiles in cost f(x)
                  * P(R | SSD) * sd(.5eQ, ... .95eQ)
            """

            if self.kind == 'reactive':
                  obs_var = self.observed.groupby(self.cond).sem().loc[:,'Go':]
                  qvar = obs_var.values[:,6:]
                  pvar = obs_var.values[:,:6]
                  go = self.data.query('ttype=="go"').response.mean()
                  st = self.data.query('ttype=="stop"').response.mean()

                  sq_ratio = (np.median(qvar, axis=1)/qvar.T).T
                  wt_go = (go*sq_ratio[:, :5].T).T
                  wt_err = (st*sq_ratio[:, -5:].T).T
                  qwts = np.hstack(np.vstack(zip(wt_go, wt_err))).reshape(self.ncond, 10)

                  if self.weight_presp:
                        pwts = (np.median(pvar, axis=1)/pvar.T).T
                  else:
                        pwts = np.ones(len(self.ssd)+1)

                  self.wts = np.hstack([np.append(p, w) for p, w in zip(pwts, qwts)])


            elif self.kind == 'proactive':
                  sd = self.observed.sem()
                  sdhi = sd.loc['hi10':'hi90'].values
                  sdlo = sd.loc['lo10':'lo90'].values
                  presp = 1-self.observed.mean().loc['0':'100'].values

                  if self.weight_presp:
                        sdp = sd.loc['0':'100'].values
                        pwts = np.median(sdp)/sdp
                  else:
                        pwts = np.ones(len(self.pGo))

                  qwts_hi = presp[3:].mean()*(np.median(sdhi)/sdhi)
                  qwts_lo = presp[:3].mean()*(np.median(sdlo)/sdlo)

                  self.wts = np.hstack([pwts, qwts_hi, qwts_lo])
