#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os, re



# 
#
# def init_distributions_XXXX(pkey, bounds, tb=.65, kind='radd', nrvs=25, loc=None, scale=None):
#       """ sample random parameter sets to explore global minima (called by
#       Optimizer method __hop_around__())
#       """
#       mu_defaults = {'a':.15, 'tr':.29, 'v':.9, 'ssv':-.9, 'z':.1, 'xb':2.5, 'sso':.15}
#       sigma_defaults = {'a':.35, 'tr':.1, 'v':.35, 'ssv':.35, 'z':.05, 'xb':1, 'sso':.01}
#
#       if pkey=='si':
#             return .01
#
#       if 'race' in kind or 'iact' in kind:
#             mu_defaults['ssv']=abs(mu_defaults['ssv'])
#       if loc is None:
#             loc = mu_defaults[pkey]
#       if scale is None:
#             scale = sigma_defaults[pkey]
#
#       # init and freeze dist shape
#       if pkey in ['z', 'xb', 'sso']:
#             dist = norm(loc, scale)
#       elif pkey in ['a', 'tr', 'v', 'ssv']:
#             dist = gamma(1, loc, scale)
#
#       # generate random variates
#       rvinits = dist.rvs(nrvs)
#       while rvinits.min()<=bounds[0]:
#             # apply lower limit
#             ix = rvinits.argmin()
#             rvinits[ix] = dist.rvs()
#       while rvinits.max()>=bounds[1]:
#             # apply upper limit
#             ix = rvinits.argmax()
#             rvinits[ix] = dist.rvs()
#
#       return rvinits
#

# class Optimizer(RADDCore):
#
#       """ Optimizer class acts as interface between Model and Simulator (see fit.py) objects.
#       Structures fitting routines so that Models are first optimized with the full set of
#       parameters free, data collapsing across conditions.
#
#       The fitted parameters are then used as the initial parameters for fitting conditional
#       models with only a subset of parameters are left free to vary across levels of a given
#       experimental conditionself.
#
#       Parameter dependencies are specified when initializing Model object via
#       <depends_on> arg (i.e.{parameter: condition})
#
#       Handles fitting routines for models of average, individual subject, and bootstrapped data
#       """
#
#       def __init__(self, dframes=None, fitparams=None, kind='radd', inits=None, fit_on='average', depends_on=None, niter=50, fit_whole_model=True, method='nelder', pc_map=None, wts=None, multiopt=False, global_method='basinhopping', data_style='re', *args, **kws):
#
#             self.multiopt=multiopt
#             self.fit_on = fit_on
#             self.data=dframes['data']
#             self.fitparams=fitparams
#             self.global_method=global_method
#             self.kind=kind
#             self.xbasin=[]
#             self.dynamic=self.fitparams['dynamic']
#             nq = len(self.fitparams['prob'])
#             nc = self.fitparams['ncond']
#
#             if fit_on in ['subjects', 'bootstrap']:
#                   self.indx_list = dframes['observed'].index
#                   self.fits = dframes['fits']
#                   self.fitinfo = dframes['fitinfo']
#
#                   if fit_on=='subjects':
#                         self.dat=dframes['dat']
#                   elif fit_on=='bootstrap':
#                         self.dat=dframes['boot']
#
#                   if data_style=='re':
#                         self.get_flaty = lambda x: x.mean(axis=0)
#                   elif data_style=='pro':
#                         self.get_flaty = lambda x:np.hstack([x[:nc].mean(),x[nc:].reshape(2,nq).mean(axis=0)])
#
#             self.method=method
#             self.avg_y=self.fitparams['avg_y'].flatten()
#             self.avg_wts=self.fitparams['avg_wts']
#
#             self.flat_y=self.fitparams['flat_y']
#             self.flat_wts=self.fitparams['flat_wts']
#
#             self.pc_map=pc_map
#             self.pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
#             self.pvc=deepcopy(['a', 'tr', 'v', 'xb'])
#
#             super(Optimizer, self).__init__(kind=kind, data=self.data, fit_on=fit_on, depends_on=depends_on, inits=inits, fit_whole_model=fit_whole_model, niter=niter)
#


      #
      #
      # def set_fitparams(self, ntrials=10000, tol=1.e-10, maxfev=5000, niter=500, disp=True, prob=np.array([.1, .3, .5, .7, .9]), get_params=False, **kwgs):
      #
      #       if not hasattr(self, 'fitparams'):
      #             self.fitparams={}
      #       self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'disp':disp, 'tol':tol, 'niter':niter, 'prob':prob, 'tb':self.tb, 'ssd':self.ssd, 'flat_y':self.flat_y, 'avg_y':self.avg_y, 'avg_wts':self.avg_wts, 'ncond':self.ncond, 'pGo':self.pGo, 'flat_wts':self.flat_wts, 'depends_on': self.depends_on, 'dynamic': self.dynamic, 'fit_whole_model': self.fit_whole_model, 'rt_cix': self.rt_cix, 'data_style':self.data_style,  'nudge_dir':self.nudge_dir}
      #
      #       if get_params:
      #             return self.fitparams
      #


      #
      # def __make_dataframes__(self, qp_cols):
      #       """ Generates the following dataframes and arrays:
      #       ::Arguments::
      #             qp_cols:
      #                   header for observed/fits dataframes
      #       ::Returns::
      #             None (All dataframes and vectors are stored in dict and assigned
      #             as <dframes> attr)
      #       observed (DF):
      #             Contains Prob and RT quant. for each subject
      #             used to calc. cost fx weights
      #       fits (DF):
      #             empty DF shaped like observed DF, used to store simulated
      #             predictions of the optimized model
      #       fitinfo (DF):
      #             stores all opt. parameter values and model fit statistics
      #       dat (ndarray):
      #             contains all subject/boot. y vectors entered into costfx
      #       avg_y (ndarray):
      #             average y vector for each condition entered into costfx
      #       flat_y (1d array):
      #             average y vector used to initialize parameters prior to fitting
      #             conditional model. calculated collapsing across conditions
      #       """
      #
      #       cond = self.cond; ncond = self.ncond
      #       data = self.data; indx = self.indx
      #       labels = self.labels
      #
      #       ic_grp = data.groupby(['idx', cond])
      #       c_grp = data.groupby([cond])
      #       i_grp = data.groupby(['idx'])
      #
      #       if self.data_style=='re':
      #             datdf = ic_grp.apply(self.rangl_data, kind=self.kind).unstack().unstack()
      #             indxx = pd.Series(indx*ncond, name='idx')
      #             obs = pd.DataFrame(np.vstack(datdf.values),columns=qp_cols[1:], index=indxx)
      #             obs.insert(0, qp_cols[0], np.sort(labels*len(indx)))
      #
      #             self.observed = obs.sort_index().reset_index()
      #             self.avg_y = self.observed.groupby(cond).mean().values[:,1:]
      #             self.flat_y = self.observed.mean().values[1:]
      #             dat = self.observed.loc[:,qp_cols[1]:].values.reshape(len(indx),ncond,16)
      #             fits = pd.DataFrame(np.zeros((len(indxx),len(qp_cols))), columns=qp_cols, index=indxx)
      #
      #
      #       elif self.data_style=='pro':
      #             datdf = ic_grp.apply(self.rangl_data, kind=self.kind).unstack()
      #             rtdat = pd.DataFrame(np.vstack(i_grp.apply(self.rt_quantiles).values), index=indx)
      #             rtdat[rtdat<.1] = np.nan
      #             rts_flat = pd.DataFrame(np.vstack(i_grp.apply(self.rt_quantiles, split_col=None).values), index=indx)
      #             self.observed = pd.concat([datdf, rtdat], axis=1)
      #             self.observed.columns = qp_cols
      #             self.avg_y = self.observed.mean().values
      #             self.flat_y=np.append(datdf.mean().mean(), rts_flat.mean())
      #             dat = self.observed.values.reshape((len(indx), len(qp_cols)))
      #             fits = pd.DataFrame(np.zeros_like(dat), columns=qp_cols, index=indx)
      #
      #       fitinfo = pd.DataFrame(columns=self.infolabels, index=indx)
      #
      #       self.dframes = {'data':self.data, 'flat_y':self.flat_y, 'avg_y':self.avg_y, 'fitinfo': fitinfo, 'fits': fits, 'observed': self.observed, 'dat':dat}
      #

      #
#
#
# def get_wts(m, weight_by='mj'):
#       """ wtc: weights applied to correct rt quantiles in cost f(x)
#              * P(R | No SSD)j * sdj(.5Qj, ... .95Qj)
#        wte: weight applied to error rt quantiles in cost f(x)
#              * P(R | SSD) * sd(.5eQ, ... .95eQ)
#       """
#
#       nc = m.ncond; cond=m.cond;
#       if m.data_style=='re':
#             go = m.data.query('ttype=="go"').response.mean()
#             st = m.data.query('ttype=="stop"').response.mean()
#             if weight_by=='mj':
#                   qwts = analyze.reactive_mj_quanterr(df=m.data)
#             else:
#                   obsrts = m.observed.loc[:, 'c10':]
#                   qwts = np.median(obsrts.std(axis=0))/obsrts.std(axis=0)
#                   qwts = np.vstack(qwts.values.reshape(nc,5))
#             qwts = np.hstack(array([[go], [st]])*qwts)
#             pwts = array([1,1,1,1,1,1])
#             m.flat_wts = np.hstack([pwts, qwts])
#             m.avg_wts = np.tile(m.flat_wts, nc)
#       elif m.data_style=='pro':
#             upper = m.data[m.data['HL']==1].mean()['response']
#             lower = m.data[m.data['HL']==2].mean()['response']
#
#             if weight_by=='mj':
#                   qwts = analyze.proactive_mj_quanterr(df=m.data, split='HL', tb=m.tb)
#                   qwts = np.hstack(np.array([upper, lower])[:,None]*qwts)
#             else:
#                   qvar = m.observed.std().iloc[6:].values
#                   hi = qvar[:5]; lo = qvar[5:]
#                   qwts = np.hstack([upper*(hi[2]/hi), lower*(lo[2]/lo)])
#
#             pwts = array([1,1,1,1,1,1])#np.median(pvar)/pvar
#             m.avg_wts = np.hstack([pwts, qwts])
#
#             nogo = m.avg_wts[:nc].mean(); quant=m.avg_wts[nc:].reshape(2, 5).mean(axis=0)
#             m.flat_wts = np.hstack([nogo, quant])
#
#             qwts = analyze.proactive_mj_quanterr(df=m.data, split='HL', tb=m.tb)
#             qwts = np.hstack(np.array([upper, lower])[:,None]*qwts)
#             pwts = np.array([1.5,1,1,1,1,1.5])
#             pwts = np.array([1.5,1.25,1,1,1.25,1.5])
#             pwts = np.median(m.observed.std()[:6])/m.observed.std()[:6]
#             m.avg_wts = np.hstack([pwts, qwts])
#             # calculate flat weights (collapsing across conditions)
#             nogo = m.avg_wts[:nc].mean(); quant=m.avg_wts[nc:].reshape(2, 5).mean(axis=0)
#             m.flat_wts = np.hstack([nogo, quant])
#
#             ovar = m.observed.var().values
#             qvar = array(array([[upper], [lower]])*(1/ovar[nc:].reshape(2,5))).flatten()
#             m.qvar=qvar
#             m.uwts = np.concatenate([1/ovar[:nc], qvar], axis=1).flatten()
#             m.flat_uwts=np.append(m.uwts[:nc].mean(), m.uwts[nc:].reshape(2,5).mean(axis=0))
#
#       m.avg_wts, m.flat_wts = analyze.ensure_numerical_wts(m.avg_wts, m.flat_wts)
      #m.avg_wts=np.ones_like(m.avg_wts); m.flat_wts=np.ones_like(m.flat_wts)
      #
      # def diffevolution_minimizer(self, z, *params):
      #       """ find global mininum using differential evolution
      #
      #       ::Arguments::
      #             z (list):
      #                   list of slice objects or tuples
      #                   boundaries for each parameter
      #             *params:
      #                   iterable of parameter point estimates
      #       ::Returns::
      #             weighted cost
      #       """
      #
      #       p = {pkey: params[i] for i, pkey in enumerate(self.diffev_params)}
      #       yhat = self.sim_fx(p, analyze=True)
      #       cost = (yhat - self.y)*self.wts
      #       return cost.flatten()
      #
      #
      # def brute_minimizer(self, z, *params):
      #       """ find global mininum using brute force
      #       (see differential_evolution for I/O details)
      #       """
      #
      #       p = {pkey: params[i] for i, pkey in enumerate(self.brute_params)}
      #       yhat = self.sim_fx(p, analyze=True)
      #       cost = (yhat - self.y)*self.wts
      #       return cost.flatten()
      #
      #
      # def analyze_irace(self, DVg, DVs, p):
      #       """ get rt and accuracy of go and stop process for simulated
      #       conditions generated from simulate_radd
      #       """
      #       dt=self.dt; nss=self.nss; ncond=self.ncond; ssd=self.ssd;
      #       tb=self.tb; prob=self.prob; scale=self.scale; a=p['a']; tr=p['tr']
      #
      #       grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      #       ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      #       ssrt = ((np.where((DVs.max(axis=3).T>=a).T,ssd[:, None]+np.argmax((DVs.T>=a).T,axis=3)*dt,np.nan).T)).T
      #
      #       # compute RT quantiles for correct and error resp.
      #       ert = array([ertx[i] * np.ones_like(ssrt[i]) for i in range(ncond)])
      #       gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*scale for rtc in grt])
      #       eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*scale for i in range(ncond)]
      #       # Get response and stop accuracy information
      #       gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
      #       sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
      #       return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])
      #


#
#
# class Theta(dict):
#
#       """ a class that inherits from a custom dictionary emulator
#       for storing and passing information cleanly between Optimizer and
#       Simulator objects (i.e., init dict, if flat, number of conditions,
#       which methods to use etc.).
#
#       This is motivated by the fact that
#       fitting a single model often involves multiple stages at which this
#       information is relevant but non constant.
#       """
#       def __init__(self, is_flat=False, ncond=1, pc_map=None):
#
#             self.pnames=['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso']
#             self.flat_pvc=deepcopy(['a', 'tr', 'v', 'xb'])
#             self.full_pvc=list(set(self.flat_pvc).intersection(pc_map.keys()))
#             self.pc_map=pc_map
#             self.is_flat=is_flat
#             self.ncond=ncond
#             #self.__dict__ = self
#
#       def __getattr__(self, name):
#             """ get items using either of the following
#             syntaxes: v=self[k]; v=self.x
#             """
#             if name in self:
#                   return self[name]
#             else:
#                   raise AttributeError("No such attribute: " + name)
#
#       def __setattr__(self, name, value):
#             """ set items using either of the following
#             syntaxes: self[k]=v; self.x=v
#             """
#             self[name] = value
#
#       def set_params(self, inits):
#             """ store a safe copy of the init params
#             and fill ThisFit attr. dict with params
#             """
#             self.orig_inits = dict(deepcopy(inits))
#             for k,v in inits.items():
#                   self.__setattr__(k, v)
#
#       def restore_inits(self):
#             self.__clear__()
#             for k,v in self.orig_inits.items():
#                   self.__setattr__(k, v)
#
#       def flat_vectorize_params(self, dt=.001):
#             if 'si' in self.keys():
#                   self.dx=np.sqrt(self['si']*dt)
#             if 'xb' not in p.keys():
#                   self['xb']=np.ones(1)
#             for pkey in self.pvc_flat:
#                   self[pkey]=np.ones(1)*self[pkey]
#
#       def full_vectorize_params(self, dt=.001):
#             full_pvc=list(set(self.flat_pvc).intersection(pc_map.keys()))
#             if 'si' in self.keys():
#                   self.dx=np.sqrt(self['si']*dt)
#             if 'xb' not in p.keys():
#                   self['xb']=np.ones(self.ncond)
#             for pkey in self.pvc:
#                   self[pkey]=np.ones(self.ncond)*self[pkey]
#             for pkey, pkc in self.pc_map.items():
#                   if pkc[0] not in p.keys():
#                         self[pkey] = self[pkey]*np.ones(len(pkc))
#                   else:
#                         self[pkey] = array([self[pc] for pc in pkc]).astype(np.float32)
#             return p

      #
      # def mean_pgo_rts(self, p, return_vals=True):
      #       """ Simulate proactive model and calculate mean RTs
      #       for all conditions rather than collapse across high and low
      #       """
      #       import pandas as pd
      #       tb = self.tb; ncond = self.ncond
      #
      #       DVg = self.simulate_pro(p, analyze=False)
      #       gdec = self.resp_up(DVg, p['a'])
      #
      #       rt = self.RT(p['tr'], gdec)
      #       mu = np.nanmean(rt, axis=1)
      #       ci = pd.DataFrame(rt.T).sem()*1.96
      #       std = pd.DataFrame(rt.T).std()
      #
      #       self.pgo_rts = {'mu': mu, 'ci': ci, 'std':std}
      #       if return_vals:
      #             return self.pgo_rts
      #
      #


      #
      #
      #
      #
      # def analyze_irace(self, DVg, DVs, p):
      #       """ get rt and accuracy of go and stop process for simulated
      #       conditions generated from simulate_radd
      #       """
      #       dt=self.dt; nss=self.nss; ncond=self.ncond; ssd=self.ssd;
      #       tb=self.tb; prob=self.prob; scale=self.scale; a=p['a']; tr=p['tr']
      #
      #       grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      #       ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
      #       ssrt = ((np.where((DVs.max(axis=3).T>=a).T,ssd[:, None]+np.argmax((DVs.T>=a).T,axis=3)*dt,np.nan).T)).T
      #
      #       # compute RT quantiles for correct and error resp.
      #       ert = array([ertx[i] * np.ones_like(ssrt[i]) for i in range(ncond)])
      #       gq = np.vstack([mq(rtc[rtc<tb], prob=prob)*scale for rtc in grt])
      #       eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob)*scale for i in range(ncond)]
      #       # Get response and stop accuracy information
      #       gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
      #       sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
      #       return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])
      #



      #
      # def diffevolution_minimizer(self, z, *params):
      #       """ find global mininum using differential evolution
      #
      #       ::Arguments::
      #             z (list):
      #                   list of slice objects or tuples
      #                   boundaries for each parameter
      #             *params:
      #                   iterable of parameter point estimates
      #       ::Returns::
      #             weighted cost
      #       """
      #
      #       p = {pkey: params[i] for i, pkey in enumerate(self.diffev_params)}
      #       yhat = self.sim_fx(p, analyze=True)
      #       cost = (yhat - self.y)*self.wts
      #       return cost.flatten()
      #
      #
      # def brute_minimizer(self, z, *params):
      #       """ find global mininum using brute force
      #       (see differential_evolution for I/O details)
      #       """
      #
      #       p = {pkey: params[i] for i, pkey in enumerate(self.brute_params)}
      #       yhat = self.sim_fx(p, analyze=True)
      #       cost = (yhat - self.y)*self.wts
      #       return cost.flatten()
      #


      #
      # def get_rts(self, DV, p):
      #
      #       nss = self.nss; prob = self.prob
      #       ssd = self.ssd; tb = self.tb
      #       nc = self.ncond; nssd=self.nssd
      #
      #       DVg, DVs = DV
      #
      #       gdec = self.resp_up(DVg, p['a'])
      #       gort = self.RT(p['tr'], gdec)
      #       sdec = self.resp_lo(DVs)
      #       ssrt = self.RT(ssd, sdec)
            #ert = gort[:,:nss][:, None]*np.ones_like(ssrt)
            #go_acc = np.where(gort<tb, 1, 0)
            #sacc = np.where(ert<ssrt, 0, 1)
            #ert=ert[sacc]
            #gort = gort[go_acc]

            #return [gort, ssrt]#, ssrt]

            #if 'pro' in self.kind:
            #      ix=self.rt_cix;
            #      hi = gort[:ix]
            #      lo = gort[ix:]
            #      hi_rt = hi[np.where(hi<tb, 1, 0)]
            #      low_rt = lo[np.where(lo<tb, 1, 0)]
            #      return [hi_rt, low_rt]
            #
            #elif self.kind=='iact':
            #      ssDVg=DV[1]
            #      nss_di = int(nss/nssd)
            #      sscancel = ssd + p['sso']
            #      # Go process (No SS Trials)
            #      gort = self.RT(p['tr'], gdec)
            #      # Go process SS Trials
            #      ssgdec = self.ss_resp_up(ssDVg, p['a'])
            #      ssgdec = ssgdec.reshape(nc, nssd*nss_di)
            #      ss_gort = self.RT(p['tr'], ssgdec).reshape(nc,nssd,nss_di)
            #
            #      sacc = np.where(ss_gort<sscancel[:,na], 0, 1)
            #      gort=gort[np.where(gort<tb, 1, 0)]
            #      ert=ss_gort[abs(1-sacc)]
            #      return [gort, ert, sscancel]





      #
      #
      # def __indx_optimize__(self, save=True, savepth='./'):
      #
      #       ri=0; nc=self.ncond; nquant=len(self.fitparams['prob'])
      #       pcols=self.fitinfo.columns
      #       fit_on = self.fit_on
      #       for i, y in enumerate(self.dat):
      #             self.avg_y = y
      #             self.fit_on = '_'.join([fit_on, str(self.indx_list[i])])
      #             self.flat_y = self.get_flaty(self.avg_y)
      #             # optimize params iterating over subjects/bootstraps
      #             yhat, finfo, popt = self.__opt_routine__()
      #
      #             self.fitinfo.iloc[i]=pd.Series({pc: finfo[pc] for pc in pcols})
      #             if self.data_style=='re':
      #                   self.fits.iloc[ri:ri+nc, :] = yhat.reshape(nc, len(self.fits.columns-1))
      #                   ri+=nc
      #             elif self.data_style=='pro':
      #                   self.fits.iloc[i] = yhat
      #             if save:
      #                   self.fits.to_csv(savepth+"fits.csv")
      #                   self.fitinfo.to_csv(savepth+"fitinfo.csv")
      #       self.popt = self.__extract_popt_fitinfo__(self, self.fitinfo.mean())

# def __indx_optimize__(self, save=True, savepth='./'):
#
#       """
#       INDIVIDUAL SUBJECT AND BOOTSTRAP OPTIMIZATION HANDLER
#       GOES IN FIT.py AS METHOD OF Optimizer CLASS
#
#       """
#
#       pass
      # ri=0; nc=self.ncond; nquant=len(self.fitparams['prob'])
      # pcols=self.fitinfo.columns

      # for i, y in enumerate(self.dat):
            # self.y = y
            # self.fit_id = getid(i)
            # self.flat_y = self.get_flaty(y)
            ##optimize params iterating over subjects/bootstraps
            # yhat, finfo, popt = self.__opt_routine__()

            # self.fitinfo.iloc[i]=pd.Series({pc: finfo[pc] for pc in pcols})
            # if self.data_style=='re':
                  # self.fits.iloc[ri:ri+nc, :] = yhat.reshape(nc, len(self.fits.columns))
                  # ri+=nc
            # elif self.data_style=='pro':
                  # self.fits.iloc[i] = yhat
            # if save:
                  # self.fits.to_csv(savepth+"fits.csv")
                  # self.fitinfo.to_csv(savepth+"fitinfo.csv")
      # self.popt = self.__extract_popt_fitinfo__(self, self.fitinfo.mean())

      #
      # def basinhopping_multivar(self, p, nsuccess=20, stepsize=.07, interval=10):
      #       """ uses L-BFGS-B in combination with basinhopping to perform bounded global
      #        minimization of multivariate model
      #       """
      #       fp = self.fitparams
      #       basin_keys = self.pc_map.keys()
      #       ncond = len(self.pc_map.values()[0])
      #       p = self.__nudge_params__(p)
      #
      #       self.simulator.__prep_global__(basin_params=p, basin_keys=basin_keys, is_flat=False)
      #       xmin, xmax = theta.format_basinhopping_bounds(basin_keys, kind=self.kind, ncond=ncond)
      #       x = np.hstack(np.hstack([p[pk] for pk in basin_keys])).tolist()
      #       bounds = map((lambda x: tuple([x[0], x[1]])), zip(xmin, xmax))
      #       mkwargs = {"method": "L-BFGS-B", "bounds":bounds, 'tol':1.e-3}
      #       # run basinhopping on simulator.basinhopping_minimizer func
      #       out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=stepsize, niter_success=nsuccess, minimizer_kwargs=mkwargs, interval=interval, disp=True)
      #       xopt = out.x
      #       funcmin = out.fun
      #       xarr = array([xopt]).reshape(len(basin_keys), self.ncond)
      #       for i, k in enumerate(basin_keys):
      #             p[k]=xarr[i]
      #       return p, funcmin
      #

      #
      # def basinhopping_univar(self, p):
      #       """ uses basinhopping to pre-optimize init cond parameters
      #       to individual conditions to prevent terminating in local minima
      #       """
      #       fp = self.fitparams
      #       nc = fp['ncond']; cols=['pkey', 'popt', 'fun', 'nfev']
      #       self.simulator.__prep_global__(method='basinhopping', basin_params=p, basin_keys=p.keys(), is_flat=True)
      #       mkwargs = {"method":"Nelder-Mead", 'jac':True}
      #       xbasin = []
      #       vals = p[pkey]
      #       for i, x in enumerate(vals):
      #             p[pkey] = x
      #             self.simulator.basin_params = p
      #             self.simulator.y = self.bdata[i]
      #             self.simulator.wts = self.bwts[i]
      #
      #             out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=.05, minimizer_kwargs=mkwargs, niter_success=20)
      #             xbasin.append(out.x[0])
      #       if self.xbasin!=[]:
      #             self.xbasin.extend(xbasin)
      #       else:
      #             self.xbasin = xbasin
      #       return xbasin
      #

      #
      # def basinhopping_multivar(self, p, nsuccess=20, stepsize=.07, interval=10, is_flat=False, disp=False):
      #       """ uses L-BFGS-B in combination with basinhopping to perform bounded global
      #        minimization of multivariate model
      #       """
      #       fp = self.fitparams
      #       if is_flat:
      #             basin_keys=p.keys()
      #             bp=dict(deepcopy(p))
      #             basin_params = theta.all_params_to_scalar(bp)
      #             ncond=1
      #       else:
      #             basin_keys = self.pc_map.keys()
      #             ncond = len(self.pc_map.values()[0])
      #             basin_params = self.__nudge_params__(p)
      #
      #       self.simulator.__prep_global__(method='basinhopping', basin_params=basin_params, basin_keys=basin_keys, is_flat=is_flat)
      #       xmin, xmax = theta.format_basinhopping_bounds(basin_keys, kind=self.kind, ncond=ncond)
      #       x = np.hstack(np.hstack([p[pk] for pk in basin_keys])).tolist()
      #
      #       bounds = map((lambda x: tuple([x[0], x[1]])), zip(xmin, xmax))
      #       mkwargs = {"method": "L-BFGS-B", "bounds":bounds, 'tol':1.e-3}
      #
      #       # run basinhopping on simulator.basinhopping_minimizer func
      #       out = basinhopping(self.simulator.basinhopping_minimizer, x, stepsize=stepsize, niter_success=nsuccess, minimizer_kwargs=mkwargs, interval=interval, disp=disp)
      #       xopt = out.x
      #       funcmin = out.fun
      #       if ncond>1:
      #             xopt = array([xopt]).reshape(len(basin_keys), ncond)
      #       for i, k in enumerate(basin_keys):
      #             p[k]=xopt[i]
      #       return p, funcmin
      #
      #
      #
      #
      # def __hop_around__(self, p, niter=10, nsuccess=20):
      #       """ initialize model with niter randomly generated parameter sets
      #       and perform global minimization using basinhopping algorithm
      #       ::Arguments::
      #             niter (int):
      #                   number of randomly generated parameter sets
      #             nsuccess (int):
      #                   tell basinhopping algorithm to exit after this many
      #                   iterations at which a common minimum is found
      #       ::Returns::
      #             parameter set with the best fit
      #       """
      #       inits = deepcopy(self.inits.keys())
      #       bnd = theta.get_bounds(kind=self.kind, tb=self.fitparams['tb'])
      #       random_inits = {pkey: theta.init_distributions(pkey, bnd[pkey], nrvs=niter, kind=self.kind) for pkey in inits}
      #       xpopt, xfmin = [], []
      #       for i in range(niter):
      #             if i==0:
      #                   p=dict(deepcopy(p))
      #             else:
      #                   p={pkey: random_inits[pkey][i] for pkey in inits}
      #             popt, fmin = self.basinhopping_multivar(p=p, is_flat=True, nsuccess=nsuccess)
      #             xpopt.append(popt)
      #             xfmin.append(fmin)
      #
      #       ix_min = np.argmin(xfmin)
      #       new_inits = xpopt[ix_min]
      #       fmin = xfmin[ix_min]
      #       if ix_min==0:
      #             self.basin_decision = "using default inits:\nfmin=%.9f"%xfmin[0]
      #       else:
      #             self.basin_decision = "found global miniumum \nnew: fmin=%.9f\norig=%9f)"%(fmin, xfmin[0])
      #             self.global_inits=dict(deepcopy(new_inits))
      #       return new_inits
      #


      # def global_min(self, inits, method='brute', basin_key=None):
      #       """ Performs global optimization via basinhopping, brute, or differential evolution
      #       algorithms.
      #
      #       basinhopping method is only used to pre-tune conditional parameters after
      #       flat optimization before entering final simplex routine (optimize_theta).
      #
      #       brute and differential evolution methods may be applied to the full parameter set
      #       (using original inits dictionary and pc_map)
      #       """
      #
      #       self.simulator.__prep_global__(method=method, basin_key=basin_key)
      #       if method=='basinhopping':
      #             keybasin = self.perform_basinhopping(p=inits, pkey=basin_key)
      #             return keybasin
      #
      #       pfit = list(set(inits.keys()).intersection(self.pnames))
      #       pbounds, params = self.slice_bounds_global(inits, pfit)
      #       self.simulator.y=self.y.flatten()
      #       self.simulator.wts = self.avg_wts
      #       if method=='brute':
      #             self.simulator.wts = self.avg_wts
      #             self.simulator.brute_params = pfit
      #             self.globalmin = brute(self.simulator.brute_minimizer, pbounds, args=params)
      #       elif method=='differential_evolution':
      #             self.simulator.diffev_params = pfit
      #             self.globalmin = differential_evolution(self.simulator.diffevolution_minimizer, pbounds, args=params)
      #
      #       return self.globalmin


#
#
# def perform_basinhoppingXXXXXX(self, p, is_flat=False, nsuccess=None, stepsize=.5):
#       """ STAGE 0/2 FITTING - GLOBAL MIN: STAGE 0 fits to find global minimum of
#       flat costfx and again at STAGE 2 in order to pre-tune conditional parameters after
#       flat optimization before entering final simplex routine (optimize_theta).
#       """
#       fp = self.fitparams
#       if is_flat:
#             basin_keys=p.keys()
#             ncond=1
#       else:
#             basin_keys=self.pc_map.keys()
#             ncond = len(self.pc_map.values()[0])
#
#       xmin, xmax = theta.format_basinhopping_bounds(basin_keys, kind=self.kind, ncond=ncond)
#       x = np.hstack(np.hstack([p[pk] for pk in basin_keys])).tolist()
#       bounds = map((lambda x: tuple([x[0], x[1]])), zip(xmin, xmax))
#       # set up custom step handler to keep basin hopping in bounds (class at the very bottom)
#       #take_step = RandomDisplacementBounds(xmin, xmax, stepsize=stepsize)
#       # set up local minimizer for polishing
#       #mkwargs = {"method": "L-BFGS-B", "bounds":bounds, 'options':{'tol':1.e-2}}
#       mkwargs = {"method": "Nelder-Mead",  'jac':True}
#       # weight error of simulations for global optimzaiton
#       self.simulator.__prep_global__(method='basinhopping', basin_params=p, basin_keys=basin_keys, is_flat=is_flat)
#       #minimizer_kwargs=mkwargs,
#       # run basinhopping on simulator.basinhopping_minimizer func
#       out = basinhopping(self.simulator.basinhopping_minimizer, x, niter_success=20, stepsize=stepsize, minimizer_kwargs=mkwargs, disp=True)#, interval=5,
#       pvals = out.x
#       funcmin = out.fun
#       for i, k in enumerate(basin_keys):
#             if ncond>1:
#                   p[k] = array(pvals[i:ncond])
#                   i+=ncond
#             else:
#                   p[k]=pvals[i]
#       if self.xbasin!=[]:
#             self.xbasin.extend(funcmin)
#       else:
#             self.xbasin = funcmin
#       return p, funcmin


def old_proactive_weighting_code(x):

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

def __simulate_functions__(x):

      pass

      #""" initiates the simulation function used in
      #optimization routine
      #"""
      #
      #if 'radd' in self.kind:
      #      self.sim_fx = self.simulate_radd
      #
      #elif 'pro' in self.kind:
      #      self.sim_fx = self.simulate_pro
      #      self.ntot = int(self.ntot/self.ncond)
      #
      #elif 'irace' in self.kind:
      #      self.sim_fx = self.simulate_irace
      #


def __init_analyze_functions__(x):

      pass

      #""" initiates the analysis function used in
      #optimization routine to produce the yhat vector
      #"""
      #
      #prob=self.prob; nss =self.nss;
      #ssd=self.ssd; tb = self.tb
      #
      ##if self.fitparams['split']=='HML':
      ##      self.ziprt=lambda rt: zip([rt[-1],hs(rt[3:-1]),hs(rt[:3])],[tb]*3)
      ##elif self.fitparams['split']=='HL':
      ##      self.ziprt=lambda rt: zip([hs(rt[3:]),hs(rt[:3])],[tb]*2)
      #
      #self.resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=2)*.001
      #self.resp_lo = lambda trace: np.argmax((trace.T<=0).T, axis=3)*.001
      #self.RT = lambda ontime, rbool: ontime[:, None]+(rbool*np.where(rbool==0, np.nan, 1))
      #self.fRTQ = lambda zpd: map((lambda x:mq(x[0][x[0]<x[1]], prob)), zpd)


def __update_go_process__(x, p):
      """ calculate go process params (Pg, Tg)
      and hyperbolic or exponential dynamic bias
      across time (exp/hyp specified in dynamic attr.)

      ::Arguments::
            p (dict):
                  parameter dictionary
      ::Returns::
            Pg (array):
                  Probability DVg(t)=+dx
            Tg (array):
                  nTimepoints tr --> tb
      """
      pass

      #t = np.cumsum([self.dt]*Tg.max())
      #if 'x' in self.kind and self.dynamic=='exp':
      #      # dynamic bias is exponential
      #      self.xtb = array([np.exp(xtb*t) for xtb in p['xb']])
      #      self.lowerb = np.ones(len(t))*p['z']
      #elif 'x' in kind and self.dynamic=='hyp':
      #      # dynamic bias is hyperbolic
      #      t = np.cumsum(np.ones(Tg.max()))[::-1]
      #      self.lowerb = p['z'] + map((lambda x: (.5*x[0])/(1+(x[1]*t))), zip(p['a'],p['xb']))[0]
      #      self.xtb = array([np.ones(len(t)) for i in range(self.ncond)])
      #else:
      #      self.xtb = array([np.ones(len(t)) for i in range(self.ncond)])
      #      self.lowerb = np.ones(len(t))*p['z']
           #return Pg, Tg






def mat2py(indir, outdir=None, droplist=None):
      pass
      #      if droplist is None:
      #            droplist = ['dt_vec', 'Speed', 'state', 'time', 'probe_trial', 'ypos', 'fill_pos', 't_vec', 'Y',		 'pos']
      #
      #      flist = filter(lambda x: 'SS' in x and '_fMRI_Proactive' in x and 'run' in x, os.listdir(indir))
      #      dflist = []
      #      noresp_group = []
      #      os.chdir(indir)
      #      for name in flist:
      #
      #            idx, run = re.split('_|-', name)[:2]
      #              date = '-'.join(re.split('_|-', name)[2:5])
      #
      #            mat = loadmat(name)  # load mat-file
      #            mdata = mat['Data']  # variable in mat file
      #              mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
      #
      #
      #            columns = [ n for n in mdtype.names]
      #            columns.insert(0, 'idx')
      #            columns.insert(1, 'run')
      #              columns.insert(2, 'date')
      #
      #            data = [[vals[0][0] for vals in trial] for trial in mat['Data'][0]]
      #            for trial in data:
      #                  trial.insert(0,int(idx[2:]))
      #                  trial.insert(1, int(run[-1]))
      #                  trial.insert(2, date)
      #            df = pd.DataFrame(data, columns=columns)
      #            df.rename(columns={'Keypress': 'go', 'Hit':'hit', 'Stop':'nogo', 'DO_STOP':'sstrial',		 'GoPoint':'pGo', 'Bonus':'bonus'}, inplace=True)
      #
      #            df['gotrial']=abs(1-df.sstrial)
      #            df['ttype']=np.where(df['gotrial']==1, 'go', 'stop')
      #            df['response']=df['go'].copy()
      #            df['rt']=df['pos']*df['Speed']
      #            df.drop(droplist, axis=1, inplace=True)
      #
      #            if 'trial_start_time' in columns:
      #                  df.drop('trial_start_time', axis=1, inplace=True)
      #
      #            if df.response.mean()<.2:
      #                  noresp_group.append(df)
      #            else:
      #                  df['run'] = df['run'].astype(int)
      #                  df['idx'] = df['idx'].astype(int)
      #                  df['response'] = df['response'].astype(int)
      #                  df['hit'] = df['hit'].astype(int)
      #                  df['nogo'] = df['nogo'].astype(int)
      #                  df['sstrial'] = df['sstrial'].astype(int)
      #                  df['gotrial'] = df['gotrial'].astype(int)
      #                  df['bonus'] = df['bonus'].astype(int)
      #                  df['pGo'] = df['pGo']*100
      #                  df['pGo'] = df['pGo'].astype(int)
      #                  df['rt'] = df['rt'].astype(float)
      #
      #                  if outdir:
      #                        df.to_csv(outdir+'sx%s_proimg_data.csv' % idx, index=False)
      #                  dflist.append(df)
      #
      #      master=pd.concat(dflist)
      #      if outdir:
      #            master.to_csv(outdir+"ProImg_All.csv", index=False)
      #      return master


def RADD(model, ncond=2, prob=([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=10000, tb=0.650, dt=.001, si=.01, return_traces=False):
      pass

      #      """
      #
      #      Main code for simulating Reactive RADD model
      #
      #      Simulates all Conditions, SSD, trials, timepoints simultaneously.
      #      Vectorized operations are set up so that any of the parameters can be
      #      a single float or a vector of floats (i.e., when simulating/fitting multiple
      #      conditions differentiated by the value of one or more model parameters)
      #
      #      args:
      #            p (dict):                           model parameters [a, tr, v, ssv, z]
      #            ssd  (array):                       full set of stop signal delays
      #            nss  (int):                         number of stop trials
      #            ntot (int):                         number of total trials
      #            tb (float):                         time boundary
      #            ncond (int):                        number of conditions to simulate
      #
      #      returns:
      #
      #            DVg (Go Process):             3d array for all conditions, trials, timepoints
      #                                          (i.e. DVg = [nCOND [NTrials [NTime]]] )
      #                                          All conditions are simulated simultaneously (i.e., BSL &    PNL)
      #
      #            DVs (Stop Process):           4d array for all conditions, SSD, SS trials, timepoints.
      #                                          i.e. ( DVs = [COND [SSD [nSSTrials [NTime]]]] )
      #                                          All ss decision traces are initiated from       DVg[Cond](t=SSD# | SSD<tr)
      #      """
      #      model.make_simulator()
      #      sim = model.simulator
      #
      #      nss = sim.nss; ntot=sim.ntot;
      #
      #      dx=np.sqrt(si*dt)
      #
      #      p = sim.vectorize_params(model.inits)
      #      #Pg, Tg = sim.__update_go_process__(p)
      #      #Ps, Ts = sim.__update_stop_process__(p)
      #
      #      a, tr, v, ssv, z = p['a'], p['tr'], p['v'], p['ssv'], p['z']
      #
      #      #Pg = 0.5*(1 + v*dx/si)
      #      #Ps = 0.5*(1 + ssv*dx/si)
      #      #Tg = np.ceil((tb-tr)/dt).astype(int)
      #      #Ts = np.ceil((tb-ssd)/dt).astype(int)
      #      Pg, Ps, Tg, Ts = sim.__update_params__(p)
      #
      #      # a/tr/v Bias: ALL CONDITIONS, ALL SSD
      #      DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
      #      init_ss = array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in  enumerate(DVg)])
      #      DVs = init_ss[:,:,:,na] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)
      #
      #      grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:,   :].T>=a).T,axis=2)*dt, np.nan).T)).T
      #      ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss,  :].T>=a).T,axis=2)*dt, np.nan).T)).T
      #      ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:,na]+np.argmax(DVs<=0, axis=3)*dt,np.nan)
      #      ert = ertx[:,na]*np.ones_like(ssrt)
      #
      #      #collapse across SSD and get average ssrt vector for each condition
      #      # compute RT quantiles for correct and error resp.
      #      gq = np.vstack([mq(rtc[rtc<tb], prob=prob) for rtc in grt])
      #      eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob) for i in range(sim.ncond)]
      #      # Get response and stop accuracy information
      #      gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
      #      sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
      #      #return gq, eq, gac, sacc
      #      if return_traces:
      #            return DVg, DVs
      #
      #      return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])



def proRADD(p, ncond=6, pGo=np.arange(.2,1.2,.2), prob=([.1, .3, .5, .7, .9]), ssd=.45, ntot=2000, tb=0.545, dt=.001, si=.01, return_traces=False, style='DDM'):
            pass
      #"""
      #
      #      main code for simulating Proactive RADD model
      #
      #      args:
      #            p (dict):                           model parameters [a, tr, v, ssv, z]
      #            ssd  (array):                       full set of stop signal delays
      #            ntot (int):                         number of total trials
      #            tb (float):                         time boundary
      #            ncond (int):                        number of conditions to simulate
      #
      #      returns:
      #
      #            DVg (Go Process):             3d array for all conditions, trials, timepoints
      #                                          (i.e. DVg = [nCOND [NTrials [NTime]]] )
      #                                          All conditions are simulated simultaneously (i.e., BSL &    PNL)
      #
      #            DVs (Stop Process):           4d array for all conditions, SSD, SS trials, timepoints.
      #                                          i.e. ( DVs = [COND [SSD [nSSTrials [NTime]]]] )
      #                                          All ss decision traces are initiated from       DVg[Cond](t=SSD# | SSD<tr)
      #      """
      #
      #      dx=np.sqrt(si*dt)
      #
      #      a, tr, v, ssv, z = p['a'], p['tr'], p['v'], p['ssv'], p['z']
      #
      #      if np.ndim(tr)==0:
      #            tr=np.ones(ncond)*tr
      #      if np.ndim(a)==0:
      #            a=np.ones(ncond)*a
      #      if np.ndim(v)==0:
      #            v=np.ones(ncond)*v
      #      if np.ndim(ssd)==0:
      #            ssd = np.ones(ncond)*ssd
      #
      #      nssd = len(ssd); nss = int(.5*ntot)
      #
      #      Pg = 0.5*(1 + v*dx/si)
      #      Ps = 0.5*(1 + ssv*dx/si)
      #      Tg = np.ceil((tb-tr)/dt).astype(int)
      #      Ts = np.ceil((tb-ssd)/dt).astype(int)
      #
      #      # a/tr/v Bias: ALL CONDITIONS
      #      DVg = z + np.cumsum(np.where((rs((ncond, int(ntot/ncond), Tg.max())).T < Pg), dx, -dx).T,       axis=2)
      #      grt = (tr+(np.where((DVg.max(axis=2).T>=a).T, np.argmax((DVg.T>=a).T,axis=2)*dt,np.nan).T)).T
      #
      #      hi = np.nanmean(grt[:ncond/2], axis=0)
      #      lo = np.nanmean(grt[ncond/2:], axis=0)
      #
      #      hilo = [hi[~np.isnan(hi)], lo[~np.isnan(lo)]]
      #
      #      # compute RT quantiles for correct and error resp.
      #      gq = hs([mq(rtc[rtc<tb], prob=prob)*10 for rtc in hilo])
      #      # Get response and stop accuracy information
      #      gac = 1-np.mean(np.where(grt<tb, 1, 0), axis=1)
      #      #return gq, eq, gac, sacc
      #      if return_traces:
      #            return DVg, DVs
      #      return hs([gac, gq])


def simulate_radd(self, p, analyze=True):
      pass
      #      """ Simulate the dependent process model (RADD)
      #
      #      ::Arguments::
      #            p (dict):
      #                  parameter dictionary. values can be single floats
      #                  or vectors where each element is the value of that
      #                  parameter for a given condition
      #            analyze (bool <True>):
      #                  if True (default) return rt and accuracy information
      #                  (yhat in cost fx). If False, return Go and Stop proc.
      #      ::Returns::
      #            yhat of cost vector (ndarray)
      #            or Go & Stop processes in list (list of ndarrays)
      #      """
      #
      #      p = self.vectorize_params(p)
      #      Pg, Tg = self.__update_go_process__(p)
      #      Ps, Ts = self.__update_stop_process__(p)
      #
      #      DVg = self.base+(self.xtb[:,na]*np.cumsum(np.where((rs((self.ncond, self.ntot,      Tg.max())).T<Pg), self.dx,-self.dx).T, axis=2))
      #      # INITIALIZE DVs FROM DVg(t=SSD)
      #      init_ss = array([[DVc[i, :self.nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc   in enumerate(DVg.reshape(self.ncond, self.nssd, self.nss, Tg.max()))])
      #      DVs = init_ss[:,:,:,na]+np.cumsum(np.where(rs((self.ncond, self.nssd, self.nss,     Ts.max()))<Ps,# self.dx, -self.dx), axis=3)
      #      print DVs.shape
      #      if analyze:
      #            return self.analyze_reactive(DVg, DVs, p)
      #
def analyze_reactive(self, DVg, DVs, p):
      pass
      #      """ get rt and accuracy of go and stop process for simulated
      #      conditions generated from simulate_radd
      #      """
      #      nss = self.nss; prob = self.prob
      #      ssd = self.ssd; tb = self.tb
      #      ncond = self.ncond; nssd=self.nssd
      #
      #      gdec = self.resp_up(DVg, p['a'])
      #      gort = self.RT(p['tr'], gdec)
      #      if 'radd' in self.kind:
      #            sdec = self.resp_lo(DVs)
      #            ssrt = self.RT(ssd, sdec)
      #            ert = gort[:, :nss][:,na] * np.ones_like(ssrt)
      #            #sdec = self.resp_lo(DVs)
      #            #ssrt = self.RT(ssd, sdec)
      #            #ert = gort.reshape(ncond, nssd, nss)
      #      elif 'irace' in self.kind:
      #            sdec = self.resp_iss(DVs, p['a'])
      #            ssrt = self.RT(ssd, sdec.reshape(ncond, nssd, nss))
      #            ert = gort.reshape(ncond, nssd, nss)
      #
      #      eq = self.RTQ(zip(ert, ssrt))
      #      gq = self.RTQ(zip(gort,[tb]*ncond))
      #      gac = np.nanmean(np.where(gort<tb, 1, 0), axis=1)
      #      sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
      #
      #      return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(self.ncond)])


#
# def RADD(model, ncond=2, prob=([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=10000, tb=0.650, dt=.0005, si=.01, return_traces=False):
#       """
#
#       Main code for simulating Reactive RADD model
#
#       Simulates all Conditions, SSD, trials, timepoints simultaneously.
#       Vectorized operations are set up so that any of the parameters can be
#       a single float or a vector of floats (i.e., when simulating/fitting multiple
#       conditions differentiated by the value of one or more model parameters)
#
#       args:
#             p (dict):                           model parameters [a, tr, v, ssv, z]
#             ssd  (array):                       full set of stop signal delays
#             nss  (int):                         number of stop trials
#             ntot (int):                         number of total trials
#             tb (float):                         time boundary
#             ncond (int):                        number of conditions to simulate
#
#       returns:
#
#             DVg (Go Process):             3d array for all conditions, trials, timepoints
#                                           (i.e. DVg = [nCOND [NTrials [NTime]]] )
#                                           All conditions are simulated simultaneously (i.e., BSL & PNL)
#
#             DVs (Stop Process):           4d array for all conditions, SSD, SS trials, timepoints.
#                                           i.e. ( DVs = [COND [SSD [nSSTrials [NTime]]]] )
#                                           All ss decision traces are initiated from DVg[Cond](t=SSD | SSD<tr)
#       """
#       model.make_simulator()
#       sim = model.simulator
#
#       nss = sim.nss; ntot=sim.ntot;
#
#       dx=np.sqrt(si*dt)
#
#       p = sim.vectorize_params(model.inits)
#       #Pg, Tg = sim.__update_go_process__(p)
#       #Ps, Ts = sim.__update_stop_process__(p)
#
#       a, tr, v, ssv, z = p['a'], p['tr'], p['v'], p['ssv'], p['z']
#
#       #Pg = 0.5*(1 + v*dx/si)
#       #Ps = 0.5*(1 + ssv*dx/si)
#       #Tg = np.ceil((tb-tr)/dt).astype(int)
#       #Ts = np.ceil((tb-ssd)/dt).astype(int)
#       Pg, Ps, Tg, Ts = sim.__update_params__(p)
#
#       # a/tr/v Bias: ALL CONDITIONS, ALL SSD
#       DVg = z + np.cumsum(np.where((rs((ncond, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
#       init_ss = array([[DVc[:nss, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])
#       DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)
#
#       grt = (tr+(np.where((DVg[:, nss:, :].max(axis=2).T>=a).T, np.argmax((DVg[:, nss:, :].T>=a).T,axis=2)*dt, np.nan).T)).T
#       ertx = (tr+(np.where((DVg[:, :nss, :].max(axis=2).T>=a).T, np.argmax((DVg[:, :nss, :].T>=a).T,axis=2)*dt, np.nan).T)).T
#       ssrt = np.where(np.any(DVs<=0, axis=3), ssd[:, None]+np.argmax(DVs<=0, axis=3)*dt,np.nan)
#       ert = ertx[:, None]*np.ones_like(ssrt)
#
#       #collapse across SSD and get average ssrt vector for each condition
#       # compute RT quantiles for correct and error resp.
#       gq = np.vstack([mq(rtc[rtc<tb], prob=prob) for rtc in grt])
#       eq = [mq(ert[i][ert[i]<ssrt[i]], prob=prob) for i in range(sim.ncond)]
#       # Get response and stop accuracy information
#       gac = np.nanmean(np.where(grt<tb, 1, 0), axis=1)
#       sacc = np.where(ert<ssrt, 0, 1).mean(axis=2)
#       #return gq, eq, gac, sacc
#       if return_traces:
#             return DVg, DVs
#
#       return hs([hs([i[ii] for i in [gac, sacc, gq, eq]]) for ii in range(ncond)])
#
#
#
# def proRADD(p, ncond=6, pGo=np.arange(.2,1.2,.2), prob=([.1, .3, .5, .7, .9]), ssd=.45, ntot=2000, tb=0.545, dt=.0005, si=.01, return_traces=False, style='DDM'):
#       """
#
#       main code for simulating Proactive RADD model
#
#       args:
#             p (dict):                           model parameters [a, tr, v, ssv, z]
#             ssd  (array):                       full set of stop signal delays
#             ntot (int):                         number of total trials
#             tb (float):                         time boundary
#             ncond (int):                        number of conditions to simulate
#
#       returns:
#
#             DVg (Go Process):             3d array for all conditions, trials, timepoints
#                                           (i.e. DVg = [nCOND [NTrials [NTime]]] )
#                                           All conditions are simulated simultaneously (i.e., BSL & PNL)
#
#             DVs (Stop Process):           4d array for all conditions, SSD, SS trials, timepoints.
#                                           i.e. ( DVs = [COND [SSD [nSSTrials [NTime]]]] )
#                                           All ss decision traces are initiated from DVg[Cond](t=SSD | SSD<tr)
#       """
#
#       dx=np.sqrt(si*dt)
#
#       a, tr, v, ssv, z = p['a'], p['tr'], p['v'], p['ssv'], p['z']
#
#       if np.ndim(tr)==0:
#             tr=np.ones(ncond)*tr
#       if np.ndim(a)==0:
#             a=np.ones(ncond)*a
#       if np.ndim(v)==0:
#             v=np.ones(ncond)*v
#       if np.ndim(ssd)==0:
#             ssd = np.ones(ncond)*ssd
#
#       nssd = len(ssd); nss = int(.5*ntot)
#
#       Pg = 0.5*(1 + v*dx/si)
#       Ps = 0.5*(1 + ssv*dx/si)
#       Tg = np.ceil((tb-tr)/dt).astype(int)
#       Ts = np.ceil((tb-ssd)/dt).astype(int)
#
#       # a/tr/v Bias: ALL CONDITIONS
#       DVg = z + np.cumsum(np.where((rs((ncond, int(ntot/ncond), Tg.max())).T < Pg), dx, -dx).T, axis=2)
#       grt = (tr+(np.where((DVg.max(axis=2).T>=a).T, np.argmax((DVg.T>=a).T,axis=2)*dt,np.nan).T)).T
#
#       hi = np.nanmean(grt[:ncond/2], axis=0)
#       lo = np.nanmean(grt[ncond/2:], axis=0)
#
#       hilo = [hi[~np.isnan(hi)], lo[~np.isnan(lo)]]
#
#       # compute RT quantiles for correct and error resp.
#       gq = hs([mq(rtc[rtc<tb], prob=prob)*10 for rtc in hilo])
#       # Get response and stop accuracy information
#       gac = 1-np.mean(np.where(grt<tb, 1, 0), axis=1)
#       #return gq, eq, gac, sacc
#       if return_traces:
#             return DVg, DVs
#       return hs([gac, gq])
