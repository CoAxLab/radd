#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.toolbox.theta import *
from radd.toolbox.analyze import *
from scipy.stats.mstats import mquantiles as mq
from radd.toolbox.messages import saygo


class RADDCore(object):

      """ Parent class for constructing shared attributes and methods
      of Model & Optimizer objects. Not meant to be used directly.

      Contains methods for building dataframes, generating observed data vectors
      that are entered into cost function during fitting, calculating summary measures
      and weight matrix for weighting residuals during optimization.

      TODO: COMPLETE DOCSTRINGS
      """

      def __init__(self, data=None, kind='radd', inits=None, fit_on='average', depends_on=None, niter=50, fit_whole_model=True, tb=None, fit_noise=False, pro_ss=False, dynamic='hyp', split=50, include_zero_rts=False, *args, **kws):

            self.data = data
            self.kind = kind
            self.fit_on = fit_on
            self.dynamic = dynamic
            self.fit_whole_model=fit_whole_model

            # BASIC MODEL STRUCTURE (kind)
            if 'pro' in self.kind:
                  self.data_style='pro'
                  if depends_on is None:
                        depends_on = {'v':'pGo'}
                  self.ssd=np.array([.450])
                  if tb==None:
                        tb=.55
                  self.split=split
                  if isinstance(self.split, int):
                        self.nrt_cond=2
                  elif isinstance(self.split, list):
                        self.nrt_cond=len(self.split)

                  self.pGo=sorted(self.data.pGo.unique())
                  self.include_zero_rts=include_zero_rts
            else:
                  self.data_style='re'
                  if depends_on is None:
                        depends_on = {'v':'Cond'}
                  ssd = data[data.ttype=="stop"].ssd.unique()
                  self.pGo = len(data[data.ttype=='go'])/len(data)
                  self.delays = sorted(ssd.astype(np.int))
                  self.ssd = array(self.delays)*.001

            # CONDITIONAL PARAMETERS
            self.depends_on = depends_on
            self.cond = self.depends_on.values()[0]
            self.labels = np.sort(data[self.cond].unique())
            self.ncond = len(self.labels)

            # index to split pro. rts during fit
            # if split!=None (is set during prep in
            # analyze.__make_proRT_conds__())
            self.rt_cix = None

            # GET TB BEFORE REMOVING OUTLIERS!!!
            if tb != None:
                  self.tb=tb
            else:
                  self.tb = data[data.response==1].rt.max()

            # PARAMETER INITIALIZATION
            if inits is None:
                  self.__get_default_inits__()
            else:
                  self.inits = inits

            self.__check_inits__(fit_noise=fit_noise, pro_ss=pro_ss)

            # DATA TREATMENT AND EXTRACTION
            self.__remove_outliers__(sd=1.5, verbose=False)

            # DEFINE ITERABLES
            if self.fit_on=='bootstrap':
                  self.indx = range(niter)
            else:
                  self.indx = list(data.idx.unique())


      def rangl_data(self, data, kind='radd', prob=np.array([.1, .3, .5, .7, .9])):

            if self.data_style=='re':
                  gac = data.query('ttype=="go"').acc.mean()
                  sacc = data.query('ttype=="stop"').groupby('ssd').mean()['acc'].values
                  grt = data.query('ttype=="go" & acc==1').rt.values
                  ert = data.query('response==1 & acc==0').rt.values
                  gq = mq(grt, prob=prob)
                  eq = mq(ert, prob=prob)
                  return np.hstack([gac, sacc, gq, eq])

            elif self.data_style=='pro':
                  godf = data[data.response==1]
                  godf['response']=np.where(godf.rt<self.tb, 1, 0)
                  data = pd.concat([godf, data[data.response==0]])
                  return 1-data.response.mean()


      def rt_quantiles(self, data, split='HL', prob=np.array([.1, .3, .5, .7, .9])):

            if not hasattr(self, "prort_conds_prepared"):
                  self.__make_proRT_conds__()

            if self.include_zero_rts:
                  godfx = data[(data.response==1)]
            else:
                  godfx = data[(data.response==1) & (data.pGo>0.)]
            godfx.loc[:, 'response'] = np.where(godfx.rt<self.tb, 1, 0)
            godf = godfx.query('response==1')

            if split == None:
                  rts = godf[godf.rt<=self.tb].rt.values
                  return mq(rts, prob=prob)

            rtq = []
            for i in range(1, self.nrt_cond+1):
                  if i not in godf[split].unique():
                        rtq.append(array([np.nan]*len(prob)))
                  else:
                        rts = godf[(godf[split]==i)&(godf.rt<=self.tb)].rt.values
                        rtq.append(mq(rts, prob=prob))

            return np.hstack(rtq)


      def resample_data(self, data, n=120, kind='radd'):

            df=data.copy(); bootlist=list()
            if n==None: n=len(df)

            if self.data_style=='re':
                  for ssd, ssdf in df.groupby('ssd'):
                        boots = ssdf.reset_index(drop=True)
                        orig_ix = np.asarray(boots.index[:])
                        resampled_ix = rwr(orig_ix, get_index=True, n=n)
                        bootdf = ssdf.irow(resampled_ix)
                        bootlist.append(bootdf)
                        #concatenate and return all resampled conditions
                        return rangl_re(pd.concat(bootlist))

            elif self.data_style=='pro':
                  boots = df.reset_index(drop=True)
                  orig_ix = np.asarray(boots.index[:])
                  resampled_ix = rwr(orig_ix, get_index=True, n=n)
                  bootdf = df.irow(resampled_ix)
                  bootdf_list.append(bootdf)
                  return rangl_pro(pd.concat(bootdf_list), rt_cutoff=rt_cutoff)


      def set_fitparams(self, ntrials=10000, tol=1.e-20, maxfev=5000, niter=500, disp=True, prob=np.array([.1, .3, .5, .7, .9]), get_params=False, **kwgs):

            if not hasattr(self, 'fitparams'):
                  self.fitparams={}

            self.fitparams = {'ntrials':ntrials, 'maxfev':maxfev, 'disp':disp, 'tol':tol, 'niter':niter, 'prob':prob, 'tb':self.tb, 'ssd':self.ssd, 'wts':self.wts, 'ncond':self.ncond, 'pGo':self.pGo, 'flat_wts':self.fwts, 'depends_on': self.depends_on, 'dynamic': self.dynamic, 'fit_whole_model': self.fit_whole_model, 'rt_cix': self.rt_cix}

            if get_params:
                  return self.fitparams


      def __extract_popt_fitinfo__(self, finfo=None):
            """ takes optimized dict or DF of vectorized parameters and
            returns dict with only depends_on.keys() containing vectorized vals.
            Is accessed by fit.Optimizer objects after optimization routine.

            ::Arguments::
                  finfo (dict/DF):
                        finfo is dict if self.fit_on is 'average'
                        and DF if self.fit_on is 'subjects' or 'bootstrap'
                        contains optimized parameters
            ::Returns::
                  popt (dict):
                        dict with only depends_on.keys() containing
                        vectorized vals
            """


            if finfo is None:
                  try:
                        finfo=self.fitinfo.mean()
                  except Exception:
                        finfo=self.fitinfo

            finfo=dict(deepcopy(finfo))
            popt=dict(deepcopy(self.inits))
            pc_map = self.pc_map;

            for pkey in popt.keys():
                  if pkey in self.depends_on.keys():
                        popt[pkey] = np.array([finfo[pc] for pc in pc_map[pkey]])
                        continue
                  popt[pkey]=finfo[pkey]

            return popt



      def __make_dataframes__(self, qp_cols):
            """ Generates the following dataframes and arrays:

            ::Arguments::
                  qp_cols:
                        header for observed/fits dataframes
            ::Returns::
                  None (All dataframes and vectors are stored in dict and assigned
                  as <dframes> attr)

            observed (DF):
                  Contains Prob and RT quant. for each subject
                  used to calc. cost fx weights
            fits (DF):
                  empty DF shaped like observed DF, used to store simulated
                  predictions of the optimized model
            fitinfo (DF):
                  stores all opt. parameter values and model fit statistics
            dat (ndarray):
                  contains all subject/boot. y vectors entered into costfx
            avg_y (ndarray):
                  average y vector for each condition entered into costfx
            flat_y (1d array):
                  average y vector used to initialize parameters prior to fitting
                  conditional model. calculated collapsing across conditions
            """

            cond = self.cond; ncond = self.ncond
            data = self.data; indx = self.indx
            labels = self.labels

            ic_grp = data.groupby(['idx', cond])
            c_grp = data.groupby([cond])
            i_grp = data.groupby(['idx'])

            if self.fit_on=='bootstrap':
                  self.dat = np.vstack([i_grp.apply(self.resample_data, kind=self.kind).values for i in indx]).unstack()

            if self.data_style=='re':
                  datdf = ic_grp.apply(self.rangl_data, kind=self.kind).unstack().unstack()
                  indxx = pd.Series(indx*ncond, name='idx')
                  obs = pd.DataFrame(np.vstack(datdf.values), columns=qp_cols, index=indxx)
                  obs[cond]=np.sort(labels*len(indx))
                  self.observed = obs.sort_index().reset_index()
                  self.avg_y = self.observed.groupby(cond).mean().loc[:,qp_cols[0] : qp_cols[-1]].values
                  self.flat_y = self.observed.loc[:, qp_cols[0] : qp_cols[-1]].mean().values
                  dat = self.observed.loc[:,qp_cols[0]:qp_cols[-1]].values.reshape(len(indx),ncond,16)
                  fits = pd.DataFrame(np.zeros((len(indxx),len(qp_cols))), columns=qp_cols, index=indxx)

            elif self.data_style=='pro':
                  datdf = ic_grp.apply(self.rangl_data, kind=self.kind).unstack()
                  rtdat = pd.DataFrame(np.vstack(i_grp.apply(self.rt_quantiles).values), index=indx)
                  rtdat[rtdat<.1] = np.nan
                  rts_flat = pd.DataFrame(np.vstack(i_grp.apply(self.rt_quantiles, split=None).values), index=indx)
                  self.observed = pd.concat([datdf, rtdat], axis=1)
                  self.observed.columns = qp_cols
                  self.avg_y = self.observed.mean().values
                  self.flat_y=np.append(datdf.mean().mean(), rts_flat.mean())
                  dat = self.observed.values.reshape((len(indx), len(qp_cols)))
                  fits = pd.DataFrame(np.zeros_like(dat), columns=qp_cols, index=indx)

            fitinfo = pd.DataFrame(columns=self.infolabels, index=indx)

            self.dframes = {'data':self.data, 'flat_y':self.flat_y, 'avg_y':self.avg_y, 'fitinfo': fitinfo, 'fits': fits, 'observed': self.observed, 'dat':dat}


      def __prep_basin_data__(self):

            fp=self.fitparams
            if 'pro' in self.kind:
                  # regroup y vectors into conditions
                  nogo = self.y.flatten()[:fp['ncond']].reshape(2, int(fp['ncond']/2))
                  wtsp = fp['wts'].flatten()[:fp['ncond']].reshape(2, int(fp['ncond']/2))

                  rts = self.y[fp['ncond']:].reshape(2,5)
                  wtsq = fp['wts'][fp['ncond']:].reshape(2,5)

                  upper =  [np.append(ng, rts[0]) for ng in nogo[0]]
                  lower =  [np.append(ng, rts[1]) for ng in nogo[1]]
                  upperwts = [np.append(wtp, wtsq[0]) for wtp in wtsp[0]]
                  lowerwts = [np.append(wtp, wtsq[1]) for wtp in wtsp[1]]

                  cond_data = np.vstack([upper, lower])
                  cond_wts = np.vstack([upperwts, lowerwts])
            else:
                  cond_data = self.y
                  cond_wts = self.wts

            return cond_data, cond_wts


      def __nudge_params__(self, p, pkey, lim=(.98, 1.02)):
            """
            nudge params so not all initialized at same val
            """
            bump = np.linspace(lim[0], lim[1], self.ncond)
            if pkey in ['a', 'tr']:
                  bump = bump[::-1]
            p[pkey] = p[pkey]*bump
            return p


      def slice_bounds_global(self, inits, pfit):

            b = self.set_bounds()
            pclists=[]
            for pkey, pcl in self.pc_map.items():
                  pfit.remove(pkey)
                  pfit.extend(pcl)
                  # add bounds for all
                  # pkey condition params
                  for pkc in pcl:
                        inits[pkc]=inits[pkey]
                        b[pkc]=b[pkey]

            pbounds = tuple([slice(b[pk][0], b[pk][1], .25*np.max(np.abs(b[pk]))) for pk in pfit])
            params = tuple([inits[pk] for pk in pfit])

            return pbounds, params


      def set_bounds(self, a=(.001, 1.000), tr=(.1, .55), v=(.0001, 4.0000), z=(.001, .900), ssv=(-4.000, -.0001), xb=(.01,10), si=(.001, .2)):

            """ set and return boundaries to limit search space
            of parameter optimization in <optimize_theta>
            """

            if 'irace' in self.kind:
                  ssv=(abs(ssv[1]), abs(ssv[0]))

            bounds = {'a': a, 'tr': tr, 'v': v, 'ssv': ssv, 'z': z, 'xb':xb, 'si':si}
            return bounds


      def get_wts(self):
            """ wtc: weights applied to correct rt quantiles in cost f(x)
                  * P(R | No SSD)j * sdj(.5Qj, ... .95Qj)
            wte: weight applied to error rt quantiles in cost f(x)
                  * P(R | SSD) * sd(.5eQ, ... .95eQ)
            """

            nc = self.ncond; cond=self.cond;

            if self.data_style=='re':
                  obs_var = self.observed.groupby(cond).sem().loc[:,'Go':]
                  qvar = obs_var.values[:,6:]
                  # round to precision of rt collection (~2ms)
                  # no rounding: ~.01 <--> ~35 /// with rounding: ~.3 <--> ~2.5
                  qvar[qvar<=.001] = .001
                  pvar = obs_var.values[:,:6]
                  go = self.data.query('ttype=="go"').response.mean()
                  st = self.data.query('ttype=="stop"').response.mean()

                  sq_ratio = (np.median(qvar, axis=1)/qvar.T).T
                  wt_go = (go*sq_ratio[:, :5].T).T
                  wt_err = (st*sq_ratio[:, -5:].T).T
                  qwts = np.hstack(np.vstack(zip(wt_go, wt_err))).reshape(nc, 10)
                  pwts = (np.median(pvar, axis=1)/pvar.T).T
                  self.wts = np.hstack([np.append(p, w) for p, w in zip(pwts, qwts)])
                  # calculate flat weights (collapsing across conditions)
                  self.fwts = self.wts.reshape(nc, 16).mean(axis=0)

            elif self.data_style=='pro':

                  upper = self.data[self.data['HL']==1].mean()['response']
                  lower = self.data[self.data['HL']==2].mean()['response']

                  pvar = self.data.groupby('pGo').std().response.values
                  psub1 = np.median(pvar[:-1])/pvar[:-1]
                  pwts = np.append(psub1, psub1.max())
                  #pwts = np.array([1.5,1,1,1,1,1.5])

                  qvar = self.observed.std().iloc[6:].values
                  # round to precision of rt collection (~2ms)
                  # no rounding: ~.01 <--> ~35 /// with rounding: ~.3 <--> ~2.5
                  qvar[qvar<=.001] = .001
                  #sq_ratio = (np.median(qvar_r, axis=1)/qvar_r.T).T

                  sq_ratio = (np.median(qvar)/qvar).reshape(2,5)
                  wt_hi = upper*sq_ratio[0, :]
                  wt_lo = lower*sq_ratio[1, :]

                  self.wts = np.hstack([pwts, wt_hi, wt_lo])
                  nogo = self.wts[:nc].mean()
                  quant = self.wts[nc:].reshape(2, 5).mean(axis=0)
                  #calculate flat weights (collapsing across conditions)
                  self.fwts = np.hstack([nogo, quant])
                  #pwts = np.array([1.5,  1,  1,  1,  2, 2])
                  #self.wts = np.hstack([pwts, qwts])
            self.wts, self.fwts = ensure_numerical_wts(self.wts, self.fwts)


      def __remove_outliers__(self, sd=1.5, verbose=False):
            self.data = remove_outliers(self.data, sd=sd, verbose=verbose)

      def __get_header__(self, params=None, data_style='re', labels=[], prob=np.array([.1, .3, .5, .7, .9])):
            if not hasattr(self, 'delays'):
                  self.delays = self.ssd
            qp_cols = get_header(params=params, data_style=self.data_style, labels=self.labels, prob=prob, delays=self.delays)
            if params is not None:
                  self.infolabels = qp_cols[1]
            return qp_cols[0]

      def __get_default_inits__(self):
            self.inits = get_default_inits(kind=self.kind, dynamic=self.dynamic, depends_on=self.depends_on)

      def __get_optimized_params__(self, include_ss=False, fit_noise=False):
            params = get_optimized_params(kind=self.kind, dynamic=self.dynamic, depends_on=self.depends_on)
            return params

      def __check_inits__(self, pro_ss=False, fit_noise=False):
            self.inits = check_inits(inits=self.inits, pdep=self.depends_on.keys(), kind=self.kind, dynamic=self.dynamic, pro_ss=pro_ss, fit_noise=fit_noise)

      def __make_proRT_conds__(self):
            self.data, self.rt_cix = make_proRT_conds(self.data, self.split)
            self.prort_conds_prepared = True

      def __rename_bad_cols__(self):
            self.data = rename_bad_cols(self.data)
