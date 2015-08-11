#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from numpy import array
import pandas as pd
from numpy import concatenate as concat
import seaborn as sns
import matplotlib.pyplot as plt
from radd import build, vis
from radd.toolbox.colors import get_cpals
from radd.toolbox.theta import get_xbias_theta
from radd.toolbox.messages import describe_model
from radd.models import Simulator


class BOLD(Simulator):
      """ Simulated BOLD response // neural activity
      from optimized RADD models

      Largley based on mapping lists of arrays from single
      conditions to lambda/def functions, due to many non-uniform
      ndarray transformations (i.e., boolean indexing that leads to
      different lengths along a given axis of the full DVg)

      """

      def __init__(self, model, sim_xbias=False):

            model.fit_whole_model=False

            if hasattr(model, 'popt'):
                  self.p = model.popt
            elif sim_xbias:
                  self.p = get_xbias_theta(model)
            else:
                  try:
                        model.fit_whole_model=False
                        model.__get_default_inits__()
                  except Exception:
                        pass

                  model.fit_whole_model=False
                  self.p = model.inits

            # GENERATE MODEL SIMULATOR
            super(BOLD, self).__init__(model=model, inits=model.inits, pc_map=model.pc_map, kind=model.kind, prepare=True, is_flat=False, is_bold=True)

            self.ncond = model.ncond
            self.depends_on = model.depends_on
            self.simulate = self.sim_fx
            self.labels = model.labels

            self.__init_process_functions__()


      def __init_process_functions__(self):

            #############################################################
            # LAMBDA FUNC: GET_RT FOR ALL BOUNDARY CROSSES, 999 OTHERWISE
            # x[0] = DVg (go accumulator: all trials in single condition)
            # x[1] = a   (boundary, single condition)
            # x[2] = tr  (onset time, cond i)
            self.get_rt = lambda x: np.where(x[0].max(axis=1)>=x[1], x[2]+np.argmax(x[0]>=x[1], axis=1)*self.dt, 999)
            self.get_ssrt = lambda dvs, delay: np.where(dvs.min(axis=3)<=0, delay+np.argmax(dvs<=0, axis=3)*self.dt, 999)

            ###############################################################
            # LAMBDA FUNC: GET_GO_TRACES FOR ALL THAT CROSS BOUND BEFORE TB
            # x[0] = DVg (go accumulator: all trials in single condition)
            # x[1] = RT  (all trials in single condition)
            # x[1] = timeboundary (simulator.tb)
            self.get_go_traces = lambda x: x[0][np.where(x[1]<=x[2], True, False)]
            self.get_ng_traces = lambda x: x[0][np.where(x[1]>=x[2], True, False)]

            ###################################################################################
            # LAMBDA FUNC: CAP_ROLL_TRACES cap traces at bound and roll nans to back (opt. decay)
            # x[0] = DVg (go accumulator: all trials in single condition)
            # x[1] = a (boundary, single condition)
            # x[2] = BOOL (decay if True, <default False>)
            self.cap_roll_traces = lambda x: self.drop_n_roll(self.cap_n_bound(x[0],x[1],x[2]))

            ###################################################################################
            # LAMBDA FUNC: GET_HEMO get last occurence of numeric vals along cumsum axis=1
            # df = BOLD DF Super (all trials cumsum(axis=1), single condition)
            #self.get_hemo = lambda df: df.cumsum().apply(self.__ix_lastnum__, axis=1)
            #self.get_first_numeric = lambda df: df.apply(self.__ix_firstnum__, axis=1)
            def get_last_numeric(x):
                  if x.last_valid_index() is None:
                        return None
                  else:
                        return x[x.last_valid_index()]

            self.get_hemo = lambda d: d.cumsum(axis=0).apply(get_last_numeric, axis=0).dropna().values


      def generate_radd_traces(self):

            """ Get go,ssrt using same function as proactive then use the indices for
            correct go and correct stop trials to slice summed, but momentary, dvg/dvs evidence vectors
            (not accumulated over time just a basic sum of the vectors)
            """
            # ensure parameters are all vectorized
            self.p = self.vectorize_params(self.p)
            Pg, Tg = self.__update_go_process__(self.p)
            Ps, Ts = self.__update_stop_process__(self.p)

            self.bound = self.p['a']
            self.onset = self.p['tr']

            ncond=self.ncond; ntot=self.ntot; dx=self.dx;
            base=self.base; self.nss_all=self.nss; nssd=self.nssd;
            ssd=self.ssd; nss = self.nss_all/self.nssd; xtb=self.xtb;

            get_ssbase = lambda Ts,Tg,DVg: array([[DVc[:nss/nssd, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i, DVc in enumerate(DVg)])[:,:,:,None]

            self.gomoments = xtb[:,None]*np.where((rs((ncond, ntot, Tg.max())).T<Pg), dx,-dx).T
            self.ssmoments = np.where(rs((ncond, nssd, nss, Ts.max()))<Ps, dx, -dx)
            DVg = base[:, None]+xtb[:,None]* np.cumsum(self.gomoments, axis=2)
            self.dvg = DVg[:,:nss_all,:]
            self.dvs = self.get_ssbase(Ts,Tg,DVg) + np.cumsum(self.ssmoments, axis=3)

            dg = self.gomoments[:,nss_all:,:].reshape(ncond,nssd,nss,Tg.max())
            ds = self.ssmoments.copy()

            ss_list, go_list=[],[]
            for i, (s, g) in enumerate(zip(ds, dg)):
                  diff = Ts-Tg[i]
                  pad_go = diff[diff>0]
                  pad_ss = abs(diff[diff<0])

                  go_list.extend([concat((np.zeros((nss, gp)), g[i]), axis=1) for gp in pad_go])
                  go_list.extend([g[x] for x in range(len(pad_ss))])

                  ss_list.extend([concat((np.zeros((nss, abs(spad))), s[i,:,-(Tg[i]-spad):]), axis=1) for spad in pad_ss])
                  ss_list.extend([s[i,:,-tss:] for tss in Ts[:len(pad_go)]])

            r = map((lambda x: np.cumsum((x[0] + x[1]), axis=1)), zip(go_list, ss_list))


      def generate_pro_traces(self):
            """ ensures parameters are vectorized and sets
            bound and onset attr. before generating simulated traces
            """

            # ensure parameters are all vectorized
            self.p = self.vectorize_params(self.p)

            # init timebound, bound, onset-time attr
            self.bound = self.p['a']
            self.onset = self.p['tr']
            #simulate decision traces
            self.dvg = self.sim_fx(self.p, analyze=False)


      def simulate_bold(self, hemodynamic=True, decay=False, get_dfs=False, shape='long', savestr='./',  save=False):

            """ gets RT of boundary crossing for boolean selecting
            go traces which are then filtered and stored in DF

            Traces (trials) that meet the criteria:
                  :: [DVg(trial_vals) >= a] & [DVg(trial_rt)<tb] ::
            are then
                  ::capped:: setting all values after initial threshold to NAN
                  ::rolled:: shifted so all NAN values are moved to end of trace

            ::Arguments::
                  hemodynamic (bool <True>):
                        simulate superposition property (cumumulative accumulator)
                  decay (bool <False>):
                        concatenate mirror image of input array
                        along axis 1 (i.e., bold up, decay down)
                  get_dfs (bool <False>):
                        return list of simulated bold dataframes
            """

            if not hasattr(self, 'dvg') and 'pro' in self.kind:
                  # simulate decision traces
                  self.generate_pro_traces()
            elif not hasattr(self, 'dvg') and 'radd' in self.kind:
                  # simulate decision traces
                  self.generate_radd_traces()

            # zip dvg[ncond, ntrials, ntime], bound[a_c..a_ncond], onset[tr_c..tr_ncond]
            zipped_input_rt = zip(self.dvg, self.bound[:, None], self.onset[:,None])
            # get rt for all conditions separately where boundary crossed [DVg >= a]
            rt = np.asarray(map(self.get_rt, zipped_input_rt))

            # zip dvg[ncond, ntrials, ntime], rt[rt_c..rt_ncond], [tb]*ncond
            zipped_rt_traces = zip(self.dvg, rt, [self.tb]*len(rt))
            # boolean index traces so that DVg[RT<=TB]
            go_trial_arrays = map(self.get_go_traces, zipped_rt_traces)

            if 'pro' in self.kind:
                  # zip dvg(resp=1), bound[a_c..a_ncond], [decay]*ncond
                  ng_trial_arrays = map(self.get_ng_traces, zipped_rt_traces)
            elif 'radd' in self.kind:
                  ssrt = self.get_ssrt(self.dvs, self.ssd[:,None])
                  strial_gorts = rt[:,:self.nss_all].reshape(self.ncond,self.nssd,self.nss)
                  # zip DVs, SSRT, ssGoRT
                  zipped_rt_traces = zip(self.dvs, ssrt, strial_gorts)
                  ng_trial_arrays = map(self.get_ng_traces, zipped_rt_traces)


            zipped_go_caproll = zip(go_trial_arrays, self.bound, [decay]*self.ncond)
            zipped_ng_caproll = zip(ng_trial_arrays, self.bound, [decay]*self.ncond)
            # generate dataframe of capped and time-thresholded go traces
            self.go_traces = map(self.cap_roll_traces, zipped_go_caproll)
            self.ng_traces = map(self.cap_roll_traces, zipped_ng_caproll)

            if hemodynamic:
                  self.make_bold_dfs(shape=shape, save=save)
                  self.mean_go_traces = [gt.mean(axis=1).dropna().values for gt in self.go_traces]
                  self.mean_ng_traces = [ng.mean(axis=1).dropna().values for ng in self.ng_traces]


      def make_bold_dfs(self, shape='long', savestr='./', save=False):

            go_csum = [gt.cumsum(axis=0).max(axis=0).values for gt in self.go_traces]
            ng_csum = [ng.cumsum(axis=0).max(axis=0).values for ng in self.ng_traces]

            if shape=='fat':
                  dfgo_csum = pd.DataFrame.from_dict(OrderedDict(zip(self.labels, go_csum)), orient='index').T
                  dfng_csum = pd.DataFrame.from_dict(OrderedDict(zip(self.labels, ng_csum)), orient='index').T
                  dfgo_csum.insert(0, 'choice', 'go')
                  dfng_csum.insert(0, 'choice', 'nogo')

                  self.bold_mag = pd.concat([dfgo_csum, dfng_csum])
                  self.bold_mag.index.name='cond'
            elif shape=='long':
                  go_csum.extend(ng_csum)
                  boldf_list = []
                  choices = np.sort(['go', 'nogo']*self.ncond)
                  lbls=[int(lbl) for lbl in self.labels]*2
                  for i, gb in enumerate(go_csum):
                        boldf_list.append(pd.DataFrame.from_dict({'cond':lbls[i],'csum':gb, 'choice': choices[i]}))
                  self.bold_mag = pd.concat(boldf_list)

            if save:
                  self.bold_mag.to_csv(''.join([savepth, 'bold_mag.csv']))



      def cap_n_bound(self, traces, bound, decay=False):
            """ take 2d array and set values >= upper boundary as np.nan
            return dataframe of capped ndarray, assumes all traces meet
            the criteria trials[any(val >= a)]

            ::Arguments::
                  traces (ndarray):
                        go decision traces for single cond
                  bound (float):
                        boundary for given condition, all
                        values>=bound --> np.nan
                  decay (bool <False>):
                        concatenate mirror image of input array
                        along axis 1 (i.e., bold up, decay down)
            ::Returns::
                  traces_df (DataFrame):
                        all go traces
            """

            traces[traces>=bound]=np.nan
            if decay:
                  traces_capped = concat((traces, traces[:,::-1]), axis=1)
            else:
                  traces_capped = traces
            traces_df = pd.DataFrame(traces_capped.T)
            return traces_df


      def __ix_lastnum__(self, s):

            """ search for index of last numeric (non-NAN)
            value in series (Modified from goo.gl/tId5Rw)

            intended acces through lambda func index_numeric
                  df.apply(get_last_numeric, axis=0)
            ::Arguments::
                  s (Series):
                        row or column slice of DataFrame
            ::Returns::
                  index of last numeric value of series
            """

            if s.last_valid_index() is None:
                  return None
            else:
                  return s[s.last_valid_index()]


      def __ix_firstnum__(self, s):

            """ see __ix_lastnum__
            ::Arguments::
                  s (Series):
                        row or column slice of DataFrame
            ::Returns::
                  index of first numeric value of series
            """

            if s.first_valid_index() is None:
                  return None
            else:
                  return s[s.first_valid_index()]


      def drop_n_roll(self, df, na_position='last'):
            """ rolls all NAN to end of column series for all cols in df
            (Modified from http://goo.gl/3SHD3q)

            ::Arguments:;
                  df (DataFrame):
                        accumulators for all ntime (ROWS) x ntrials (COLS)
                        with possible NANs separating inital and
                        re-crossing of threshold
                  na_position (str):
                        'first': roll all NANs to top rows
                        'last': roll all NANs to bottom rows
            ::Returns::
                  df (DataFrame):
                        accumulators for all ntime (ROWS) x ntrials (COLS)
                        with all NANs rolled to end of row axis
            """

            for c, cseries in df.iteritems():
                  result = np.full(len(cseries), np.nan, dtype=cseries.dtype)
                  mask = cseries.notnull()
                  N = mask.sum()
                  if na_position == 'last':
                        result[:N] = cseries.loc[mask]
                  elif na_position == 'first':
                        result[-N:] = cseries.loc[mask]
                  else:
                        raise ValueError('na_position {!r} unrecognized'.format(na_position))
                  df[c]=result

            return df

      def simulate(self, theta=None, analyze=True):
            """ simulate yhat vector using popt or inits
            if model is not optimized

            :: Arguments ::
                  analyze (bool):
                        if True (default) returns yhat vector
                        if False, returns decision traces
            :: Returns ::
                  out (array):
                        1d array if analyze is True
                        ndarray of decision traces if False
            """

            theta=self.p
            theta = self.simulator.vectorize_params(theta)
            out = self.simulator.sim_fx(theta, analyze=analyze)

            return out

      def plot_means(self, save=False):

            redgreen = lambda nc: sns.blend_palette(["#c0392b", "#27ae60"], n_colors=nc)

            titl=describe_model(self.depends_on)
            df = self.bold_mag.copy()

            df.ix[(df.choice=='go')&(df.cond<=50), 'cond']=60
            df.ix[(df.choice=='nogo')&(df.cond>=50), 'cond']=40
            ax = sns.barplot('cond', 'csum', data=df, order=np.sort(df.cond.unique()), palette=redgreen(6))

            mu = df.groupby(['choice','cond']).mean()['csum']
            ax.set_ylim(mu.min()*.55, mu.max()*1.15)

            ax.set_xlabel('pGo', fontsize=22)
            ax.set_ylabel('$\Sigma \Theta_{G}$',fontsize=26)
            ax.set_title(" ".join([titl, 'effect(s) on BOLD Simulations']))
            sns.despine()

            if save:
                  plt.savefig(''.join([titl,'.png']), dpi=300)


      def plot_traces(self, save=False):

            f, ax = plt.subplots(1, figsize=(6,4))
            cpals = get_cpals(); sns.set(style='white', font_scale=1.5)

            titl=describe_model(self.depends_on)
            gmu = [ggt.mean(axis=1) for ggt in self.go_traces]
            nmu = [ngt.mean(axis=1) for ngt in self.ng_traces]

            gc = cpals['gpal'](len(gmu))
            nc = cpals['rpal'](len(nmu))
            tr = self.onset*1000

            gx = [tr[i] + np.arange(len(gmu[i])) for i in range(self.ncond)]
            nx = [tr[i] + np.arange(len(nmu[i])) for i in range(self.ncond)]

            for i in range(len(gmu)):
                  ax.plot(gx[i], gmu[i], color=gc[i])
            for i in range(len(nmu)):
                  ax.plot(nx[i], nmu[i], color = nc[i])

            ax.set_ylim(0, gmu[0].max()*1.2)
            ax.set_xlim(gx[0].min()*.9, nx[0].max())

            ax.set_xlabel('Time', fontsize=18); ax.set_xticklabels([])
            ax.set_ylabel('$\Theta_{G}$', fontsize=23)
            ax.set_title(" ".join(['Rising Acitivty of', titl, 'Accumulator']))

            sns.despine()
            plt.tight_layout()
            if save:
                  plt.savefig(''.join([titl, '_traces.png']), dpi=300)
