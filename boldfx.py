from __future__ import division
from copy import deepcopy
import numpy as np
from pandas import DataFrame as pDF
import seaborn as sns
import matplotlib.pyplot as plt
from radd import build, vis
from radd.toolbox.utils import get_xbias_theta

class BOLD(object):
      """ Simulated BOLD response // neural activity
      from optimized RADD models

      Largley based on mapping lists of arrays from single
      conditions to lambda/def functions, due to many non-uniform
      ndarray transformations (i.e., boolean indexing that leads to
      different lengths along a given axis of the full DVg)

      """


      def __init__(self, model, sim_xbias=False):

            if not hasattr(model, 'simulator'):
                  # GENERATE MODEL SIMULATOR
                  model.make_simulator()

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
                  self.p = model.inits

            self.ncond = model.ncond
            self.simulator = model.simulator
            self.dt = self.simulator.dt

            self.__init_process_functions__()


      def __init_process_functions__(self):

            #############################################################
            # LAMBDA FUNC: GET_RT FOR ALL BOUNDARY CROSSES, 999 OTHERWISE
            # x[0] = DVg (go accumulator: all trials in single condition)
            # x[1] = a   (boundary, single condition)
            # x[2] = tr  (onset time, cond i)
            self.get_rt = lambda x: np.where(x[0].max(axis=1)>=x[1], x[2]+np.argmax(x[0]>=x[1], axis=1)*self.dt, 999)

            ###############################################################
            # LAMBDA FUNC: GET_GO_TRACES FOR ALL THAT CROSS BOUND BEFORE TB
            # x[0] = DVg (go accumulator: all trials in single condition)
            # x[1] = RT  (all trials in single condition)
            # x[1] = timeboundary (simulator.tb)
            self.get_go_traces = lambda x: x[0][np.where(x[1]<=x[2], True, False)]
            self.get_nogo_traces = lambda x: x[0][np.where(x[1]>=x[2], True, False)]
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


      def generate_go_traces(self):
            """ ensures parameters are vectorized and sets
            bound and onset attr. before generating simulated traces
            """

            # ensure parameters are all vectorized
            self.p = self.simulator.vectorize_params(self.p)

            # init timebound, bound, onset-time attr
            self.tb = self.simulator.tb
            self.bound = self.p['a']
            self.onset = self.p['tr']
            #simulate decision traces
            self.dvg = self.simulator.sim_fx(self.p, analyze=False)


      def get_bold_dfs(self, hemodynamic=True, decay=False, get_dfs=False):
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

            if not hasattr(self, 'dvg'):
                  # simulate decision traces
                  self.generate_go_traces()

            # zip dvg[ncond, ntrials, ntime], bound[a_c..a_ncond], onset[tr_c..tr_ncond]
            zipped_input_rt = zip(self.dvg, self.bound[:, None], self.onset[:,None])
            # get rt for all conditions separately where boundary crossed [DVg >= a]
            rt = np.asarray(map(self.get_rt, zipped_input_rt))

            # zip dvg[ncond, ntrials, ntime], rt[rt_c..rt_ncond], [tb]*ncond
            zipped_input_traces = zip(self.dvg, rt, [self.tb]*len(rt))
            # boolean index traces so that DVg[RT<=TB]
            resp_trial_arrays = map(self.get_go_traces, zipped_input_traces)
            nogo_trial_arrays = map(self.get_nogo_traces, zipped_input_traces)
            # zip dvg(resp=1), bound[a_c..a_ncond], [decay]*ncond
            zipped_input_caproll = zip(resp_trial_arrays, self.bound, [decay]*self.ncond)
            zipped_nogo_caproll = zip(nogo_trial_arrays, self.bound, [decay]*self.ncond)
            # generate dataframe of capped and time-thresholded go traces
            self.traces = map(self.cap_roll_traces, zipped_input_caproll)
            self.nogo = map(self.cap_roll_traces, zipped_nogo_caproll)

            if hemodynamic:
                  #bold_df_super = map(lambda x: x.cumsum(), self.bold_df_list)
                  #self.hemodynamics = np.array(map(self.get_hemo, self.bold_df_list))
                  hemo = map(self.get_hemo, self.traces)
                  nogo_hemo = map(self.get_hemo, self.nogo)

                  self.hemo = np.array([np.mean(np.cumsum(gtrace, axis=0).max().values) for gtrace in self.traces])
                  self.nogo_hemo = np.array([np.mean(np.cumsum(strace, axis=0).max().values) for strace in self.nogo])

            if get_dfs:
                  return self.traces



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
                  traces_capped = np.concatenate((traces, traces[:,::-1]), axis=1)
            else:
                  traces_capped = traces
            traces_df = pDF(traces_capped.T)
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



      def plot_results(self, save=False):

            redgreen = lambda nc: sns.blend_palette(["#c0392b", "#27ae60"], n_colors=nc)

            x=np.array(['0', '25', '50', '50', '75', '100'])
            y=np.concatenate((self.nogo_hemo[:3], self.hemo[:3]))
            sns.set(style='ticks', font_scale=1.5)
            ax = sns.barplot(x=np.arange(len(x)), y=y, color=redgreen(6))
            ax.set_xticklabels(x)
            ax.set_xlabel('pGo')
            ax.set_ylabel('$\Sigma \Theta_{Go}$', fontsize=26)
            ax.set_title('Drift-Rate Effect on BOLD Activity (no decay assumed)')
            ax.set_ylim(50, 100)
            sns.despine()
            if save:
                  plt.savefig('bold_pred_drift.png', dpi=300)
