#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os, re


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
      #self.resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=2)*.0005
      #self.resp_lo = lambda trace: np.argmax((trace.T<=0).T, axis=3)*.0005
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


def RADD(model, ncond=2, prob=([.1, .3, .5, .7, .9]), ssd=np.arange(.2, .45, .05), ntot=10000, tb=0.650, dt=.0005, si=.01, return_traces=False):
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



def proRADD(p, ncond=6, pGo=np.arange(.2,1.2,.2), prob=([.1, .3, .5, .7, .9]), ssd=.45, ntot=2000, tb=0.545, dt=.0005, si=.01, return_traces=False, style='DDM'):
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
