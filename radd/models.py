#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import hstack as hs
from numpy import newaxis as na
from numpy import cumsum as csum
from scipy.stats.mstats import mquantiles as mq
from radd import theta

class Simulator(object):
    """ Core code for simulating models. All cond, trials, &
    timepoints are simulated simultaneously
    """
    def __init__(self, fitparams=None, pc_map=None, kind='xdpm', dt=.005, si=.01, learn=False):
        self.learn = learn
        self.kind = kind
        self.pc_map = pc_map
        self.fitparams = fitparams
        self.__update_steps__(dt=dt, si=si)
        self.__update__(fitparams=fitparams)

    def __update__(self, fitparams=None):
        """ update critical simulator parameters for each fit
        by providing an updated fitparams dictionary"""
        if fitparams:
            fp = dict(deepcopy(fitparams))
            self.fitparams = fp
        else:
            fp = dict(deepcopy(self.fitparams))
        self.ntime = 0
        self.tb = fp['tb']
        # set y, wts to be used cost function
        self.y = fp['y'].flatten()
        self.wts = fp['wts'].flatten()
        # misc data & model parameters used during
        # simulations/optimizations
        self.nlevels = fp['y'].ndim
        self.ntot = fp['ntrials']
        self.quantiles = fp['quantiles']
        self.dynamic = fp['dynamic']
        # include SSD's if stop-signal task
        if 'ssd_info' in list(fp):
            self.ssd_info = fp['ssd_info']
            ss=True
        self.__update_rand__(ss)
        # non conditional parameters
        self.pvc = ['a', 'tr', 'v', 'xb', 'ssv']
        if self.nlevels==1 or fp['flat']:
            self.is_flat_fit = True
        else:
            self.is_flat_fit = False
        if not self.is_flat_fit and len(list(self.pc_map))>=1:
            # remove any parameters free to vary across experimental conditions
            map((lambda pkey: self.pvc.remove(pkey)), list(self.pc_map))
        self.__init_model_functions__()
        self.__init_analyze_functions__()

    def __update_steps__(self, dt=None, si=None, tb=None):
        """ update and store stepsize parameters
        dx: size of step up/down t-->t+dt
        dt: timestep of decision traces
        si: diffusion constant (noise)
        tb: time-boundary of trial
        """
        if tb is not None:
            self.tb = tb
        if si is not None:
            self.si = si
        if dt is not None:
            self.dt = dt
        self.dx = np.sqrt(self.si * self.dt)

    def __prep_global__(self, basin_params={}, basin_keys=[]):
        """ prepare simulator for global optimization using
        scipy.optimize.basinhopping algorithm
        """
        # set parameter names used to populate params dict
        self.basin_keys = basin_keys
        # set all constant parameters in basin_params object
        self.basin_params = basin_params
        self.chunk = lambda x, nl: [array(x[i:i+nl]) for i in range(0, len(x), nl)]

    def global_cost_fx(self, x):
        """ used specifically by fit.perform_basinhopping() for Model
        objects with multiopt attr (See __opt_routine__ and perform_basinhopping
        methods of Optimizer object)
        """
        p = dict(deepcopy(self.basin_params))
        # segment 'x' into equal len arrays (one array,
        # nlevels vals long per free parameter) in basin_keys
        px = self.chunk(x, self.nlevels)
        for i, pk in enumerate(self.basin_keys):
            p[pk] = px[i]
        # simulate using filled params dictionary
        yhat = self.sim_fx(p)
        # calculate and return cost error
        return np.sum((self.wts * (yhat - self.y))**2).astype(np.float32)

    def cost_fx(self, theta, sse=False):
        """ Main cost function used for fitting all models self.sim_fx
        determines which model is simulated (determined when Simulator
        is initiated)
        """
        if type(theta) == dict:
            p = dict(deepcopy(theta))
        else:
            p = theta.valuesdict()
        yhat = self.sim_fx(p, analyze=True)
        residuals = array(self.wts * (yhat - self.y))
        if sse:
            return np.sum(residuals**2).astype(np.float32)
        return residuals.astype(np.float32)

    def __init_model_functions__(self):
        """ initiates the simulation function used in
        optimization routine
        """
        if 'dpm' in self.kind:
            self.sim_fx = self.simulate_dpm
            self.analyze_fx = self.analyze_reactive
        elif 'pro' in self.kind:
            self.sim_fx = self.simulate_pro
            self.analyze_fx = self.analyze_proactive
        elif 'irace' in self.kind:
            self.sim_fx = self.simulate_irace
            self.analyze_fx = self.analyze_reactive
        # SET STATIC/DYNAMIC BASIS FUNCTIONS
        if 'x' in self.kind and self.dynamic == 'hyp':
            # dynamic bias is hyperbolic
            self.dynamics_fx = lambda p, t: np.cosh(p['xb'][:, na] * t)
        elif 'x' in self.kind and self.dynamic == 'exp':
            # dynamic bias is exponential
            self.dynamics_fx = lambda p, t: np.exp(p['xb'][:, na] * t)
        else:
            self.dynamics_fx = lambda p, t: np.ones((self.nlevels, len(t)))

    def __init_analyze_functions__(self):
        """ initiates the analysis function used in
        optimization routine to produce the yhat vector
        """
        prob = self.quantiles
        go_axis, ss_axis = 2, 3
        if self.learn:
            go_axis-=1; ss_axis-=1
        self.go_resp = lambda trace, upper: np.argmax((trace.T >= upper).T, axis=go_axis) * self.dt
        self.ss_resp_up = lambda trace, upper: np.argmax((trace.T >= upper).T, axis=ss_axis) * self.dt
        self.ss_resp_lo = lambda trace, x: np.argmax((trace.T <= 0).T, axis=ss_axis) * self.dt
        self.go_RT = lambda ontime, rbool: ontime[:, na] + (rbool*np.where(rbool==0., np.nan, 1))
        self.ss_RT = lambda ontime, rbool: ontime[:, :, na] + (rbool*np.where(rbool==0., np.nan, 1))
        self.RTQ = lambda zpd: map((lambda x: mq(x[0][x[0] < x[1]], prob)), zpd)
        if 'irace' in self.kind:
            self.ss_resp = self.ss_resp_up
        else:
            self.ss_resp = self.ss_resp_lo

    def vectorize_params(self, p):
        """ ensures that all parameters are converted to arrays before simulation. see
        doc strings for prepare_fit() method of Model class (in build.py) for details
        regarding pc_map and logic for fitting models with parameters that depend on
        experimental conditions
        ::Arguments::
            p (dict):
                dictionary with all model parameters as
                scalars/vectors/or both
        ::Returns::
            p (dict):
                dictionary with all parameters as vectors
        """
        nl_ones = np.ones(self.nlevels).astype(np.float32)
        if 'si' in list(p):
            self.dx = np.sqrt(p['si'] * self.dt)
        if 'xb' not in list(p):
            p['xb'] = 1.0
        if self.is_flat_fit:
            p = theta.scalarize_params(p)
            return {pk:p[pk]*nl_ones for pk in list(p)}
        for pkey in self.pvc:
            p[pkey] = p[pkey] * nl_ones
        for pkey, pkc in self.pc_map.items():
            if pkc[0] not in list(p):
                p[pkey] = p[pkey] * nl_ones
            else:
                p[pkey] = array([p[pc] for pc in pkc]).astype(np.float32)
        return p

    def __update_rand__(self, ss=False):
        """ update rvector (random_floats)
        for go process and get get dynamic bias signal if 'x' model
        """
        nl, ntot, ntime = self.nlevels, self.ntot, self.ntime
        self.rvector = rs((nl, ntot, ntime))
        if ss:
            ssd, nssd, nss, nss_per, ssd_ix = self.ssd_info
            self.rvector_ss=self.rvector[:, :nss, :].reshape(nl, nssd, nss_per, ntime)

    def __update_trace_params__(self, p, ss=False):
        """ update Pg (probability of DVg +dx) and Tg (n timepoints)
        for go process and get get dynamic bias signal if 'x' model
        """
        Pg = 0.5 * (1 + p['v'] * self.dx / self.si)
        Tg = np.ceil((self.tb - p['tr']) / self.dt).astype(int)
        #t = csum([self.dt] * Tg.max())
        if ss:
            Ps, Ts = self.__update_ss_trace_params__(p)
            ss_on = 0
            if 'dpm' in self.kind:
                ss_on = np.where(Ts<Tg[:, na], Tg[:, na]-Ts, 0)
        ntime_new = np.max([Tg.max(), Ts.max()])
        if ntime_new > self.ntime:
            self.ntime = ntime_new
            self.__update_rand__(ss)
        xtb = self.dynamics_fx(p, csum([self.dt] * self.ntime))
        return Pg, xtb, Ps, ss_on

    def __update_ss_trace_params__(self, p, sso=0):
        """ update Ps (probability of DVs +dx) and Ts (n timepoints)
        for condition and each SSD of stop process
        """
        ssd = self.ssd_info[0]
        if 'sso' in list(p):
            sso = p['sso']
        Ps = 0.5 * (1 + p['ssv'] * self.dx / self.si)
        Ts = np.ceil((self.tb - (ssd + sso)) / self.dt).astype(int)
        return Ps, Ts

    def simulate_dpm(self, p, analyze=True):
        """ Simulate the dependent process model (DPM)
        ::Arguments::
            p (dict):
                parameter dictionary. values
            analyze (bool <True>):
                if True (default) return rt and accuracy information
                else, return Go and Stop decision traces.
        ::Returns::
            yhat of cost vector (ndarray)
            or list of decision traces (list of ndarrays)
        """
        p = self.vectorize_params(p)
        nl, ntot, dx = self.nlevels, self.ntot, self.dx
        ssd, nssd, nss, nss_per, ssd_ix = self.ssd_info
        Pg, xtb, Ps, ss_on = self.__update_trace_params__(p, ss=True)

        # generate Go traces (nlevels, ntrials, ntimepoints)
        DVg = xtb[:,na] * csum(np.where(self.rvector.T < Pg, dx, -dx).T, axis=2)
        ssDVg = DVg[:, :nss, :].reshape(nl, nssd, nss_per, DVg.shape[-1])
        # use array-indexing to initialize SS at DVg[:nlevels, :ssd, :trials, t=SSD]
        ssBase = ssDVg[np.arange(nl)[:,na], ssd_ix, :, ss_on][:,:,:,na]
        #self.rvector_ss=self.rvector[:, :nss, :].reshape(nl, nssd, nss_per, self.ntime)
        # add ssBaseline to SS traces (nlevels, nSSD, ntrials_perssd, ntimepoints)
        DVs = ssBase + csum(np.where(self.rvector_ss.T < Ps, dx, -dx).T, axis=3)
        if analyze:
            return self.analyze_fx(DVg, DVs, p)
        return [DVg, DVs]

    def simulate_irace(self, p, analyze=True):
        """ simulate the independent race model
        (see simulate_dpm() for I/O details)
        """
        p = self.vectorize_params(p)
        nl, ntot, dx = self.nlevels, self.ntot, self.dx
        ssd, nssd, nss, nss_per, ssd_ix = self.ssd_info
        Pg, xtb, Ps, ss_on = self.__update_trace_params__(p, ss=True)

        DVg = xtb[:,na] * csum(np.where(self.rvector.T < Pg, dx, -dx).T, axis=2)
        DVs = csum(np.where(self.rvector_ss.T < Ps, dx, -dx).T, axis=3)
        if analyze:
            return self.analyze_fx(DVg, DVs, p)
        return [DVg, DVs]

    def simulate_pro(self, p, analyze=True):
        """ Simulate the proactive competition model
        (see simulate_dpm() for I/O details)
        """
        p = self.vectorize_params(p)
        Pg, Tg, dvg = self.__update_go_process__(p)
        nl, ntot, dx = self.nlevels, self.ntot, self.dx
        DVg = xtb[:,na] * csum(np.where((rs((nl, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
        if analyze:
            return self.analyze_fx(DVg, p)
        return DVg

    def analyze_reactive(self, DVg, DVs, p):
        """ get rt and accuracy of go and stop process for simulated
        conditions generated from simulate_dpm
        """
        ssd, nssd, nss, nss_per, ssd_ix = self.ssd_info
        nl, ntot = self.nlevels, self.ntot
        gdec = self.go_resp(DVg, p['a'])
        sdec = self.ss_resp(DVs, p['a'])
        gort = self.go_RT(p['tr'], gdec)
        ssrt = self.ss_RT(ssd, sdec)
        ert = gort[:, :nss].reshape(nl, nssd, nss_per)
        eq = self.RTQ(zip(ert, ssrt))
        gq = self.RTQ(zip(gort, [self.tb] * nl))
        gacc = np.nanmean(np.where(gort < self.tb, 1, 0), axis=1)
        sacc = np.where(ert < ssrt, 0, 1).mean(axis=2)
        return hs([hs([i[ii] for i in [gacc, sacc, gq, eq]]) for ii in range(nl)])

    def analyze_proactive(self, DVg, p):
        """ get proactive rt and accuracy of go process for simulated
        conditions generated from simulate_pro
        """
        nl, ntot = self.nlevels, self.ntot
        gdec = self.go_resp(DVg, p['a'])
        gort = self.go_RT(p['tr'], gdec)
        gq = self.RTQ(zip(gort, [self.tb] * nl))
        # Get response and stop accuracy information
        gacc = 1 - np.mean(np.where(gort < self.tb, 1, 0), axis=1)
        return hs([gacc, gq])

    def simulate_rldpm(self, p, analyze=True):
        """ Simulate the dependent process model (DPM)
        with learning
        """
        p = self.vectorize_params(p)
        Pg, Tg = self.__update_go_process__(p)
        Ps, Ts = self.__update_stop_process__(p)
        ssd, nssd, nss, nss_per_ssd, ssd_ix = self.ssd_info
        nl, ntot, dx = self.nlevels, self.ntot, self.dx

        for trial in range(ntot):
            DVg = self.xtb[:, na] * csum(np.where((rs((nl, Tg.max())).T<Pg), dx, -dx).T, axis=1)
            # INITIALIZE DVs FROM DVg(t=SSD)
            if trial%2:
                init_ss = array([[DVg[i, ix] for ix in np.where(Ts<Tg[i], Tg[i]-Ts, 0)] for i in range(nl)])
                DVs = init_ss[:,:,na] + csum(np.where(rs((nss, Ts.max()))<Ps, dx, -dx), axis=1)
        if analyze:
            return self.analyze_fx(DVg, DVs, p)
        return [DVg, DVs]
