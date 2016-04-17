#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import hstack as hs
from numpy import newaxis as na
from scipy.stats.mstats import mquantiles as mq

class Simulator(object):
    """ Core code for simulating models

          * All cond, trials, & timepoints are simulated simultaneously

          * a, tr, and v parameters are initialized as vectors,
          1 x Ncond so Optimizer class can minimize a single cost function
          for multiple conditions.
    """

    def __init__(self, model=None, fitparams=None, inits=None, pc_map=None, kind='dpm', dt=.001, si=.01):

        self.dt = dt
        self.si = si
        self.dx = np.sqrt(self.si * self.dt)
        if model:
            self.fitparams = model.fitparams
            self.inits = model.inits
            self.kind = model.kind
            self.pc_map = model.pc_map
        else:
            self.fitparams = fitparams
            self.inits = inits
            self.kind = kind
            self.pc_map = pc_map
        self.__prepare_simulator__()


    def __prepare_simulator__(self):

        fp = dict(deepcopy(self.fitparams))
        self.tb = fp['tb']
        self.ncond = fp['ncond']
        self.ntot = fp['ntrials']
        self.prob = fp['prob']
        self.ssd = fp['ssd']
        self.dynamic = fp['dynamic']
        self.nssd = len(self.ssd)
        self.nss = int((.5 * self.ntot) / self.nssd)
        self.rt_cix = fp['rt_cix']
        self.is_flat = False
        self.base = 0

        if not hasattr(self, 'pvc'):
            self.__update__()

        self.__init_model_functions__()
        self.__init_analyze_functions__()


    def __update__(self, is_flat=False, y=None, wts=None):

        fp = self.fitparams
        if is_flat:
            self.pvc = deepcopy(['a', 'tr', 'v', 'xb'])
            if y is None:
                y = fp['flat_y']
                wts = fp['flat_wts']
            self.ncond = 1
        else:
            self.pvc = ['a', 'tr', 'v', 'xb']
            map((lambda pkey: self.pvc.remove(pkey)), self.pc_map.keys())
            if y is None:
                y = fp['avg_y']
                wts = fp['avg_wts']
            self.ncond = fp['ncond']
        self.y = y.flatten()
        self.wts = wts.flatten()


    def __prep_global__(self,  basin_params={}, basin_keys=[], is_flat=False):

        self.basin_keys = basin_keys
        self.basin_params = basin_params
        self.__update__(is_flat=is_flat)
        self.chunk = lambda x, n: [array(x[i:i + n]) for i in xrange(0, len(x), n)]


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
            self.ntot = int(self.ntot / self.ncond)
        elif 'irace' in self.kind:
            self.sim_fx = self.simulate_irace
            self.analyze_fx = self.analyze_reactive
        elif 'iact' in self.kind:
            self.sim_fx = self.simulate_interactive
            self.analyze_fx = self.analyze_interactive

        # SET STATIC/DYNAMIC BASIS FUNCTIONS
        self.temporal_dynamics = lambda p, t: np.ones((self.ncond, len(t)))
        if 'x' in self.kind and self.dynamic == 'hyp':
            # dynamic bias is hyperbolic
            self.temporal_dynamics = lambda p, t: np.cosh(p['xb'][:, na] * t)
        elif 'x' in self.kind and self.dynamic == 'exp':
            # dynamic bias is exponential
            self.temporal_dynamics = lambda p, t: np.exp(p['xb'][:, na] * t)


    def __init_analyze_functions__(self):
        """ initiates the analysis function used in
        optimization routine to produce the yhat vector
        """

        prob = self.prob
        nss = self.nss
        ssd = self.ssd
        tb = self.tb
        dt = self.dt

        if self.fitparams['data_style'] == 're':
            self.predict_data = self.analyze_re_data
        else:
            self.predict_data = self.analyze_pro_data

        self.resp_up = lambda trace, a: np.argmax((trace.T >= a).T, axis=2) * dt
        self.ss_resp_up = lambda trace, a: np.argmax((trace.T >= a).T, axis=3) * dt
        self.resp_lo = lambda trace: np.argmax((trace.T <= 0).T, axis=3) * dt
        self.RT = lambda ontime, rbool: ontime[:, na] + (rbool * np.where(rbool == 0, np.nan, 1))
        self.RTQ = lambda zpd: map((lambda x: mq(x[0][x[0] < x[1]], prob)), zpd)


    def vectorize_params(self, p):
        """ ensures that all parameters are converted to arrays before simulation. see
        doc strings for prepare_fit() method of Model class (in build.py) for details
        regarding pc_map and logic for fitting models with parameters that depend on
        experimental conditions

        ::Arguments::
              p (dict):
                    keys are parameter names (e.g. ['v', 'a', 'tr' ... ])
                    values are parameter values, can be vectors or floats
        ::Returns::
              p (dict):
                    dictionary with all parameters as vectors
        """

        if 'si' in p.keys():
            self.dx = np.sqrt(p['si'] * self.dt)
        if 'xb' not in p.keys():
            p['xb'] = 1.0
        for pkey in self.pvc:
            p[pkey] = p[pkey] * np.ones(self.ncond).astype(np.float32)
        for pkey, pkc in self.pc_map.items():
            if self.ncond == 1:
                break
            elif pkc[0] not in p.keys():
                p[pkey] = p[pkey] * np.ones(len(pkc)).astype(np.float32)
            else:
                p[pkey] = array([p[pc] for pc in pkc]).astype(np.float32)
        return p


    def basinhopping_minimizer(self, x):
        """ used specifically by fit.perform_basinhopping() for Model
        objects with multiopt attr (See __opt_routine__ and perform_basinhopping
        methods of Optimizer object)
        """
        p = dict(deepcopy(self.basin_params))

        # segment 'x' into equal len arrays (one array,
        # ncond vals long per free parameter) in basin_keys
        px = self.chunk(x, self.ncond)
        for i, pk in enumerate(self.basin_keys):
            p[pk] = px[i]

        if np.any(p['tr'] >= self.tb):
            return 1.e5

        yhat = self.sim_fx(p)
        cost = np.sum((self.wts * (yhat - self.y)**2))
        if hasattr(cost, '__iter__'):
            return cost[0]
        return cost


    def __cost_fx__(self, theta):
        """ Main cost function used for fitting all models self.sim_fx
        determines which model is simulated (determined when Simulator
        is initiated)
        """
        if type(theta) == dict:
            p = dict(deepcopy(theta))
        else:
            p = theta.valuesdict()
        yhat = self.sim_fx(p, analyze=True)
        return np.sum(self.wts * (self.y - yhat)**2).astype(np.float32)


    def __update_go_process__(self, p):
        """ update Pg (probability of DVg +dx) and Tg (num go process timepoints)
        for go process and get get dynamic bias signal if 'x' model
        """
        Pg = 0.5 * (1 + p['v'] * self.dx / self.si)
        Tg = np.ceil((self.tb - p['tr']) / self.dt).astype(int)
        t = np.cumsum([self.dt] * Tg.max())
        self.xtb = self.temporal_dynamics(p, t)
        #diff = [Tg.max()-tg for tg in Tg]
        #self.xtb = np.vstack([np.append(np.ones(diff[i]), xtb[i][:tg]) for i, tg in enumerate(Tg)])
        return Pg, Tg


    def __update_stop_process__(self, p, sso=0):
        """ update Ps (probability of DVs +dx) and Ts (num ss process timepoints)
        for stop process
        """
        if 'sso' in p.keys():
            sso = p['sso']
        Ps = 0.5 * (1 + p['ssv'] * self.dx / self.si)
        Ts = np.ceil((self.tb - (self.ssd + sso)) / self.dt).astype(int)
        return Ps, Ts


    def __update_interactive_params__(self, p):
        # add ss interact delay to SSD
        Ps, Ts = self.__update_stop_process__(p)
        Pg = 0.5 * (1 + p['v'] * self.dx / self.si)
        Tg = np.ceil((self.tb - p['tr']) / self.dt).astype(int)
        nt = np.max(np.hstack([Tg, Ts]))
        t = np.cumsum([self.dt] * nt)
        self.xtb = self.temporal_dynamics(p, t)
        return Pg, Tg, Ps, Ts, nt


    def simulate_dpm(self, p, analyze=True):
        """ Simulate the dependent process model (DPM)
        ::Arguments::
              p (dict):
                    parameter dictionary. values can be single floats
                    or vectors where each element is the value of that
                    parameter for a given condition
              analyze (bool <True>):
                    if True (default) return rt and accuracy information
                    (yhat in cost fx). If False, return Go and Stop proc.
        ::Returns::
              yhat of cost vector (ndarray)
              or Go & Stop processes in list (list of ndarrays)
        """

        p = self.vectorize_params(p)
        Pg, Tg = self.__update_go_process__(p)
        Ps, Ts = self.__update_stop_process__(p)
        nssd = self.nssd
        nss = self.nss
        nc = self.ncond
        dx = self.dx
        ntot = self.ntot

        DVg = self.xtb[:, na] * np.cumsum(np.where((rs((nc, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
        # INITIALIZE DVs FROM DVg(t=SSD)
        init_ss = array([[DVg[i, :nss, ix] for ix in np.where(Ts < Tg[i], Tg[i] - Ts, 0)] for i in range(nc)])
        DVs = init_ss[:, :, :, None] + np.cumsum(np.where(rs((nss, Ts.max())) < Ps, dx, -dx), axis=1)
        if analyze:
            return self.analyze_reactive(DVg, DVs, p)
        else:
            return [DVg, DVs]


    def simulate_pro(self, p, analyze=True):
        """ Simulate the proactive competition model
        (see simulate_dpm() for I/O details)
        """
        p = self.vectorize_params(p)
        Pg, Tg = self.__update_go_process__(p)
        nc = self.ncond
        dx = self.dx
        ntot = self.ntot

        DVg = self.xtb[:, na] * np.cumsum(np.where((rs((nc, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
        if analyze:
            return self.analyze_proactive(DVg, p)
        return DVg


    def simulate_irace(self, p, analyze=True):
        """ simulate the independent race model
        (see simulate_dpm() for I/O details)
        """
        p = self.vectorize_params(p)
        Pg, Tg = self.__update_go_process__(p)
        Ps, Ts = self.__update_stop_process__(p)
        nssd = self.nssd
        nss = self.nss
        nc = self.ncond
        dx = self.dx
        ntot = self.ntot

        DVg = self.xtb[:, None] * np.cumsum(np.where((rs((nc, ntot, Tg.max())).T < Pg), dx, -dx).T, axis=2)
        # INITIALIZE DVs FROM 0
        DVs = np.cumsum(np.where(rs((nc, nssd, nss, Ts.max())) < Ps, dx, -dx), axis=3)
        if analyze:
            return self.analyze_reactive(DVg, DVs, p)
        return [DVg, DVs]


    def analyze_reactive(self, DVg, DVs, p):
        """ get rt and accuracy of go and stop process for simulated
        conditions generated from simulate_dpm
        """
        nss = self.nss
        prob = self.prob
        ssd = self.ssd
        tb = self.tb
        nc = self.ncond
        nssd = self.nssd

        gdec = self.resp_up(DVg, p['a'])
        if 'irace' in self.kind:
            sdec = self.ss_resp_up(DVs, p['a'])
        else:
            sdec = self.resp_lo(DVs)
        gort = self.RT(p['tr'], gdec)
        ssrt = self.RT(ssd, sdec)
        ert = np.tile(gort[:, :nss], nssd).reshape(nc, nssd, nss)

        eq = self.RTQ(zip(ert, ssrt))
        gq = self.RTQ(zip(gort, [tb] * nc))
        gacc = np.nanmean(np.where(gort < tb, 1, 0), axis=1)
        sacc = np.where(ert < ssrt, 0, 1).mean(axis=2)
        return hs([hs([i[ii] for i in [gacc, sacc, gq, eq]]) for ii in range(nc)])


    def analyze_proactive(self, DVg, p):
        """ get proactive rt and accuracy of go process for simulated
        conditions generated from simulate_pro
        """
        prob = self.prob
        ssd = self.ssd
        tb = self.tb
        ncond = self.ncond
        ix = self.rt_cix
        gdec = self.resp_up(DVg, p['a'])
        rt = self.RT(p['tr'], gdec)
        if self.ncond == 1:
            qrt = mq(rt[rt < tb], prob=prob)
        else:
            zpd = zip([hs(rt[ix:]), hs(rt[1:ix])], [tb] * 2)
            qrt = hs(self.RTQ(zpd))
        # Get response and stop accuracy information
        gacc = 1 - np.mean(np.where(rt < tb, 1, 0), axis=1)
        return hs([gacc, qrt])

    def mean_pgo_rts(self, p, return_vals=True):
        """ Simulate proactive model and calculate mean RTs
        for all conditions rather than collapse across high and low
        """
        import pandas as pd
        tb = self.tb
        ncond = self.ncond

        DVg = self.simulate_pro(p, analyze=False)
        gdec = self.resp_up(DVg, p['a'])

        rt = self.RT(p['tr'], gdec)
        mu = np.nanmean(rt, axis=1)
        ci = pd.DataFrame(rt.T).sem() * 1.96
        std = pd.DataFrame(rt.T).std()

        self.pgo_rts = {'mu': mu, 'ci': ci, 'std': std}
        if return_vals:
            return self.pgo_rts


    def analyze_data(self, DVg, DVs=None, p=None):
        """ get rt and accuracy of go and stop process for simulated
        conditions generated from simulate_dpm
        """
        nss = self.nss
        prob = self.prob
        ssd = self.ssd
        tb = self.tb
        nc = self.ncond
        nssd = self.nssd

        if 'sso' in p.keys():
            ssd = ssd + p['sso']
        gdec = self.resp_up(DVg, p['a'])
        gort = self.RT(p['tr'], gdec)
        if 'pro' not in self.kind:
            if 'irace' in self.kind:
                sdec = self.ss_resp_up(DVs, p['a'])
            else:
                sdec = self.resp_lo(DVs)
            ssrt = self.RT(ssd, sdec)

        #ert = gort[:,:nss][:, None]*np.ones_like(ssrt)
        ert = np.tile(gort[:, :nss], nssd).reshape(nc, nssd, nss)

        eq = self.RTQ(zip(ert, ssrt))
        gq = self.RTQ(zip(gort, [tb] * nc))
        gacc = np.nanmean(np.where(gort < tb, 1, 0), axis=1)
        sacc = np.where(ert < ssrt, 0, 1).mean(axis=2)
        return hs([hs([i[ii] for i in [gacc, sacc, gq, eq]]) for ii in range(nc)])


    def analyze_pro_data(self, DVg, p):
        prob = self.prob
        tb = self.tb
        ncond = self.ncond
        ix = self.rt_cix
        gdec = self.resp_up(DVg, p['a'])
        rt = self.RT(p['tr'], gdec)
        hi = hs(rt[ix:])
        low = hs(rt[1:ix])
        # Get response and stop accuracy information
        gacc = np.where(rt < self.tb, 0, 1)
        return [gacc, hi[~np.isnan(hi)], low[~np.isnan(low)]]


    def analyze_re_data(self, dv, p):
        import pandas as pd
        prob = self.prob
        ssd = self.ssd
        tb = self.tb
        nc = self.ncond
        nss = self.nss
        nssd = self.nssd
        ntot = self.ntot
        nsstot = nss * nssd
        # get condition labels
        conditions = np.sort(self.fitparams['labels'] * int(ntot))
        # assign go trials pseudo ssd of 1000
        delays = np.append(np.array([[c] * nss for c in sd]), [1000] * nss * nssd)
        ttype = np.tile(array([['stop'] * nsstot + ['go'] * nsstot]), nc)

        DVg, DVs = dv
        if 'sso' in p.keys():
            ssd = ssd + p['sso']
        gdec = self.resp_up(DVg, p['a'])
        if 'irace' in self.kind:
            sdec = self.ss_resp_up(DVs, p['a'])
        else:
            sdec = self.resp_lo(DVs)
        gort = self.RT(p['tr'], gdec)
        ssrt = self.RT(ssd, sdec)
        ert = np.tile(gort[:, :nss], nssd).reshape(nc, nssd, nss)

        # compare go proc. rt to ssrt and get error resp. (go proc win err)
        ssgo = map((lambda z: z[0][z[0] < z[1]]), zip(ert, ssrt))
        gresp = np.where(gort[:, nss * nssd:] < tb, 1, 0)
        ssresp = np.where(ert < ssrt, 1, 0).reshape(nc, nsstot)

        resp = np.concatenate([ssresp, gresp], axis=1)
        rt = np.concatenate([ert.reshape(nc, nsstot), gort[:, :nsstot]], axis=1)

        out = {'ssd': np.tile(delays, nc),
               'response': np.hstack(resp),
               'rt': np.hstack(rt),
               'cond': conditions,
               'ttype': ttype}

        return pd.DataFrame(out)
