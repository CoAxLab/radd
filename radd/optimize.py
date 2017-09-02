#!/usr/local/bin/env python
# Authors:
#   Kyle Dunovan (ked64@pitt.edu) &
#   Jeremy Huang (jeremyhuang@cmu.edu)

from __future__ import division
from future.utils import listvalues
import os
import numpy as np
import pandas as pd
from radd import models, theta
from radd.adapt import models_rl
from radd.tools import messages, utils
from copy import deepcopy
from scipy.optimize import basinhopping
from numpy.random import uniform
from lmfit import minimize, fit_report



class BasinBounds(object):
    """ sets conditions for step acceptance during
    basinhopping optimization routine
    Arguments:
        xmin (list): lower boundaries for each parameter
        xmax (list): upper boundaries for each parameter
    """

    def __init__(self, xmin, xmax):
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmin = bool(np.all(x >= self.xmin))
        tmax = bool(np.all(x <= self.xmax))
        return tmin and tmax


class HopStep(object):
    """ scale stepsize of basinhopping optimization according to
    individual parameters ranges (smaller steps for more sensitive params).
    See theta.get_stepsize_scalars() for parameter <--> scalar mapping.
    Arguments:
        keys (list): list of parameter names
        nlevels (list): list of levels per parameter
        stepsize (list): initial stepsize
    """

    def __init__(self, keys, nlevels, stepsize=.05):
        self.stepsize_scalars = self.get_stepsize_scalars(keys, nlevels)
        self.stepsize = stepsize
        self.np = self.stepsize_scalars.size
        self.stepsizeList = []

    def get_stepsize_scalars(self, keys, nlevels):
        """ returns an array of scalars used by for controlling
        stepsize of basinhopping algorithm for each parameter
        """
        scalar_dict={'a': .25,
                    'tr': .2,
                    'v': .35,
                    'ssv': .35,
                    'xb': .2,
                    'si': .01,
                    'z': .1,
                    'sso': .2,
                    'C': .05,
                    'B': .05,
                    'R': .01,
                    'vi': 1.1,
                    'vd': 1.5,
                    'Beta': .1}

        nl = [np.ones(nl) for nl in nlevels]
        stepsize_scalars = np.hstack([scalar_dict[k]*nl[i] for i,k in enumerate(keys)])
        stepsize_scalars = stepsize_scalars.squeeze()
        return stepsize_scalars

    def __call__(self, x):
        s = self.stepsize
        self.stepsizeList.append(s)
        ss = self.stepsize_scalars
        x = np.array([x[i] + uniform(-ss[i]*s, ss[i]*s) for i in range(self.np)])
        return x


def format_local_bounds(xmin, xmax):
    """ groups (xmin, xmax) for each parameter """
    tupler = lambda xlim: tuple([xlim[0], xlim[1]])
    # return map((tupler), zip(xmin, xmax))
    return [tupler(xl) for xl in zip(xmin, xmax)]


def format_basinhopping_bounds(basin_keys, nvary, kind='xdpm'):
    """ returns separate lists of all parameter
    min and max values """
    allbounds = theta.get_bounds(kind=kind)
    xmin, xmax = [], []
    nlevels = [np.ones(nl) for nl in nvary]
    for i, pk in enumerate(basin_keys):
        xmin.append([allbounds[pk][0]] * nlevels[i])
        xmax.append([allbounds[pk][1]] * nlevels[i])
    xmin = np.hstack(xmin).tolist()
    xmax = np.hstack(xmax).tolist()
    return xmin, xmax



class Optimizer(object):
    """ Primary optimization class for handling global and local optimization routines.

    Arguments:
        fitparams (Series): fitparams Series object
        inits (dict): parameter dictionary to initialize model with
        param_sets (list): list of init dicts to run basinhopping over
        basinparams (dict): dictionary of global optimization params
        progress (bool): initialize progress bars (default=True)
        custompath (str): local path from ~ to save results
        custompath (str): local path from ~ to save results
    """

    def __init__(self, fitparams=None, inits=None, param_sets=None, basinparams=None, progress=True, custompath=None, nruns=10):

        self.fitparams = fitparams
        self.kind = fitparams.kind
        self.learn = fitparams.learn
        self.depends_on = fitparams.depends_on
        self.ssdMethod = fitparams.ssd_method

        if inits is None:
            inits = theta.get_default_inits(kind=self.kind, depends_on=self.depends_on, learn=self.learn)
        self.inits = deepcopy(inits)

        self.progress = progress
        self.callback = None
        self.constants = {}
        self.param_sets = param_sets

        self.set_basinparams()
        self.make_progress_bars(inits=True, basin=True)
        self.nruns=nruns

        if self.learn:
            data = self.get_trials_data()
            self.simRL = models_rl.Simulator(self.inits, data=data, fitparams=self.fitparams, ssdMethod=self.ssdMethod, constants=list(constants))
        self.sim = models.Simulator(self.inits, fitparams=self.fitparams, ssdMethod=self.ssdMethod)
        self.update()
        self.make_results_dir(custompath=custompath)


    def update(self, **kwargs):
        kw_keys = list(kwargs)
        kw = pd.Series(kwargs)

        if 'fitparams' in kw_keys:
            self.fitparams = kw.fitparams
        if 'basinparams' in kw_keys:
            self.basinparams = kw.basinparams

        self.ix = self.fitparams['ix']
        self.nlevels = self.fitparams['nlevels']
        self.learn = self.fitparams['learn']
        self.pcmap = self.fitparams.pcmap
        self.inits = self.fitparams.inits
        if 'nruns' in kw_keys:
            self.nruns = kw.nruns
        if self.learn:
            constants = self.simRL.constants
            if 'constants' in kw_keys:
                constants = kw['constants']
            data = self.get_trials_data()
            self.simRL.update(inits=self.inits, fitparams=self.fitparams, data=data, constants=constants, nruns=self.nruns)
        self.sim.update(inits=self.inits, fitparams=self.fitparams)


    def hop_around(self, param_sets=None, learn=False, ratesOnly=True, fitDynamics=True, rateParams=['B', 'C', 'R']):
        """ initialize model with niter randomly generated parameter sets
        and perform global minimization using basinhopping algorithm
        ::Arguments::
            p (dict):
                parameter dictionary
        ::Returns::
            parameter set with the best fit
        """
        if param_sets is None:
            self.sample_param_sets(learn)
            param_sets = self.param_sets
        self.make_progress_bars(inits=True, basin=True)
        self.xpopt, self.xfmin = [], []
        for i, p in enumerate(param_sets):
            self.update(inits=p, force='flat', learn=learn)
            if self.progress:
                self.ibar.update(value=i, status=i+1)
            popt, fmin = self.run_basinhopping(p=p, learn=learn, ratesOnly=ratesOnly, fitDynamics=fitDynamics, rateParams=rateParams, resetProgress=False)
            self.xpopt.append(popt)
            self.xfmin.append(fmin)
        if self.progress:
            self.gbar.clear()
            self.ibar.clear()
        popt = self.xpopt[np.argmin(self.xfmin)]
        if learn:
            self.rlpopt = deepcopy(popt)
        else:
            self.popt = deepcopy(popt)
        return popt


    def popt_array_to_dict(self, popt_arr, learn=False):
        if learn:
            pdict = dict(zip(self.simRL.pvary, popt_arr))
            pdict.update(self.simRL.fixedParams)
        elif self.sim.nlevels>1:
            nvary = self.sim.nvary
            pvary = self.sim.pvary
            pdict = {}; start = 0
            for i, n in enumerate(nvary):
                pdict[pvary[i]] = np.array(popt_arr[start:start+n])
                start = n
        else:
            pdict = dict(zip(self.sim.pvary, popt_arr))
        return pdict


    def run_basinhopping(self, p, learn=False, fitDynamics=True, ratesOnly=True, rateParams=['B', 'C', 'R'], resetProgress=True, return_all=False):
        bp = self.basinparams
        popt = deepcopy(p)

        if learn:
            if fitDynamics:
                costfx = self.simRL.cost_fx_rl
            else:
                costfx = self.simRL.cost_fx
            if ratesOnly:
                pkeys = [k for k in list(p) if k not in rateParams]
            else:
                pkeys = []
            self.simRL.update(inits=p, constants=pkeys)
            x0 = self.simRL.preproc_params(p, asarray=True)
        else:
            self.sim.set_pconstant_values_matrix(p)
            x0 = self.sim.pdict_to_array(p)
            costfx = self.sim.cost_fx

        if resetProgress:
            self.make_progress_bars(inits=True, basin=True)
        self.callback = self.gbar.reset(get_call=True, gbasin=resetProgress)

        # make list of init values for all pkeys included in fit
        self.set_basin_options(learn)
        # run basinhopping on simulator.basinhopping_minimizer func
        out = basinhopping(costfx, x0=x0, minimizer_kwargs=self.tnc_options, T=bp['T'], stepsize=bp['stepsize'], niter_success=bp['nsuccess'], niter=bp['niter'], interval=bp['interval'], take_step=self.custom_step, accept_test=self.accept_step, callback=self.callback)

        self.bh_results = out
        self.fit_info = out.lowest_optimization_result

        if return_all:
            return self.fit_info

        pdict = self.popt_array_to_dict(self.fit_info.x,  learn=learn)
        popt.update(pdict)
        fmin = self.fit_info.fun
        return popt, fmin


    def gradient_descent(self, p, learn=False):

        self.update(inits=p)
        fp = self.fitparams
        flat=False
        if fp.nlevels==1:
            flat=True

        optkws = {'xtol': fp['tol'], 'ftol': fp['tol'], 'maxfev': fp['maxfev'], 'maxiter': fp['maxiter']}

        if learn:
            sim = self.simRL
            pflat = sim.pflat
            pvary = sim.pvary
            x0 = sim.preproc_params(p, asarray=True)
            lmParams = theta.loadParameters_RL(inits=p, pflat=pflat, pvary=pvary, kind=self.kind)
        else:
            sim = self.sim
            sim.set_pconstant_values_matrix(p)
            # make lmfit Parameters object
            lmParams = theta.loadParameters(inits=p, pcmap=self.pcmap, is_flat=flat, kind=self.kind)
            lmParamsNames = list(lmParams.valuesdict())
            sim.update(lmParamsNames=lmParamsNames)
            self.lmParams = lmParams

        self.lbar = utils.GradientCallback(n=fp['maxfev'], fmin=10000)

        self.lmMin = minimize(sim.cost_fx_lmfit, lmParams, method=fp['method'], tol=fp['tol'],  options=optkws, iter_cb=self.lbar.callback)

        if hasattr(self, 'lbar'):
            self.lbar.clear()
        self.param_report = fit_report(self.lmMin.params)
        return self.assess_fit(flat=flat, learn=learn)


    def set_basinparams(self, **kwargs):
        """ dictionary of global fit parameters, passed to Optimizer/Simulator objects
        """
        if not hasattr(self, 'basinparams'):
            self.basinparams =  {'ninits': 3,
                                'nsamples': 2000,
                                'interval': 10,
                                'T': .025,
                                'stepsize': .01,
                                'niter': 400,
                                'nsuccess': 100,
                                'tol': 1.e-40,
                                'method': 'TNC',
                                'init_sample_method': 'best',
                                'progress': True,
                                'disp': True}
        else:
            # fill with kwargs for the upcoming fit
            for kw_arg, kw_val in kwargs.items():
                self.basinparams[kw_arg] = kw_val


    def set_basin_options(self, learn=False):
        if learn:
            sim = self.simRL
        else:
            sim = self.sim
        bp = self.basinparams
        xmin, xmax = format_basinhopping_bounds(sim.pvary, sim.nvary, self.kind)
        tnc_bounds = format_local_bounds(xmin, xmax)

        # define custom take_step and accept_test functions
        self.accept_step = BasinBounds(xmin, xmax)
        self.custom_step = HopStep(sim.pvary, nlevels=sim.nvary, stepsize=bp['stepsize'])
        self.tnc_options = {"method": 'TNC', 'bounds': tnc_bounds, 'tol': bp['tol'], 'options': {'xtol': bp['tol'], 'ftol': bp['tol']}}

        if self.progress:
            if not hasattr(self, 'gbar') or self.gbar.pbar.displayed==False:
                self.make_progress_bars(inits=False, basin=True)
            self.callback = self.gbar.reset(get_call=True, gbasin=False, history=False)


    def sample_param_sets(self, learn=False, fitDynamics=True):
        if learn:
            self.sample_rl_theta(fitDynamics=fitDynamics)
        else:
            self.sample_theta()


    def sample_theta(self):
        """ sample *nsamples* (default=5000, see set_fitparams) different
        parameter sets (param_sets) and get model yhat for each set (param_yhats)
        """
        pkeys = self.sim.theta.keys().tolist()
        nsamples = self.basinparams['nsamples']
        if not hasattr(self, 'Theta'):
            Params = theta.random_inits(pkeys, ninits=nsamples, kind=self.kind, as_list=True)
            yhats = np.vstack([self.sim.simulate_model(p) for p in Params])
            yhatsDF = pd.DataFrame(yhats)
            self.Theta = Params
            self.ThetaYhat = yhatsDF.copy()
        Params = self.Theta
        yhatsDF = self.ThetaYhat.copy()
        psets = self.filter_params(Params, yhatsDF)
        self.param_sets = psets


    def sample_rl_theta(self, fitDynamics=True):
        if fitDynamics:
            costfx = self.simRL.cost_fx_rl
        else:
            costfx = self.simRL.cost_fx
        nsamples = self.basinparams['nsamples']
        nkeep = self.basinparams['ninits']
        idxParams = pd.DataFrame([pd.Series(self.popt)]*nsamples)
        rlParams = theta.random_inits(['B', 'C', 'R'], ninits=nsamples, kind=self.kind, as_list=True)
        rlParamsDF = pd.concat([idxParams, pd.DataFrame(rlParams.tolist())], axis=1)
        rlParams = np.array(rlParamsDF.to_dict('records'))

        self.simRL.update(inits=self.popt, constants=list(self.popt))
        cost = np.vstack([costfx(p) for p in rlParams])
        rlPsets = rlParams[np.argsort(np.hstack(cost))[:nkeep]]
        self.param_sets = rlPsets


    def filter_params(self, psets=None, yhatDF=None):
        """ sample *nsamples* (default=5000, see set_fitparams) different
        parameter sets (param_sets) and get model yhat for each set (param_yhats)
            if fit_on==subjects flat_y shape is (n_idx X ndata)
            elseif fit_on==average flat_y shape is (1 X ndata)
        """
        if psets is None:
            psets = self.Theta
            yhatDF = self.ThetaYhat.copy()

        nkeep = self.basinparams['ninits']
        y = self.fitparams.y
        wts = self.fitparams.wts
        keys = yhatDF.columns.tolist()
        ySeries = pd.Series(dict(zip(keys, y)))
        wSeries = pd.Series(dict(zip(keys, wts)))
        psets = np.asarray(psets)
        diff = yhatDF - ySeries
        wDiff = diff * wSeries
        sqDiff = wDiff.apply(lambda x: x**2, axis=0)
        sseDF = sqDiff.apply(lambda x: np.sum(x), axis=1)
        sseVals =sseDF.values
        bestIX = sseVals.argsort()[:nkeep]
        return psets[bestIX]


    def get_trials_data(self):
        if self.fitparams.fit_on=='subjects':
            idx = np.int(self.fitparams['idx'])
            return self.data[self.data.idx==idx]
        else:
            return self.data.copy()


    def assess_fit(self, flat=True, learn=False):
        """ wrapper for analyze.assess_fit calculates and stores
        rchi, AIC, BIC and other fit statistics
        ::Arguments::
            flat (bool):
                if flat, yhat have ndim=1, else ndim>1
        ::Returns::
            yhat (array), finfo (pd.Series), popt (dict)
            see gradient_descent() docstrings
        """
        fp = deepcopy(self.fitparams)
        if learn:
            y = np.hstack([self.simRL.rtBlocks, self.simRL.saccBlocks])
            wts = np.hstack([self.simRL.rt_weights, self.simRL.sacc_weights])
        else:
            y = self.sim.y.flatten()
            wts = self.sim.wts.flatten()
        # gen dict of lmfit optimized Parameters object
        popt = dict(self.lmMin.params.valuesdict())

        # TODO: extract, calculate, and store std.errors of popt
        # presults is scipy.minimize object (how to get from lmParams?)
        # presults = self.lmMin.params (maybe?)
        # then take sqrt of the diag. of the hessian to get errors
        # poptErr = np.sqrt(np.diag(presults.hess_inv.todense()))

        if not self.learn:
            # un-vectorize all parameters except conditionals
            popt = theta.scalarize_params(popt, self.pcmap)
            if fp.nlevels>1:
                popt = theta.pvary_levels_to_array(popt, self.pcmap)
        finfo = pd.Series()
        # get model-predicted yhat vector
        #self.sim.simulate_model(popt)#(self.lmMin.residual / wts) + y
        fp['yhat'] = (self.lmMin.residual / wts) + y
        # fill finfo dict with goodness-of-fit info
        finfo['idx'] = fp.idx
        finfo['pvary'] = '_'.join(list(fp.depends_on))
        finfo['cnvrg'] = self.lmMin.success
        finfo['nfev'] = self.lmMin.nfev
        finfo['nvary'] = len(self.lmMin.var_names)
        finfo['chi'] = np.sum((wts*(fp['yhat'] - y))**2)
        finfo['ndata'] = len(fp['yhat'])
        finfo['df'] = finfo.ndata - finfo.nvary
        finfo['rchi'] = finfo.chi / finfo.df
        finfo['logp'] = finfo.ndata * np.log(finfo.chi / finfo.ndata)
        try:
            finfo['AIC'] = self.lmMin.aic
            finfo['BIC'] = self.lmMin.bic
        except Exception:
            finfo['AIC'] = finfo.logp + 2 * finfo.nvary
            finfo['BIC'] = finfo.logp + np.log(finfo.ndata) * finfo.nvary
        return finfo, popt, fp['yhat']


    def write_results(self, finfo, popt, yhat):
        """ logs fit info to txtfile, fills yhatdf and fitdf
        """
        self.yhatdf = self.model.handler.fill_yhatdf(yhat=yhat, fitparams=self.fitparams)
        self.fitdf = self.model.handler.fill_fitdf(finfo=finfo, fitparams=self.fitparams)
        self.poptdf = self.model.handler.fill_poptdf(popt=popt, fitparams=self.fitparams)


    def save_results(self, custompath=None, saveobserved=False):
        """ Saves yhatdf and fitdf results to model output dir
        ::Arguments::
            saveobserved (bool):
                if True will write observedDF & wtsDF to
                model output dir
        """
        if custompath is not None:
            self.make_results_dir(custompath=custompath)
        fname = self.model.model_id
        if self.model.is_nested:
            fname='nested_models'
        resultsPath = os.path.join(self.resultsdir, fname)
        make_fname = lambda savestr: '_'.join([resultsPath, savestr+'.csv'])
        self.yhatdf.to_csv(make_fname('yhat'), index=False)
        self.fitdf.to_csv(make_fname('finfo'), index=False)
        self.poptdf.to_csv(make_fname('popt'), index=False)
        if saveobserved:
            self.model.observedDF.to_csv(make_fname('observed_data'))
            self.model.wtsDF.to_csv(make_fname('cost_weights'))


    def log_fit_info(self, finfo=None, popt=None, yhat=None):
        """ write meta-information about latest fit
        to logfile (.txt) in working directory
        """
        fp = deepcopy(self.fitparams)
        fp['yhat'] = yhat
        # lmfit-structured fit_report to write in log file
        param_report = self.param_report
        # log all fit and meta information in working directory
        messages.logger(param_report, finfo=finfo, popt=popt, fitparams=fp, kind=fp.kind, fit_on=fp.fit_on, savepath=self.resultsdir)


    def make_results_dir(self, custompath=None):
        savedir = os.path.expanduser('~')
        if custompath is not None:
            savedir = os.path.join(savedir, custompath)
            if not os.path.isdir(savedir):
                os.mkdir(savedir)
        self.resultsdir = savedir


    def make_progress_bars(self, inits=True, basin=True, lBasin=True):
        bp = self.basinparams
        if inits:
            ninits = bp['ninits']
            status=' / '.join(['Inits {}', '{}'.format(ninits)])
            self.ibar = utils.PBinJ(n=ninits, color='b', status=status)
        if basin:
            if hasattr(self, 'gbar'):
                self.gbar.clear()
            #fmin = self.sim.cost_fx(self.param_sets[0][0])
            self.gbar = utils.BasinCallback(n=bp['nsuccess'], fmin=10000)
            self.callback = self.gbar.reset(get_call=True)
        if lBasin:
            fp = self.fitparams
            if hasattr(self, 'lbar'):
                self.lbar.clear()
            self.lbar = utils.GradientCallback(n=fp['maxfev'], fmin=10000)
            self.callback = self.lbar.reset(get_call=True)
