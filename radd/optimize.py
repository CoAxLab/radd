#!/usr/local/bin/env python
# Authors:
#   Kyle Dunovan (ked64@pitt.edu) &
#   Jeremy Huang (jeremyhuang@cmu.edu)

from __future__ import division
from future.utils import listvalues
import os, sys
import numpy as np
import pandas as pd
from radd import models, theta
from radd.adapt import models_rl
from radd.tools import messages, utils
from copy import deepcopy
from scipy.optimize import basinhopping, differential_evolution, fmin
from numpy.random import uniform
from lmfit import minimize, fit_report
from IPython.display import clear_output


class GlobalBounds(object):
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

    def __init__(self, keys, nlevels, stepsize=.15):
        self.stepsize_scalars = self.get_stepsize_scalars(keys, nlevels)
        self.stepsize = stepsize
        self.np = self.stepsize_scalars.size
        self.stepsizeList = []

    def get_stepsize_scalars(self, keys, nlevels):
        """ returns an array of scalars used by for controlling
        stepsize of basinhopping algorithm for each parameter
        """
        scalar_dict={'a': .075,
                    'tr': .025,
                    'v': .18,
                    'ssv': .2,
                    'xb': .3,
                    'sso': .01,
                    'z': .05,
                    'si': .005,
                    'C': .002,
                    'B': .008,
                    'R': .002,
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


def format_basinhopping_bounds(basin_keys, nvary, kind='xdpm', tb=None):
    """ returns separate lists of all parameter
    min and max values """
    allbounds = theta.get_bounds(kind=kind, tb=tb)
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

    def __init__(self, fitparams={}, basinparams={}, inits=None, param_sets=None, custompath=None, nruns=10, data=None):

        self.fitparams = fitparams
        self.basinparams = basinparams
        self.kind = fitparams.kind
        self.learn = fitparams.learn
        self.depends_on = fitparams.depends_on
        self.ssdMethod = fitparams.ssd_method
        self.data = data

        if inits is None:
            inits = theta.get_default_inits(kind=self.kind, depends_on=self.depends_on, learn=self.learn)

        self.inits = deepcopy(inits)
        self.callback = None
        self.constants = list(inits)
        self.param_sets = param_sets
        self.nruns=nruns

        self.progress = self.basinparams['progress']
        if self.progress:
            self.make_progress_bars(inits=True, basin=True)

        if self.learn:
            data = self.get_trials_data()
            self.simRL = models_rl.Simulator(self.inits, data=data, fitparams=self.fitparams, ssdMethod=self.ssdMethod, constants=self.constants)

        self.sim = models.Simulator(self.inits, fitparams=self.fitparams, ssdMethod=self.ssdMethod)
        self.update()
        self.make_results_dir(custompath=custompath)


    def update(self, ksData=None, **kwargs):
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
        self.progress = self.basinparams['progress']
        self.ksTest = False

        if ksData is not None:
            self.ksTest = True

        if 'nruns' in kw_keys:
            self.nruns = kw.nruns

        if self.learn:
            constants = self.simRL.constants
            if 'constants' in kw_keys:
                constants = kw['constants']
            data = self.get_trials_data()
            self.simRL.update(inits=self.inits, fitparams=self.fitparams, data=data, constants=constants, nruns=self.nruns)

        self.sim.update(inits=self.inits, fitparams=self.fitparams, ksData=ksData)


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
        if self.progress:
            self.make_progress_bars(inits=True, basin=True)

        xpopt, xfmin, global_results = [], [], []
        self.gpopt = []
        for i, p in enumerate(param_sets):
            self.update(inits=p, force='flat', learn=learn)

            if self.progress:
                self.ibar.update(value=i, status=i+1)
            popt, fmin, out = self.optimize_global(p=p, learn=learn, ratesOnly=ratesOnly, fitDynamics=fitDynamics, rateParams=rateParams, resetProgress=False, return_all=True)
            xpopt.append(popt)
            self.gpopt.append(deepcopy(popt))
            xfmin.append(fmin)
            global_results.append(out)

        if self.progress:
            self.gbar.clear()
            self.ibar.clear()

        keep_ix = np.argmin(xfmin)
        popt = xpopt[keep_ix]
        self.global_popt = deepcopy(popt)
        self.global_fmin = xfmin[keep_ix]
        self.global_results = global_results[keep_ix]

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


    def optimize_global(self, p, learn=False, fitDynamics=True, ratesOnly=True, rateParams=['B', 'C', 'R'], resetProgress=False, return_all=False):
        """ Global optimization with basinhopping (or differential_evolution)
        ::Arguments::
            p (dict):               parameter dictionary
            learn (bool):           fit adaptive model if True (default: False)
            fitDynamics (bool):     fit adaptive model to timeseries of rt/accuracy
            ratesOnly (bool):       restrict free parameters to learning rates
            rate params (list):     learning rate parameters
            resetProgress (bool):   reset global progress bar
            return_all (bool):      return all results from optimzation
        ::Returns::
            popt (dict):            optimized parameter dictionary
            fmin (float):           function minimum
        """
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
            if self.ksTest:
                costfx = self.sim.ks_stat

        if self.progress:
            if resetProgress:
                # print("")
                self.make_progress_bars(inits=True, basin=True)
                clear_output()
            self.callback = self.gbar.reset(get_call=True, gbasin=resetProgress)

        # create args for customizing global optimizer
        self.set_global_options(learn=learn)

        # run global optimization (basinhopping/differential_evolution)
        if bp['method']=='basin':
            out = basinhopping(costfx, x0=x0, minimizer_kwargs=self.polish_args, T=bp['T'], stepsize=bp['stepsize'], niter_success=bp['nsuccess'], niter=bp['niter'], interval=bp['interval'], take_step=self.custom_step, accept_test=self.accept_step, callback=self.callback)
            fit_info = out.lowest_optimization_result

        elif bp['method']=='evolution':
            out = differential_evolution(costfx, bounds=self.polish_args['bounds'], popsize=bp['popsize'], recombination=bp['recombination'], mutation=bp['mutation'], strategy=bp['strategy'], disp=bp['disp'], polish=True, maxiter=bp['maxiter'], tol=bp['tol'], callback=self.callback, atol=self.fitparams['tol'])
            if self.progress:
                self.gbar.clear()
            fit_info = out

        pdict = self.popt_array_to_dict(fit_info.x,  learn=learn)
        popt.update(pdict)
        fmin = fit_info.fun
        if return_all:
            return popt, fmin, out

        return popt, fmin


    def gradient_descent(self, p, learn=False, flat=False):
        """ Local optimization with Nelder-Mead Simplex algorithm (gradient descent)
        ::Arguments::
            p (dict):       parameter dictionary
            learn (bool):   fit adaptive model if True (default: False)
            flat
        ::Returns::
            finfo (Series): summary stats of optimization
            popt (dict):    optimized parameter dictionary
            yhat (array):   model-predicted data vector
        """
        flat= False

        fp = self.fitparams
        fp['inits'] = p
        self.update(fitparams=fp, ksData=self.sim.ksData)
        fp = self.fitparams
        if fp.nlevels==1:
            flat=True

        optkws = {'xtol': fp['tol'], 'ftol': fp['tol'], 'maxfev': fp['maxfev'], 'maxiter': fp['maxiter']}
        if learn:
            sim = self.simRL
            pflat = sim.pflat
            pvary = sim.pvary
            x0 = sim.preproc_params(p, asarray=True)
            lmParams = theta.loadParameters_RL(inits=p, pflat=pflat, pvary=pvary, kind=self.kind)
            costfx = sim.cost_fx_lmfit
        else:
            sim = self.sim
            sim.set_pconstant_values_matrix(p)
            lmParams = theta.loadParameters(inits=p, pcmap=self.pcmap, is_flat=flat, kind=self.kind)
            lmParamsNames = list(lmParams.valuesdict())
            sim.update(lmParamsNames=lmParamsNames, ksData=sim.ksData)
            costfx = sim.cost_fx_lmfit
            # if self.ksTest:
                # costfx = sim.ks_stat_lmfit

        self.lmParams = lmParams

        if self.progress:
            self.lbar = utils.GradientCallback(n=fp['maxfev'], fmin=10000)
            self.lcallback = self.lbar.callback
        else:
            self.lcallback = None

        if 'least' in fp['method']:
            self.lmMin = minimize(costfx, lmParams, method=fp['method'], iter_cb=self.lbar.callback)
        elif fp['method']=='brute':
            #rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
            self.lmMin = minimize(costfx, lmParams, method=fp['method'], iter_cb=self.lbar.callback, Ns=20)
        else:
            self.lmMin = minimize(costfx, lmParams, method=fp['method'], tol=fp['tol'],  options=optkws, iter_cb=self.lbar.callback)
        if hasattr(self, 'lbar'):
            self.lbar.clear()

        self.param_report = fit_report(self.lmMin.params)
        return self.assess_fit(flat=flat, learn=learn)


    def set_global_options(self, learn=False):
        if learn:
            sim = self.simRL
        else:
            sim = self.sim
        bp = self.basinparams
        fp = self.fitparams
        xmin, xmax = format_basinhopping_bounds(sim.pvary, sim.nvary, self.kind, tb=fp['tb'])
        bounds = format_local_bounds(xmin, xmax)

        # define custom take_step and accept_test functions
        self.accept_step = GlobalBounds(xmin, xmax)
        self.custom_step = HopStep(sim.pvary, nlevels=sim.nvary, stepsize=bp['stepsize'])
        self.polish_args = {"method": bp['local_method'], 'bounds': bounds, 'tol': bp['polish_tol'], 'options': {'xtol': bp['polish_tol'], 'ftol': bp['polish_tol']}}

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
        if not hasattr(self, 'init_params'):
            init_params = theta.random_inits(pkeys, ninits=nsamples, kind=self.kind, as_list=True, method=self.basinparams['sample_method'])
            init_yhats = pd.DataFrame(np.vstack([self.sim.simulate_model(p) for p in init_params]))
            self.init_params = init_params
            self.init_yhats = init_yhats.copy()
        init_params = self.init_params
        init_yhats = self.init_yhats.copy()
        psets = self.filter_params(init_params, init_yhats)
        self.param_sets = psets


    def sample_rl_theta(self, fitDynamics=True):
        if fitDynamics:
            costfx = self.simRL.cost_fx_rl
        else:
            costfx = self.simRL.cost_fx
        nsamples = self.basinparams['nsamples']
        nkeep = self.basinparams['ninits']
        idxParams = pd.DataFrame([pd.Series(self.popt)]*nsamples)
        rlParams = theta.random_inits(['B', 'C', 'R'], ninits=nsamples, kind=self.kind, as_list=True, method=self.basinparams['sample_method'])
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

            psets = self.init_params
            yhatDF = self.init_yhats.copy()

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
            sim = self.simRL
        else:
            y = self.sim.y.flatten()
            wts = self.sim.wts.flatten()
            sim = self.sim

        # gen dict of lmfit optimized Parameters object
        popt = dict(self.lmMin.params.valuesdict())
        fmin = self.lmMin.chisqr
        nvary = len(self.lmMin.var_names)
        residuals = self.lmMin.residual
        yhat = (residuals / wts) + y
        if self.lmMin.method=='brute':
            success = not self.lmMin.aborted
        else:
            success = self.lmMin.success
        nfev = self.lmMin.nfev
        try: niter = self.lmMin.nit
        except Exception: niter = nfev
        # resContainer = self.lmMin
        #
        # # check if global optimization better
        # if hasattr(self, "global_popt"):
        #     checkGlobal = list(self.global_popt) == list(popt)
        #     if fmin > self.global_fmin and checkGlobal:
        #         popt = deepcopy(self.global_popt)
        #         fmin = self.global_fmin
        #         resContainer = self.global_results.lowest_optimization_result
        #         resContainer['nfev'] = self.global_results.nfev
        #         resContainer['nit'] = self.global_results.nit
        #
        # residualList = []
        # for i in range(5):
        #     # sim.update(inits=popt)
        #     yhat = sim.simulate_model(popt)
        #     residualList.append(wts * (yhat - y))
        #
        # residual = np.mean(residualList, axis=0)
        # yhat = (residual / wts) + y
        # success = resContainer.success
        # nfev = resContainer.nfev
        # niter = resContainer.nit

        # TODO: extract, calculate, and store std.errors of popt
        # presults is scipy.minimize object (see hop_around() for global_results)
        # presults = self.global_results.lowest_optimization_result
        # then take sqrt of the diag. of the hessian to get errors
        # poptErr = np.sqrt(np.diag(presults.hess_inv.todense()))

        if not self.learn:
            # un-vectorize all parameters except conditionals
            popt = theta.scalarize_params(popt, self.pcmap)
            if fp.nlevels>1:
                popt = theta.pvary_levels_to_array(popt, self.pcmap)

        finfo = pd.Series()
        # get model-predicted yhat vector
        fp['yhat'] = yhat
        finfo['idx'] = fp.idx
        finfo['pvary'] = '_'.join(list(fp.depends_on))
        finfo['cnvrg'] = success
        finfo['nfev'] = nfev
        finfo['niter'] = niter
        finfo['nvary'] = nvary
        finfo['chi'] = fmin
        finfo['ndata'] = y.size
        finfo['df'] = finfo.ndata - finfo.nvary
        finfo['rchi'] = finfo.chi / finfo.df
        finfo['logp'] = finfo.ndata * np.log(finfo.chi / finfo.ndata)
        finfo['AIC'] = finfo.logp + 2 * finfo.nvary
        finfo['BIC'] = finfo.logp + finfo.nvary * np.log(finfo.ndata)
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
        else:
            if 'Dropbox' in os.listdir(savedir):
                savedir = os.path.join(savedir, 'Dropbox')
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
            if bp['method']=='basin':
                niter = bp['nsuccess']
            else:
                niter = bp['maxiter']
            self.gbar = utils.GlobalCallback(n=niter, fmin=10000, method=bp['method'])
            self.callback = self.gbar.reset(get_call=True)
        if lBasin:
            fp = self.fitparams
            if hasattr(self, 'lbar'):
                self.lbar.clear()
            self.lbar = utils.GradientCallback(n=fp['maxfev'], fmin=10000)
            self.callback = self.lbar.reset(get_call=True)
