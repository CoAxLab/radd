#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd.models import Simulator
from radd.CORE import RADDCore
from radd import vis
from radd.tools import utils, analyze, messages
from radd.tools.analyze import pandaify_results, rangl_data


class Model(RADDCore):
    """ Main class for instantiating, fitting, and simulating models.
    Inherits from RADDCore parent class (see CORE module).
    ::Arguments::
        data (pandas DF):
            data frame with columns 'idx', 'rt', 'acc', 'ttype', 'response',
            <Condition Name> declared in depends_on values
        kind (str):
            declares model type ['dpm', 'irace', 'pro']
            append 'x' to front of model name to include a dynamic
            bias signal in model
        inits (dict):
            dictionary of parameters (v, a, tr, ssv, z) used to initialize model
        fit_on (str):
            set if model fits 'average', 'subjects', 'bootstrap' data
        depends_on (dict):
            set parameter dependencies on task conditions
            (ex. depends_on={'v': 'Condition'})
        weighted (bool):
            if True (default), perform fits using a weighted least-squares approach
        quantiles (array):
            set the RT quantiles used to fit model
    """

    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', depends_on={'all':'flat'}, weighted=True, ssd_method=None, learn=False, bwfactors=None, custompath=None, ssdelay=False, quantiles=np.arange(.1, 1.,.1)):

        # quantiles=np.arange(.1, 1.,.1)
        super(Model, self).__init__(data=data, inits=inits, fit_on=fit_on, depends_on=depends_on, kind=kind, quantiles=quantiles, weighted=weighted, ssd_method=ssd_method, learn=learn, bwfactors=bwfactors, custompath=custompath, ssdelay=ssdelay)


    def optimize(self, plotfits=True, saveplot=False, saveresults=True, custompath=None, progress=True):
        """ Method to be used for accessing fitting methods in Optimizer class
        see Optimizer method optimize()
        ::Arguments::
            plotfits (bool):
                if True (default), plot model predictions over observed data
            saveplot (bool):
                if True (default is False), save plots to "~/<self.model_id>/"
            saveresults (bool):
                if True (default), save fitdf, yhatdf, and txt logs to "~/<self.model_id>/"
            saveobserved (bool):
                if True (default is False), save observedDF to "~/<self.model_id>/"
            custompath (str):
                path starting from any subdirectory of "~/" (e.g., home).
                all saved output will write to "~/<custompath>/<self.model_id>/"
            progress (bool):
                track progress across ninits and basinhopping
        """
        self.toggle_pbars(progress=progress)
        self.custompath=custompath

        if self.fit_on == 'subjects':
            self.fitdf, self.poptdf, self.yhatdf = self.optimize_idx_params()

        else:
            self.set_fitparams(ix=ix, force='flat', nlevels=1)
            flat_popt = self.optimize_flat(self.param_sets)

            if not self.is_flat:
                self.set_fitparams(ix=ix, force='cond')
                self.optimize_conditional(flat_popt)

            if plotfits:
                self.plot_model_fits(save=saveplot)

        if progress and not self.is_nested:
            self.opt.gbar.clear()
            self.opt.ibar.clear()


    def optimize_flat(self, param_sets=None):
        """ optimizes flat model to data collapsing across all conditions
        ::Arguments::
            None
        ::Returns::
            None
        ::Attributes Created::
            yhat_flat (array): model-predicted data array (ndim = self.levels)
            finfo_flat (pd.Series): fit info (AIC, BIC, chi2, redchi, etc)
            popt_flat (dict): optimized parameters dictionary
        """
        if self.basinparams['method']=='basin':
            # Global Optimization w/ Basinhopping (+TNC)
            gpopt = self.opt.hop_around(param_sets)
        else:
            # Global Optimization w/ Diff. Evolution (+TNC)
            gpopt, gfmin = self.opt.optimize_global(self.inits)
        self.gpopt = deepcopy(gpopt)
        # Flat Simplex Optimization of Parameters at Global Minimum
        self.finfo, self.popt, self.yhat = self.opt.gradient_descent(p=gpopt)
        if self.is_flat:
            self.write_results()
        return self.popt


    def optimize_conditional(self, flatp, hop=False):
        """ optimizes full model to all conditions in data
        ::Arguments::
            None
        ::Returns::
            None
        ::Attributes Created::
            yhat (array): model-predicted data array (ndim = self.levels)
            finfo (pd.Series): fit info (AIC, BIC, chi2, redchi, etc)
            popt (dict): optimized parameters dictionary
            flat_popt (dict): deepcopy of popt
        """
        gpopt = self.__check_inits__(deepcopy(flatp))
        self.set_fitparams(force='cond', inits=gpopt)
        # Pretune Conditional Parameters
        if hop:
            gpopt, fmin = self.opt.optimize_global(gpopt)
            self.opt.update(inits=gpopt)
        # Final Simplex Optimization
        self.finfo, self.popt, self.yhat = self.opt.gradient_descent(p=gpopt)
        # self.write_results()


    def nested_optimize(self, models=[], flatp=None, saveplot=True, plotfits=True, custompath=None, progress=False, saveresults=True, saveobserved=False):
        """ optimize a series of models using same init parameters where the i'th model
            has depends_on = {<models[i]> : <cond>}.
            NOTE: only for models with fit_on='average'
        ::Arguments::
            models (list):
                list of depends_on dictionaries to fit using a single set of init parameters (self.flat_popt)
            plotfits (bool):
                if True (default), plot model predictions over observed data
            saveplot (bool):
                if True (default), save plots to model.handler.results_dir
            custompath (str):
                path starting from any subdirectory of "~/" (e.g., home).
                all saved output will write to "~/<custompath>/<self.model_id>/"
            progress (bool):
                track progress across model fits, ninits, and basinhopping
        """
        self.is_nested = True
        self.custompath = custompath
        if flatp is None:
            models = [{'all': 'flat'}] + models
            self.param_sets = self.opt.sample_param_sets()

        pnames = self.toggle_pbars(progress=progress, models=models)
        # loop over depends_on dictionaries and optimize cond models
        for i, depends_on in enumerate(models):
            self.set_fitparams(depends_on=depends_on)
            if progress:
                self.mbar.update(value=i, status=pnames[i])
            if flatp is None:
                self.param_sets = self.opt.sample_param_sets()
                flatp = self.optimize_flat(self.param_sets)
                continue
            self.optimize_conditional(flatp)
            if plotfits:
                self.plot_model_fits(save=saveplot)
            if saveresults:
                self.handler.save_results(saveobserved=saveobserved)
        if progress:
            self.mbar.clear()
            self.opt.gbar.clear()
            self.opt.ibar.clear()


    def log_fit_info(self, finfo=None, popt=None, yhat=None):
        """ write meta-information about latest fit
        to logfile (.txt) in working directory
        """
        self.opt.log_fit_info(finfo, popt, yhat)


    def write_results(self, finfo=None, popt=None, yhat=None):
        """ logs fit info to txtfile, fills yhatdf and fitdf
        """

        finfo, popt, yhat = self.set_results(finfo, popt, yhat)
        self.log_fit_info(finfo, popt, yhat)

        self.yhatdf = self.handler.fill_yhatdf(yhat=yhat, fitparams=self.fitparams)
        self.fitdf = self.handler.fill_fitdf(finfo=finfo, fitparams=self.fitparams)
        self.poptdf = self.handler.fill_poptdf(popt=popt, fitparams=self.fitparams)


    def plot_model_fits(self, y=None, yhat=None, kde=True, err=None, save=False, bw='scott', savestr=None, same_axis=True, clrs=None, lbls=None, cumulative=True, simdf=None, suppressLegend=False, simData=None, condData=None, shade=True, plot_error_rts=True, figure=None):
        """ wrapper for radd.tools.vis.plot_model_fits
        """
        data = self.handler.data.copy()

        if y is None:
            y = self.fitparams.y

        if yhat is None:
            try:
                yhat = self.yhat
            except AttributeError:
                print("No Model Predictions to Plot (need yhat argument)")
                yhat = analyze.rangl_data(simdf, quantiles=self.quantiles)
        if save:
            if savestr is None:
                savestr = self.fitparams.model_id
            if self.fitparams['fit_on']=='subjects':
                savestr = savestr + str(self.fitparams['idx'])

        if err is None:
            err = self.handler.observed_err

        if lbls is None and self.fitparams.nlevels>1:
            from itertools import product
            levels = [self.clmap[cond] for cond in self.conds]
            level_data = list(product(*levels))
            lbls = ['_'.join([str(lvl) for lvl in lvls]) for lvls in level_data]

        if self.ssd_method == 'central':
            ssd = self.ssdDF.groupby(self.conds).mean()[0].values
            ssderr = self.ssdDF.groupby(self.conds).sem()[0].values
        else:
            ssd = self.fitparams.ssd_info[0]
            ssderr = None

        fig = vis.plot_model_fits(y, yhat, err=err, quantiles=self.quantiles, ssd=ssd, ssderr=ssderr, bw=bw, same_axis=same_axis, clrs=clrs, lbls=lbls, cumulative=cumulative, save=save, savestr=savestr, suppressLegend=suppressLegend, shade=shade, plot_error_rts=plot_error_rts, figure=figure)


    def simulate(self, p=None, analyze=True):
        """ simulate yhat vector using
        :: Arguments ::
            p (dict):
                parameters dictionary
            analyze (bool):
                if True (default) returns yhat vector. else, returns decision traces
        :: Returns ::
            out (array):
                1d array if analyze is True, else ndarray of decision traces
        """
        if p is None:
            try:
                p = self.popt
            except Exception:
                p = self.__get_default_inits()
        p = deepcopy(p)
        yhat = self.sim.simulate_model(p, analyze=analyze)
        return yhat




    def optimize_idx_params(self, param_sets=None, force=None, save=True):
        """ optimize parameters for individual subjects, store results
        :: Arguments ::
            param_sets (list):
                parameters dictionary
            force (str):
                if 'cond' forces fits to conditional data, if 'flat' forces flat data
            save (bool):
                save output dataframes if True
        :: Returns ::
            fitdf (DataFrame): fit statistics
            poptdf (DataFrame): optimized parameters
            yhatdf (DataFrame): model predictions
        """

        self.iohandler = ModelIO(fitparams=self.fitparams, mname=self.model_id)

        # subject lists
        poptAll, yhatAll, finfoAll = [], [], []
        if self.opt.param_sets is None and param_sets is None:
            self.opt.sample_param_sets()

        for ix, idx in enumerate(self.idx):

            if hasattr(self, 'idxbar'):
                self.idxbar.update(value=ix, status=ix)
            # set subject data & wts
            self.set_fitparams(ix=ix, force=force)
            nl = self.fitparams.nlevels
            if nl==1:
                params = self.opt.filter_params()
            else:
                print("must provide list of flat popt dicts for conditional models")
                exit(0)

            # fit result lists for each param set
            poptList, yhatList, finfoList, fminList = [], [], [], []
            for i, pdict in enumerate(params):

                # reset global pbar
                if self.opt.progress:
                    self.opt.make_progress_bars(inits=False, basin=True)

                # set fixed parameters
                self.popt = deepcopy(pdict)
                self.opt.update(inits=pdict)
                # global optimization of conditional parameters
                gpopt, gfmin = self.opt.optimize_global(pdict)
                # run gradient descent on globally optimized params
                finfo, popt, yhat = self.opt.gradient_descent(gpopt)

                # store results
                finfo['pset'] = i
                finfoList.append(finfo)
                poptList.append(deepcopy(popt))
                yhatList.append(pd.DataFrame(yhat.reshape(nl, -1)))
                fminList.append(finfo.chi)
                #clear pbars
                self.opt.gbar.clear()

            # make results dataframes for subject ix
            fitdf = pd.concat(finfoList, axis=1).T
            poptdf = pd.DataFrame.from_dict(poptList)
            poptdf['fmin'] = fminList
            yhatdf = pd.concat(yhatList)

            if force=='cond':
                yhatdf = self.iohandler.format_yhatdf(yhatdf, param_sets)

            # store subject results dataframes in lists
            finfoAll.append(fitdf); poptAll.append(poptdf); yhatAll.append(yhatdf)
            # concatenate all subjects together into single fitdf, poptdf, & yhatdf
            fitdf, poptdf, yhatdf = [pd.concat(dfList) for dfList in [finfoAll, poptAll, yhatAll]]
            self.iohandler.save_model_results(fitdf, poptdf, yhatdf, write=save)
            fitdf, poptdf, yhatdf = self.iohandler.read_model_results()

        return fitdf, poptdf, yhatdf




class ModelIO(object):
    """ generates model read and write paths for
    handling I/O of model DataFrames, figures, etc
    """

    def __init__(self, fitparams, mname='radd_model', savestr=None, idxdir='subj_fits'):

        self.fitparams = fitparams
        self.idx = self.fitparams.idx
        self.nidx = len(self.idx)
        self.mname = mname

        if savestr==None:
            savestr = mname
        self.savestr = savestr
        if idxdir is not None:
            idxdir = os.path.join(os.path.expanduser('~'), idxdir)
        self.idxdir = idxdir
        self.mdir = os.path.join(self.idxdir, mname)

        makePath = lambda dfname: os.path.join(self.mdir, '_'.join([self.mname, dfname])+'.csv')
        dfnames = ['fitdf', 'poptdf', 'yhatdf']
        self.fitPath, self.poptPath, self.yhatPath = [makePath(dfname) for dfname in dfnames]
        self.paths = [self.idxdir, self.mdir]


    def save_model_results(self, fitdf=None, poptdf=None, yhatdf=None, write=True):
        """ save model fitdf, poptdf, and yhatdf dataframes
        """

        if fitdf is not None:
            self.fitdf = fitdf
        if poptdf is not None:
            try:
                self.poptdf = self.format_poptdf(poptdf)
            except Exception:
                self.poptdf = poptdf
        if yhatdf is not None:
            try:
                self.yhatdf = self.format_yhatdf(yhatdf)
            except Exception:
                self.yhatdf = yhatdf
        if write:
            for pth in self.paths:
                if not os.path.isdir(pth):
                    os.mkdir(pth)
            self.fitdf.to_csv(self.fitPath, index=False)
            self.poptdf.to_csv(self.poptPath, index=False)
            self.yhatdf.to_csv(self.yhatPath, index=False)


    def read_model_results(self):
        """ save model fitdf, poptdf, and yhatdf dataframes
        """
        self.fitdf = pd.read_csv(self.fitPath)
        poptdf = pd.read_csv(self.poptPath)
        yhatdf = pd.read_csv(self.yhatPath)
        try:
            self.poptdf = self.format_poptdf(poptdf)
        except Exception:
            self.poptdf = poptdf
        try:
            self.yhatdf = self.format_yhatdf(yhatdf)
        except Exception:
            self.yhatdf = yhatdf
        return self.fitdf, self.poptdf, self.yhatdf


    def save_fit_figure(self, f, savestr='avgYhat'):
        """ save model fits figure
        """
        savepath = os.path.join(self.mdir, '_'.join([self.mname, savestr]) + '.png')
        plt.tight_layout()
        f.savefig(savepath, dpi=600)


    def format_yhatdf(self, yhatdf):
        yhatdf = yhatdf.copy()
        nidx = self.nidx
        nl = self.fitparams.nlevels
        nparams = int(yhatdf.shape[0] / (nidx * nl))
        for i, (k,v) in enumerate(self.fitparams.clmap.items()):
            yhatdf.insert(i, k, list(v) * nidx * nparams)
        idxVals = np.sort(np.hstack([self.idx]*(nparams*nl)))
        yhatdf.insert(0, 'idx', idxVals)
        pN = np.tile(np.sort(np.tile(np.arange(0,nparams), nl)), nidx)
        try:
            yhatdf.insert(1, 'pset', pN)
        except Exception:
            yhatdf['pset'] = pN
        yhatdf = yhatdf.reset_index(drop=True)
        return yhatdf


    def format_poptdf(self, poptdf):
        nidx = self.nidx
        nl = self.fitparams.nlevels
        nparams = int(poptdf.shape[0] / nidx)
        idxVals = np.sort(np.hstack([self.idx]*nparams))
        poptdf.insert(0, 'idx', idxVals)
        pN = np.tile(np.sort(np.arange(1,nparams+1)), nidx)
        poptdf.insert(1, 'pset', pN)
        poptdf = poptdf.reset_index(drop=True)
        return poptdf


    # def plot_idx_fits(self, clrs=['#6C7A89'], figure=None, save=False, savestr='avgYhat'):
    #
    #     if not hasattr(self, 'yhatdf'):
    #         print('Need to Save Result DataFrames\n(hint: save_model_results())')
    #         return
    #     m = self.model
    #     poptdf = self.poptdf
    #     # get avg. observed data vectors
    #     idx_data = np.vstack(m.observed)
    #     nl = self.fitparams.nlevels
    #     y = idx_data.mean(0).reshape(nl,-1)
    #     yErr = sem(idx_data, axis=0).reshape(nl,-1) * 1.96
    #
    #     pBest = poptdf[poptdf.fmin.isin(poptdf.groupby('idx').fmin.min().values)]
    #     yhatBest = self.yhatdf.iloc[pBest.index.values, 3:]
    #     yhat = yhatBest.mean().values
    #     yhatErr = yhatBest.sem().values*1.96
    #
    #     # get param dict and pass to model
    #     pnames = list(self.model.inits)
    #     mu_popt = pBest.loc[:, pnames].mean().to_dict()
    #     self.model.popt = deepcopy(mu_popt)
    #
    #     # plot avg subject fit
    #     self.model.plot_model_fits(y=y, yhat=yhat, clrs=clrs, err=yErr, figure=figure)
    #
    #     if save:
    #         self.save_fit_figure(plt.gcf(), savestr=savestr)
