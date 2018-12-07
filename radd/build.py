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
import multiprocessing as mp
import matplotlib.pyplot as plt
from IPython.display import clear_output


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

    def __init__(self, data=pd.DataFrame, kind='xdpm', inits=None, fit_on='average', depends_on={'all':'flat'}, weighted=True, ssd_method=None, learn=False, bwfactors=None, custompath=None, presample=False, ssdelay=False, gbase=False, quantiles=np.arange(.1, 1.,.1), ksfit=False):

        super(Model, self).__init__(data=data, inits=inits, fit_on=fit_on, depends_on=depends_on, kind=kind, quantiles=quantiles, weighted=weighted, ssd_method=ssd_method, learn=learn, bwfactors=bwfactors, custompath=custompath, presample=presample, ssdelay=ssdelay, gbase=gbase, ksfit=ksfit)

        groups = self.handler.groups
        bwcol = None
        if self.bwfactors is not None:
            groups += [self.bwfactors]
            self.bwcol = [df[self.bwfactors].unique()[0] for _, df in self.data.groupby('idx')]


    def optimize(self, plotfits=False, saveplot=False, saveresults=False, custompath=None, progress=True, get_results=False, powell=True, hop=False):
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
        #self.toggle_pbars(progress=progress)
        self.custompath=custompath

        if self.fit_on == 'subjects':
            finfo, popt, yhat  = self.optimize_idx_params(self.idx)

            if plotfits:
                self.plot_model_idx_fits(save=saveplot)

        else:
            self.set_fitparams(force='flat', nlevels=1)
            finfo, popt, yhat = self.optimize_flat(get_results=True)

            if not self.is_flat:
                finfo, popt, yhat = self.optimize_conditional(popt, get_results=True, powell=powell, hop=hop)

            if plotfits:
                self.plot_model_fits(save=saveplot)

        if progress:
            if self.fit_on=='subjects':
                self.idxbar.clear()
            self.opt.ibar.clear()
            self.opt.gbar.clear()

        try:
            self.write_results(finfo=finfo, popt=popt, yhat=yhat)
        except Exception:
            pass

        if get_results:
            return finfo, popt, yhat


    def fit_multi_idx(self, nproc=1, progress=True):
        nruns = int(len(self.idx) / nproc)
        fits, popts, yhats = [], [], []
        self.toggle_pbars(progress=progress)
        self.idxbar.update(value=0, status=0)
        self.opt.make_progress_bars(inits=False, basin=True)

        # self.iohandler = ModelIO(fitparams=self.fitparams, mname=self.model_id)
        self.yhatdf = self.observedDF[self.observedDF.idx.isin(self.idx)].copy()
        datcols = self.yhatdf.loc[:, 'acc':].columns.tolist()

        def fit_idx_i(pos, idx, output):
            self.set_fitparams(ix=self.idx.index(idx), force='flat', nlevels=1)
            finfo, popt, yhat = self.optimize_flat(get_results=True)
            if not self.is_flat:
                finfo, popt, yhat = self.optimize_conditional(popt, get_results=True)
            output.put((pos, {'finfo': finfo, 'popt': popt, 'yhat': yhat}))

        for i in range(nruns):
            # initiate process workers
            output = mp.Queue()
            ix0 = i * nproc
            ix1 = ix0 + nproc
            idxlist = self.idx[ix0:ix1]
            processes = [mp.Process(target=fit_idx_i, args=(pos, idx, output)) for pos, idx in enumerate(idxlist)]
            for p in processes: p.start()
            for p in processes: p.join()

            results = [output.get() for p in processes]; results.sort()
            results = np.hstack([r[1] for r in results]).tolist()
            fits.append(pd.concat([res['finfo'] for res in results], axis=1).T)
            popts.append(pd.DataFrame([res['popt'] for res in results]))
            yhat = np.array([res['yhat'] for res in results])
            self.yhatdf.loc[ix0:ix1, datcols] = yhat.reshape(nproc, -1)
            self.idxbar.update(value=ix1, status=ix1)

        self.fitdf = pd.concat(fits)
        self.poptdf = pd.concat(popts)
        self.poptdf.insert(0, 'idx', self.idx)
        if self.bwfactors is not None:
            self.poptdf.insert(1, self.bwfactors, self.bwcol)
            self.fitdf.insert(1, self.bwfactors, self.bwcol)
        self.idxbar.clear()
        return self.fitdf, self.poptdf, self.yhatdf



    def optimize_flat(self, param_sets=None, get_results=False):
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
            gpopt = self.opt.hop_around(param_sets, learn=self.learn)
        else:
            # Global Optimization w/ Diff. Evolution (+TNC)
            gpopt, gfmin = self.opt.optimize_global(self.inits, learn=self.learn, resetProgress=False)

        # Flat Simplex Optimization of Parameters at Global Minimum
        self.set_fitparams(inits=gpopt)
        self.finfo, gpopt, self.yhat = self.opt.gradient_descent(p=gpopt, learn=self.learn)
        self.gpopt = deepcopy(gpopt)
        self.popt = deepcopy(gpopt)
        # if self.is_flat:
        #     self.write_results()
        if get_results:
            return self.finfo, self.popt, self.yhat
        if self.opt.progress:
            self.opt.ibar.clear()
            self.opt.gbar.clear()
            clear_output()
        return self.popt


    def optimize_conditional(self, flatp, hop=False, powell=True, get_results=False):
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
            gpopt, fmin = self.opt.optimize_global(gpopt, learn=self.learn)
            self.opt.update(inits=gpopt)

        # Final Simplex Optimization(s)
        self.set_fitparams(method='nelder', maxfev=300)
        finfo, popt, yhat = self.opt.gradient_descent(p=gpopt, learn=self.learn)

        if powell:
            self.set_fitparams(method='powell', inits=popt, force='cond', maxfev=1000)

            self.opt.update(inits=popt)
            finfo, popt, yhat = self.opt.gradient_descent(p=popt, learn=self.learn)

        self.finfo, self.popt, self.yhat = finfo, popt, yhat

        if get_results:
            return self.finfo, self.popt, self.yhat


    def nested_optimize(self, depends=[], flatp=None, saveplot=True, plotfits=False, custompath=None, progress=True, saveresults=True, saveobserved=False):
        """ optimize a series of models using same init parameters where the i'th model
            has depends_on = {<depends[i]> : <cond>}.
            NOTE: only for models with fit_on='average'
        ::Arguments::
            depends (list):
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

        if flatp is None:
            self.set_basinparams(method='basin', nsamples=1000)
            self.set_testing_params()
            self.set_fitparams(force='flat')
            flatp = deepcopy(self.optimize_flat())

        self.fitdf, self.poptdf, self.yhatdf = nested_optimize(depends, self.data, kind=self.kind, flatp=flatp, saveplot=saveplot, plotfits=plotfits, custompath=custompath, progress=progress, saveresults=saveresults)

        return self.fitdf, self.poptdf, self.yhatdf


    def log_fit_info(self, finfo=None, popt=None, yhat=None):
        """ write meta-information about latest fit
        to logfile (.txt) in working directory
        """
        self.opt.log_fit_info(finfo, popt, yhat)


    def write_results(self, finfo=None, popt=None, yhat=None):
        """ logs fit info to txtfile, fills yhatdf and fitdf
        """

        from radd import theta

        finfo, popt, yhat = self.set_results(finfo, popt, yhat)
        self.log_fit_info(finfo, popt, yhat)

        self.yhatdf = self.handler.fill_yhatdf(yhat=yhat, fitparams=self.fitparams)
        self.fitdf = self.handler.fill_fitdf(finfo=finfo, fitparams=self.fitparams)

        # pcopy = theta.scalarize_params(deepcopy(popt))
        pcopy = deepcopy(popt)
        self.poptdf = self.handler.fill_poptdf(popt=pcopy, fitparams=self.fitparams)


    def plot_model_fits(self, y=None, yhat=None, kde=True, err=None, save=False, bw='scott', savestr=None, same_axis=True, clrs=None, lbls=None, cumulative=True, simdf=None, suppressLegend=False, simData=None, condData=None, shade=True, plot_error_rts=True, figure=None, reorder=None):
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

        if reorder is not None:
            y = y.reshape(self.fitparams.nlevels, -1)
            yhat = yhat.reshape(self.fitparams.nlevels, -1)
            y = np.array([y[i] for i in reorder])
            yhat = np.array([yhat[i] for i in reorder])
        if save:
            if savestr is None:
                savestr = self.fitparams.model_id
            if self.fitparams['fit_on']=='subjects':
                savestr = savestr + str(self.fitparams['idx'])

        if err is None:
            err = self.observedErr.loc[:, 'acc':].values.squeeze()

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


    def optimize_idx_params(self, idxlist=None, pos=0, output=None, progress=True):
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
        yhatdf = self.observedDF[self.observedDF.idx.isin(self.idx)].copy()
        datcols = yhatdf.loc[:, 'acc':].columns.tolist()
        # fit result lists for each param set
        finfoList, poptList, yhatList = [], [], []
        for idx in self.idx:

            self.toggle_pbars(progress=progress)

            ix = self.idx.index(idx)
            if hasattr(self, 'idxbar'):
                self.idxbar.update(value=ix, status=ix+1)

            # set subject data & wts
            self.set_fitparams(ix=ix, force='flat', nlevels=1)
            # optimize constants (flat model)
            finfo, popt, yhat = self.optimize_flat(get_results=True)
            flatPopt = {p: popt[p] for p in list(self.sim.inits)}
            # optimize conditional parameters
            if not self.is_flat:
                self.set_fitparams(force='cond')
                finfo, popt, yhat = self.optimize_conditional(popt, get_results=True)
                yhat = yhat.reshape(self.nlevels, -1)
                for p in list(self.depends_on):
                    popt[p] = flatPopt[p]

            # store results
            finfoList.append(finfo)
            poptList.append(deepcopy(popt))
            yhatdf.loc[yhatdf.idx==idx, datcols] = yhat

            self.opt.ibar.clear()
            self.opt.gbar.clear()
            clear_output()

        # concatenate all subjects together into single fitdf, poptdf, & yhatdf
        fitdf = pd.concat(finfoList, axis=1).T
        poptdf = pd.DataFrame.from_dict(poptList)
        poptdf.insert(0, 'idx', self.idx)
        if self.bwfactors is not None:
            poptdf.insert(1, self.bwfactors, self.bwcol)
            fitdf.insert(1, self.bwfactors, self.bwcol)

        self.iohandler.save_model_results(fitdf, poptdf, yhatdf, write=True)
        self.fitdf, self.poptdf, self.yhatdf = self.iohandler.read_model_results()
        return self.fitdf, self.poptdf, self.yhatdf


def nested_optimize(depends, data, kind='xdpm', flatp=None, saveplot=True, plotfits=False, custompath=None, progress=True, saveresults=False, saveobserved=False):
    """ optimize a series of models using same init parameters where the i'th model
        has depends_on = {<depends[i]> : <cond>}.
        NOTE: only for models with fit_on='average'
    ::Arguments::
        depends (list):
            list of depends_on dictionaries to fit using a single set of init parameters (m.flat_popt)
        data (DataFrame):
            input dataframe to fit nested models to
        plotfits (bool):
            if True (default), plot model predictions over observed data
        saveplot (bool):
            if True (default), save plots to model.handler.results_dir
        custompath (str):
            path starting from any subdirectory of "~/" (e.g., home).
            all saved output will write to "~/<custompath>/<m.model_id>/"
        progress (bool):
            track progress across model fits, ninits, and basinhopping
    """

    fits, popts, yhats = [], [], []

    for i, depends_on in enumerate(depends):
        m = Model(data=data, kind=kind, depends_on=depends_on, quantiles=np.arange(.1, 1.,.1), ssd_method='all')
        m.set_basinparams(method='basin', nsamples=1000)
        m.custompath = custompath
        pnames = m.toggle_pbars(progress=progress, models=depends)

        if progress:
            m.mbar.update(value=i, status=pnames[i])

        m.set_fitparams(force='cond')
        flatp_ = deepcopy(flatp)
        finfo, popt, yhat = m.optimize_conditional(flatp_, hop=True, get_results=True)

        m.write_results(finfo, popt, yhat)
        m.fitdf.insert(0, 'modelID', '_'.join(list(m.depends_on)))
        m.poptdf.insert(0, 'modelID', '_'.join(list(m.depends_on)))
        m.yhatdf.insert(0, 'modelID', '_'.join(list(m.depends_on)))
        fits.append(m.fitdf); popts.append(m.poptdf); yhats.append(m.yhatdf)

        if plotfits:
            m.plot_model_fits(save=saveplot)

        if progress:
            m.mbar.clear()
            m.opt.gbar.clear()
            m.opt.ibar.clear()

    return [pd.concat(outdata).reset_index(drop=True) for outdata in [fits, popts, yhats]]


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
            self.poptdf = poptdf
        if yhatdf is not None:
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
        self.poptdf = pd.read_csv(self.poptPath)
        self.yhatdf = pd.read_csv(self.yhatPath)
        return self.fitdf, self.poptdf, self.yhatdf


    def save_fit_figure(self, f, savestr='avgYhat'):
        """ save model fits figure
        """
        savepath = os.path.join(self.mdir, '_'.join([self.mname, savestr]) + '.png')
        plt.tight_layout()
        f.savefig(savepath, dpi=600)
