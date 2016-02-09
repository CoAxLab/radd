#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from itertools import product


def run_pipeline(data=pd.DataFrame, kinds='pro', depends_keys=[], condition='pGo', dynamics=[], initials=None, fit_on='average', weighted=True, rootdir='~/Dropbox/', ntrials=10000, tol=1.e-20, maxfev=5000, niter=500, results='both'):
    """ builds, fits, simulates, plots a sequence of different models

    ::Arguments::
          data (pd.DataFrame):
                Data for fitting. Must be formatted for the specific kind
                of models to be fit (i.e., reactive, proactive)
          kinds (list or str):
                model kinds ('radd', 'irm', 'pro', 'xradd', ..etc)
          depends_keys (list):
                list of parameters to use as keys in depends_on dict
                elements can be lists (i.e. ['v', 'tr']) or str 'v'
          condition (str):
                column in data that defines the different conditions in the model
          dynamics (list or str):
                Temporal dynamics of Go process ('exp': exponential, 'hyp': hyperbolic).
                If type str all models will be assigned the same dynamic.
          initials (list or None):
                list of init parameters to use for each model. If None,
                init parameters are drawn from tools.theta.get_default_inits()
          fit_on (str):
                Fit models to 'average', 'subjects', or 'bootstrap' data
          rootdir (str):
                rootdir for model save paths
          results (str):
                plot model predictions for 'behavior', 'bold', or 'both'

    ::Returns::
          mlist (list):
                list of all model objects generated
    """
    nmodels = len(depends_keys)
    # make list of depends_on dictionaries
    depends_on = make_depends_on_list(depends_keys, condition)
    # if no inits then generarte permutations of model attr
    if initials is None:
        initials = [None] * nmodels
        minfo = model_permutations(deplist=depends_on, kinds=kinds, dynamics=dynamics)
        kinds, depends_on, dynamics = minfo['kinds'], minfo['depends_on'], minfo['dynamics']
    elif isinstance(initials, dict):
        initials = [initials] * nmodels

    if not isinstance(dynamics, list):
        dynamics = [dynamics] * nmodels
    if not isinstance(kinds, list):
        kinds = [kinds] * nmodels

    mlist = []
    for i in range(nmodels):
        kind, dep, dyn = kinds[i], depends_on[i], dynamics[i]
        # get id
        model_id = id_model(kind, dep, dyn)
        # make/change to model save dir
        go_to(kind=kind, model_id=model_id, rootdir=rootdir)
        # build model
        fitted_model = model_and_analyze(data=data, kind=kind, inits=initials[i], depends_on=dep, dynamic=dyn, fit_on=fit_on, verbose=False, weighted=weighted, model_id=model_id, ntrials=10000, tol=1.e-20, maxfev=5000, niter=500, results='both')
        # store model object
        mlist.append(fitted_model)

    return mlist


def model_and_analyze(data=pd.DataFrame, kind=None, inits={}, depends_on={}, dynamic=None, fit_on='average', verbose=False, weighted=True, model_id=None, ntrials=10000, tol=1.e-20, maxfev=5000, niter=500, results='both'):

    from radd import build

    # build model
    model = build.Model(data=data, kind=kind, inits=inits, depends_on=depends_on, dynamic=dynamic, fit_on=fit_on, verbose=False, weighted=weighted)

    # fit model to data
    model.optimize(tol=tol, maxfev=maxfev, ntrials=ntrials, niter=niter)
    # return model
    plot_results(model, results=results, model_id=model_id)
    return model


def model_permutations(kinds=[], deplist=[], dynamics=[]):

    if not isinstance(dynamics, list):
        dynamics = [dynamics]
    if not isinstance(kinds, list):
        kinds = [kinds]
    minfo = {}
    print kinds
    # generate permutations of kinds, depends_on, and dynamics
    infop = [product([k], deplist) if not 'x' in k else product([k], deplist, dynamics) for k in kinds]

    # convert product tuples to lists then combine permutations of dynamic
    # models and linear models
    allmodels = [list(tup) for tup in list(infop[0])] + [list(tup) + ['Null'] for tup in list(infop[1])]

    minfo['kinds'] = [mi[0] for mi in allmodels]
    minfo['depends_on'] = [mi[1] for mi in allmodels]
    minfo['dynamics'] = [mi[2] for mi in allmodels]
    return minfo


def id_model(kind, depends_on, dynamic):
    #### AS LAMBDA ###
    model_id = ['_'.join([dkeys, dynamic]).upper() if 'x' in kind else dkeys.upper() for dkeys in ['_'.join([k for k in depends_on.keys()])]][0]
    return model_id


def go_to(kind='pro', model_id='TEST', rootdir='~/Dropbox/'):
    """ make/chdir to model_path
    """

    if rootdir[0] == "~":
        rootdir = '/'.join(os.sys.path[-1].split('/')[:3]) + rootdir[1:]

    kind_path = '/'.join([rootdir, kind.upper()])
    if not os.path.isdir(kind_path):
        os.mkdir(kind_path)
    model_path = '/'.join([kind_path, model_id])
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    os.chdir(model_path)


def plot_results(model, results='both', model_id=None):
    """ plot model predictions
    ::Arguments::
          model (RADD model)
          results (str): 'behavior', 'bold', or 'both'
    """

    from radd import neuro, vis

    if model_id is None:
        model_id = id_model(model)

    if 'pro' not in model.kind:
        yhat = model.fits.reshape(model.ncond, int(len(model.fits) / model.ncond))
        for i, yh in enumerate(yhat):
            vis.plot_fits(model.y[i], yh, kind='radd', save=True, savestr='_'.join([model_id, model.labels[i]]))
    else:
        if results in ['behavior', 'both']:
            vis.plot_fits(model.y, model.fits, kind='pro', save=True, savestr=model_id)
        if results in ['bold', 'both']:
            bold = neuro.BOLD(model)
            bold.simulate_bold(save=True, savestr=model_id)
            bold.plot_means(save=True)
            bold.plot_traces(save=True)


def make_depends_on_list(depends_keys=[], condition=None):
    depends_on_list = []
    for dkey in depends_keys:
        if hasattr(dkey, '__iter__'):
            depends_on_list.append({k: condition for k in dkey})
        else:
            depends_on_list.append({dkey: condition})
    return depends_on_list
