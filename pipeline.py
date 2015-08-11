#!/usr/local/bin/env python
from __future__ import division
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from radd import build, neuro, vis
from itertools import product


def run_pipeline(data=pd.DataFrame, kinds='pro', depends_keys=[], condition='pGo', dynamics=[], initials=None, fit_on='average', rootdir='~/Dropbox/', plot_bold=True, plot_behavior=True, ntrials=10000, tol=1.e-20, maxfev=5000, niter=500, results='both'):

      """ builds, fits, simulates, plots a sequence of different models
      in which different parameters to vary across a condition

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
                  init parameters are drawn from toolbox.theta.get_default_inits()
            fit_on (str):
                  Fit models to 'average', 'subjects', or 'bootstrap' data
            rootdir (str):
                  rootdir for model save paths
            results (str):
                  plot model predictions for 'behavior', 'bold', or 'both'

      ::Returns::
            None
      """
      nmodels = len(depends_keys)
      # make list of depends_on dictionaries
      deplist = make_depends_on_list(depends_keys, condition)
      # if no inits then generarte permutations of model attr
      if initials is None:
            initials=[None]*nmodels
            minfo = model_permutations(deplist=deplist, kinds=kinds, dynamics=dynamics)
            kinds, depends_on, dynamics = minfo['kinds'], minfo['depends_on'], minfo['dynamics']

      if not isinstance(dynamics, list):
            dynamics=[dynamics]*nmodels
      if not isinstance(kinds, list):
            kinds=[kinds]*nmodels

      for i in range(nmodels):
            model = build.Model(data=data, kind=kinds[i], inits=initials[i], depends_on=depends_on[i], dynamic=dynamics[i], fit_on=fit_on, verbose=False)
            # create model id
            model_id = id_model(model)
            # make/change to model save dir
            go_to(kind=model.kind, model_id=model_id, rootdir=rootdir)
            # fit model to data
            model.optimize(tol=tol, maxfev=maxfev, ntrials=ntrials, niter=niter)

            plot_results(model, results=results, model_id=model_id)


def model_permutations(kinds=[], deplist=[], dynamics=[]):

      if not isinstance(dynamics, list):
            dynamics=[dynamics]
      if not isinstance(kinds, list):
            kinds=[kinds]
      minfo = {}
      # generate permutations of kinds, depends_on, and dynamics
      infop = [product([k], deplist) if not 'x' in k else product([k], deplist, dynamics) for k in kinds]
      # convert product tuples to lists then combine permutations of dynamic models and linear models
      allmodels = [list(tup)+['Null'] for tup in list(infop[0])]
      if len(infop)>1:
            allmodels = allmodels + [list(tup) for tup in list(infop[1])]
      minfo['kinds'] = [mi[0] for mi in allmodels]
      minfo['depends_on'] = [mi[1] for mi in allmodels]
      minfo['dynamics'] = [mi[2] for mi in allmodels]
      return minfo


def id_model(model):
      #### AS LAMBDA ###
      #id_model = lambda mod: [['_'.join([dirname, mod.dynamic]).upper() if 'x' in mod.kind else dirname.upper() for dirname in ['_'.join([k for k in mod.depends_on.keys()])]][0]]
      model_id = ['_'.join([dirname, model.dynamic]).upper() if 'x' in model.kind else dirname.upper() for dirname in ['_'.join([k for k in model.depends_on.keys()])]][0]
      return model_id


def go_to(kind='pro', model_id='TEST', rootdir='~/Dropbox/'):
      """ make/chdir to model_path
      """

      if rootdir[0]=="~":
            rootdir = '/'.join(os.sys.path[-1].split('/')[:3])+rootdir[1:]

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

      if model_id is None:
            model_id = id_model(model)

      if 'pro' not in kind:
            yhat = model.fits.reshape(model.ncond, int(len(model.avg_y)/model.ncond))
            for i, yh in enumerate(yhat):
                  vis.plot_fits(model.avg_y, yh, kind='radd', save=True, savestr='_'.join([model_id, model.labels[i]]))
      else:
            if results in ['behavior', 'both']:
                  vis.plot_fits(model.avg_y, model.fits, kind='pro', save=True, savestr=model_id)
            if results in ['bold', 'both']:
                  bold=neuro.BOLD(model)
                  bold.simulate_bold(save=True, savestr=model_id)
                  bold.plot_means(save=True, savestr=model_id)
                  bold.plot_traces(save=True, savestr=model_id)


def make_depends_on_list(depends_keys=[], condition=None):
      depends_on_list = []
      for dkey in depends_keys:
            if hasattr(dkey, '__iter__'):
                  depends_on_list.append({k: condition for k in dkey})
            else:
                  depends_on_list.append({dkey: condition})
      return depends_on_list
