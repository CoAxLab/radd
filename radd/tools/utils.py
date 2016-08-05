#!/usr/local/bin/env python
from __future__ import division
import sys
from future.utils import listvalues
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from radd.tools import colors, messages, analyze
from radd import theta
from IPython.display import display, Latex
from ipywidgets import IntProgress, HTML, Box


class PBinJ(object):
    """ initialize multiple progress bars for tracking nested stages of fitting routine
    """
    def __init__(self, n=1, value=0, status='{}', color='r', width='50%', height='25px'):
        self.displayed = False
        self.style_bar(n=n, value=value, status=status, color=color, width=width, height=height)

    def style_bar(self, n=1, value=0, status='{}', color='r', width='50%', height='25px'):
        colordict = {'g': '#16a085', 'b': '#4168B7', 'r': "#e74c3c", 'y': "#f39c12"}
        self.bar = IntProgress(min=0, max=n, value=value)
        self.status = status
        self.bar.color = colordict[color]
        self.bar.width = width
        self.bar.height = height

    def reset_bar(self):
        self.update(value=0)

    def update(self, value=None, status=None):
        if not self.displayed:
            display(self.bar)
            self.displayed=True
        if status is not None:
            if hasattr(status, '__iter__'):
                status = self.status.format(*status)
            else:
                status = self.status.format(status)
            self.bar.description = status
        if value is not None:
            self.bar.value = value+1

    def clear(self):
        self.bar.close()


class BasinCallback(object):
    """ A callback function for reporting basinhopping status
    Arguments:
        x (array):
            parameter values
        fmin (float):
            function value of the trial minimum, and
        accept (bool):
            whether or not that minimum was accepted
    """
    def __init__(self,  n=1, value=0, status='{:.3fz} / {:.3fz}', color='r'):
        self.pbar = PBinJ(n=n, value=value, status=status, color='r')
        self.reset(history=1, gbasin=1, get_call=0)

    def reset(self, history=True, bar=False, gbasin=False, get_call=False):
        if history:
            self.history = [MyFloat(1.)]
        if gbasin:
            self.gbasin = MyFloat(1.)
        if bar:
            self.pbar.reset_bar()
        if get_call:
            return self.callback

    def callback(self, x, fmin, accept):
        if fmin <= np.min(self.history) and fmin<=self.gbasin:
            self.gbasin = fmin
        if accept:
            self.history.append(fmin)
            status=(MyFloat(x) for x in [fmin, self.gbasin])
            self.pbar.update(value=len(self.history), status=status)
            if len(self.history)>=self.pbar.bar.max:
                # halt run if candidate global minimum has
                # not changed in nsuccess steps (return True)
                return True

    def clear(self):
        self.pbar.clear()



class MyFloat(float):
    """ remove leading zeros from string formatted floats
    """
    def remove_leading_zero(self, value, string):
        if 1 > value > -1:
            string = string.replace('0', '', 1)
        return string

    def __format__(self, format_string):
        if format_string.endswith('z'):
            format_string = format_string[:-1]
            removezero = True
        else:
            removezero = False
        string = super(MyFloat, self).__format__(format_string)
        return self.remove_leading_zero(self, string) if removezero else string


def rwr(X, get_index=False, n=None):
    """
    Modified from http://nbviewer.ipython.org/gist/aflaxman/6871948
    """
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)
    resample_i = np.floor(np.random.rand(n) * len(X)).astype(int)
    X_resample = (X[resample_i])
    if get_index:
        return resample_i
    else:
        return X_resample

def resample_data(data, n=120, groups=['ssd']):
    """ generates n resampled datasets using rwr()
    for bootstrapping model fits
    """
    df = data.copy()
    bootlist = list()
    if n == None:
        n = len(df)
    for level, level_df in df.groupby(groups):
        boots = level_df.reset_index(drop=True)
        orig_ix = np.asarray(boots.index[:])
        resampled_ix = rwr(orig_ix, get_index=True, n=n)
        bootdf = level_df.irow(resampled_ix)
        bootlist.append(bootdf)
    # concatenate and return all resampled conditions
    return self.model.rangl_data(pd.concat(bootlist))

def extract_popt_fitinfo(finfo=None, plist=None, pc_map=None):
    """ takes optimized dict or DF of vectorized parameters and
    returns dict with only depends_on.keys() containing vectorized vals.
    Is accessed by fit.Optimizer objects after optimization routine.
    ::Arguments::
    finfo (dict/DF):
        finfo is dict if self.fit_on is 'average'
        and DF if self.fit_on is 'subjects' or 'bootstrap'
        contains optimized parameters
    ::Returns::
    popt (dict):
        dict with only depends_on.keys() containing
        vectorized vals
    """
    finfo = dict(deepcopy(finfo))
    plist = list(inits)
    popt = {pkey: finfo[pkey] for pkey in plist}
    for pkey in list(pc_map):
        popt[pkey] = np.array([finfo[pc] for pc in pc_map[pkey]])
    return popt

def params_io(p={}, io='w', iostr='popt'):
    """ read // write parameters dictionaries
    """
    if io == 'w':
        pd.Series(p).to_csv(''.join([iostr, '.csv']))
    elif io == 'r':
        ps = pd.read_csv(''.join([iostr, '.csv']), header=None)
        p = dict(zip(ps[0], ps[1]))
        return p

def fits_io(fitparams, fits=[], io='w', iostr='fits'):
    """ read // write y, wts, yhat arrays
    """
    y = fitparams['y'].flatten()
    wts = fitparams['wts'].flatten()
    fits = fits.flatten()
    if io == 'w':
        index = np.arange(len(fits))
        df = pd.DataFrame({'y': y, 'wts': wts, 'yhat': fits}, index=index)
        df.to_csv(''.join([iostr, '.csv']))
    elif io == 'r':
        df = pd.read_csv(''.join([iostr, '.csv']), index_col=0)
        return df
