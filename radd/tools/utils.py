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
from IPython.display import HTML, Javascript, display

class PBinJ(object):
    """ animated ProgressBar (PB) to be used (inJ)upyter notebooks
    (set bartype to 'uglybar' if running from terminal)
    """
    def __init__(self, bartype='colorbar', n=None, color='blue', title='Progress'):
        colors = {'green': '#16a085', 'blue': '#3A539B', 'red': "#e74c3c"}
        self.color = colors[color]
        self.n=n
        self.displayed=False
        self.bartype=bartype
        self.title=title
        self.init_bars()

    def init_bars(self):
        if self.bartype=='uglybar':
            import progressbar
            if self.n is not None:
                self.bar = progressbar.ProgressBar(0, self.n)
            self.new_prog_string = ''.join([self.title, ': {0}'])
            self.update = self.update_uglybar
        else:
            import uuid
            self.barid=str(uuid.uuid4())
            if self.bartype=='colorbar':
                args = [self.title, self.color, "500", self.barid, '0%', 'left']
                self.update = self.update_colorbar
            else:
                args = [self.title, self.color, "108", self.barid, '100%', 'center']
                self.update = self.update_progress
            self.bar="""<div<p>{0}</p> <div style="border: 2px solid {1}; width:{2}px">
            <div id="{3}" style="background-color:{1}; width:{4}; text:''; color:#fff; text-align:{5};">
            &nbsp;</div> </div> </div>""".format(*args)

    def display_bars(self):
        if self.bartype=='uglybar':
            self.bar.start()
        else:
            display(HTML(self.bar))
        self.displayed=True

    def update_progress(self, new_progress=None):
        if self.displayed==False:
            self.display_bars()
        display(Javascript("$('div#{}').text({:.5f})".format(self.barid, new_progress)))

    def update_colorbar(self, i=None, new_progress=None):
        if self.displayed==False:
            self.display_bars()
        if i is not None:
            display(Javascript("$('div#{}').width('{:.2f}%')".format(self.barid, ((i+1)*1./(self.n))*100)))
        if new_progress is not None:
            self.update_progress(new_progress)

    def update_uglybar(self, i=None, new_progress=None):
        if self.displayed==False:
            self.display_bars()
        if new_progress is not None:
            sys.stdout.write('\r'+self.new_prog_string.format(str(new_progress)))
            sys.stdout.flush()
        if i is not None:
            self.bar.update(i)

    def clear(self):
        if self.bartype=='uglybar':
            sys.stdout.flush
        else:
            from IPython.display import clear_output
            clear_output()


class NestedProgress(object):
    """ initialize multiple progress bars for tracking nested stages of fitting routine
    """
    def __init__(self, name='inits_bar', bartype='colorbar', n=None, init_state=None, color='blue', title='global fmin'):
        self.bars = {}
        self.history = []
        self.add_bar(name=name, bartype=bartype, n=n, init_state=init_state, color=color, title=title)

    def add_bar(self, name='inits_bar', bartype='colorbar', n=None, init_state=None, color='blue', title='global fmin'):
        bar = PBinJ(bartype=bartype, n=n, color=color, title=title)
        self.bars[name] = bar
        if init_state is not None:
            self.reset_bar(name, init_state)

    def reset_bar(self, name, init_state=None):
        self.history = [init_state]
        self.bars[name].update(new_progress=init_state)

    def update(self, name='all', i=None, new_progress=None):
        if name=='all':
            update_list = listvalues(self.bars)
        else:
            update_list = [self.bars[name]]
        for bar in update_list:
            if bar.bartype=='infobar' and new_progress:
                bar.update(new_progress)
            elif bar.bartype=='colorbar':
                bar.update(i=i, new_progress=new_progress)

    def callback(self, x, fmin, accept):
        """ A callback function for reporting basinhopping status
        Arguments:
            x (array):
                parameter values
            fmin (float):
                function value of the trial minimum, and
            accept (bool):
                whether or not that minimum was accepted
        """
        if fmin <= np.min(self.history):
            self.bars['glb_basin'].update(new_progress=fmin)
        if accept:
            self.history.append(fmin)
            self.bars['lcl_basin'].update(new_progress=fmin)

    def clear(self):
        for bar in listvalues(self.bars):
            bar.clear()


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
