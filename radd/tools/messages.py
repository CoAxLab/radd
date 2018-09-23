#!/usr/bin/env python
import os
from copy import deepcopy
from future.utils import listvalues
import numpy as np
from numpy.random import randint
from lmfit import fit_report
from time import strftime


def logger(param_report, savepath=None, finfo={}, popt={}, fitparams={}, kind='xdpm', fit_on='average', array_names = ['y', 'wts', 'yhat']):
    """ logs information by opening and appending to an existing log file
    (named according to parameter dependencies) or creating a new log.
    """
    newSection = '-' * 3 + '\n'
    newHeader = lambda x: x + '\n' + '-' * len(x) + '\n'

    if savepath is None:
        savepath = os.path.expanduser('~')
        if 'Dropbox' in os.listdir(savepath):
            savepath = os.path.join(savepath, 'Dropbox')

    # functions for writing numpy arrays to strings (ex. "y = np.array([1,2,3])"")
    name_equals = lambda name, strvector: '{0} = np.array([{1}])'.format(name, strvector)
    stringify = lambda x: name_equals(x[0], ', '.join('{:f}'.format(n) for n in x[1]))
    # brief-ify fitparams reference
    fp = deepcopy(fitparams)
    # list flattened y, wts, and yhat arrays
    arrays = [fp[k].flatten() for k in array_names]
    # write arays to strings for easy logging
    names_arrays = zip(array_names, arrays)
    y_str, wts_str, yhat_str = [stringify(narr) for narr in names_arrays]
    fit_on = '  |  '.join(fp['model_id'].split('_'))

    # limit popt to 6 decimal points
    for k,v in popt.items():
        if isinstance(v, np.ndarray):
            v = np.array([float('{:.6f}'.format(vi)) for vi in v])
        else:
            v = float('{:.6f}'.format(v))
        popt[k] = v
    #popt = {k: float('{:.6f}'.format(v)) for k,v in popt.items()}
    # remove heading from param_report
    param_report = param_report.split(']]')[1]

    if fp['nlevels']==1:
        dep_id = "flat model (no conditional parameters)"
        fname = './' + kind + '_flat.md'
    else:
        depends_on = fp['depends_on']
        dep_id = "\n"
        for p, conds in depends_on.items():
            if not isinstance(conds, list):
                dep_id += "{0} depends on {1}\n".format(p, conds)
            else:
                dep_id += "{0} depends on {1}\n".format(p, ', '.join(np.hstack(conds)))
        dparams = '_'.join(np.hstack(list(depends_on)))
        fname = '_'.join(['./' + kind, dparams + '.md'])

    fname = os.path.abspath(os.path.join(savepath, fname))

    with open(fname, 'a') as f:
        f.write('\n\n')
        f.write(newHeader('MODEL INFO:'))
        f.write(' '.join(['TIMESTAMP:', strftime('%m/%d/%y %I:%M%p'), '\n']))
        f.write(str(fit_on) + '\n')
        f.write(str(dep_id) + '\n\n')
        f.write(newHeader('DATA, YHAT & WEIGHTS:'))
        f.write('\n')
        f.write(wts_str + '\n\n')
        f.write(yhat_str + '\n\n')
        f.write(y_str + '\n')
        f.write('\n\n')
        f.write(newHeader("FIT REPORT:"))
        f.write(param_report + '\n')
        f.write('\n')
        f.write(' = '.join(['popt', repr(popt)]) + '\n')
        f.write('\n\n')
        f.write('ndata: {}\n'.format(finfo['ndata']))
        f.write('df: {}\n'.format(finfo['df']))
        f.write('nfev: {}\n'.format(finfo['nfev']))
        f.write('niter: {}\n'.format(finfo['niter']))
        f.write('chi: {:.8f}\n'.format(finfo['chi']))
        f.write('rchi: {:.8f}\n'.format(finfo['rchi']))
        f.write('logp: {:.4f}\n'.format(finfo['logp']))
        f.write('AIC: {:.4f}\n'.format(finfo['AIC']))
        f.write('BIC: {:.4f}\n'.format(finfo['BIC']))
        f.write('converged: %s' % finfo['cnvrg'] + '\n\n')
        f.write(newSection + '\n\n')


def saygo(depends_on={}, cond_map=None, kind='xdpm', fit_on='subjects'):
    """ generate confirmation message that model is prepared and ready to fit.
    repeats structure and important fit details for user to confirm """
    depkeys = describe_model(depends_on)
    if 'x' in kind:
        bias = '(w/ dynamic bias)'
    else:
        bias = ""
    dep = listvalues(depends_on)
    # flatten list of all cond levels
    lbls = ', '.join(sum(listvalues(depends_on), []))
    msg = get_nonsense()
    strings = (kind, bias, fit_on, pdep, dep, lbls, msg)
    print("""
      Model is prepared to fit %s model %s to %s data,
      allowing %s to vary across levels of %s (%s)
      %s \n""" % strings)
    return True


def describe_model(depends_on=None):
    """ get names of any conditional parameters included in the model
    """
    pkeys = list(depends_on)
    deplist = []
    if 'a' in pkeys:
        deplist.append('boundary')
    if 'tr' in pkeys:
        deplist.append('onset')
    if 'v' in pkeys:
        deplist.append('drift')
    if 'xb' in pkeys:
        deplist.append('xbias')
    if len(pkeys) > 1:
        depkeys = ' and '.join(deplist)
    else:
        depkeys = deplist[0]
    return depkeys

def global_logger(logs):
    arr_str = lambda x: ', '.join(str(elem)[:6] for elem in x)
    str_str = lambda x: ', '.join(str(elem) for elem in x)
    single_str = lambda x: ', '.join(str(elem) for elem in x)
    # create a dictionary string that can be copied and pasted into cell
    #pdict=[':'.join(["'"+str(k)+"'",str(v)]) for k,v in logs['popt'].items()]
    popt = ''.join(['popt =', repr(logs['popt'])])
    fmin_str = 'fmin = %s' % str(logs['fmin'])[:6]
    yhat_str = 'yhat = array([' + arr_str(logs['yhat']) + '])'
    y_str = 'y = array([' + arr_str(logs['y']) + '])'
    cost_str = 'cost = %s' % str(logs['cost'])[:6]
    with open('global_min_report.txt', 'a') as f:
        f.write('==' * 20 + '\n')
        f.write(popt + '\n')
        f.write(fmin_str + '\n')
        f.write(cost_str + '\n')
        f.write(yhat_str + '\n')
        f.write(y_str + '\n')
        f.write('==' * 20 + '\n')

def get_nonsense():
    """ random nonsensical words of encouragement
    """
    msgs = ["Optimize On, Wayne",
            "Optimize On, Garth",
            "See, it's not that simplex...",
            "I wish you a merry fit, and a happy Nature paper",
            "It'll probably work this time",
            "'They dont think it be like it is, but it do' -Oscar Gamble",
            "Check your zoomfile before optimizing. It should be in your computer",
            "It's IN the computer?",
            "What is this... a model for ANTS!?"]
    return msgs[randint(0, len(msgs))]
