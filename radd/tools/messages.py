#!/usr/bin/env python
import os
from copy import deepcopy
from future.utils import listvalues
import numpy as np
from numpy.random import randint
from lmfit import fit_report

def logger(param_report, finfo={}, popt={}, fitparams={}, kind='xdpm', fit_on='average', array_names = ['y', 'wts', 'yhat']):
    """ logs information by opening and appending to an existing log file
    (named according to parameter dependencies) or creating a new log.
    """
    # functions for writing numpy arrays to strings (ex. "y = np.array([1,2,3])"")
    name_equals = lambda name, strvector: '{0} = array([{1}])'.format(name, strvector)
    stringify = lambda x: name_equals(x[0], ', '.join('{:f}'.format(n) for n in x[1]))
    # brief-ify fitparams reference
    fp = dict(deepcopy(fitparams))
    # list flattened y, wts, and yhat arrays
    arrays = [fp[k].flatten() for k in array_names]
    # write arays to strings for easy logging
    names_arrays = zip(array_names, arrays)
    y_str, wts_str, yhat_str = map(stringify, names_arrays)

    if fp['nlevels']==1:
        fit_on = ' '.join([fit_on, 'FLAT'])
        dep_id = "flat model (no conditional parameters)"
        fname = './' + kind + '_flat.txt'
    else:
        depends_on = fp['depends_on']
        fit_on = ' '.join([fit_on, 'FULL'])
        pkeys = '_'.join(list(depends_on))
        pconds = '_'.join(listvalues(depends_on))
        dep_id = "{0} depends on {1}".format(pconds, pkeys)
        fname = '_'.join(['./' + kind, pconds + '.txt'])

    with open(fname, 'a') as f:
        f.write('\n\n')
        f.write('==' * 30 + '\n\n')
        f.write(str(fit_on) + '\n')
        f.write(str(kind) + '\n')
        f.write(str(dep_id) + '\n\n')
        f.write(wts_str + '\n\n')
        f.write(yhat_str + '\n\n')
        f.write(y_str + '\n\n')
        f.write('--' * 30 + '\n')
        f.write("FIT REPORT")
        f.write('\n' + '--' * 30 + '\n')
        f.write(param_report)
        f.write('\n' + '--' * 30 + '\n')
        f.write('='.join(['popt', repr(popt)]) + '\n')
        f.write('AIC: %.8f' % finfo['AIC'] + '\n')
        f.write('BIC: %.8f' % finfo['BIC'] + '\n')
        f.write('chi: %.8f' % finfo['chi'] + '\n')
        f.write('rchi: %.8f' % finfo['rchi'] + '\n')
        f.write('converged: %s' % finfo['cnvrg'] + '\n\n')
        f.write('==' * 30 + '\n\n\n')

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
