#!/usr/bin/env python
from numpy.random import randint
from lmfit import report_fit, fit_report
from copy import deepcopy
import os


def get_one():

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


def saygo(depends_on={}, labels=[], kind='radd', fit_on='subjects', dynamic='hyp'):

    pdeps = depends_on.keys()
    deplist = []
    pdep = describe_model(depends_on)

    if 'x' in kind:
        bias = '(w/ %s dynamic bias)' % dynamic
    else:
        bias = ""
    dep = depends_on.values()[0]
    lbls = ', '.join(labels)
    msg = get_one()
    strings = (kind, bias, fit_on, pdep, dep, lbls, msg)

    print("""
      Model is prepared to fit %s model %s to %s data,
      allowing %s to vary across levels of %s (%s)

      %s \n""" % strings)

    return True


def basin_accept_fun(x, f, accepted):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))


def describe_model(depends_on=None):

    pdeps = depends_on.keys()
    deplist = []
    if 'a' in pdeps:
        deplist.append('boundary')
    if 'tr' in pdeps:
        deplist.append('onset')
    if 'v' in pdeps:
        deplist.append('drift')
    if 'xb' in pdeps:
        deplist.append('xbias')

    if len(pdeps) > 1:
        pdep = ' and '.join(deplist)
    else:
        pdep = deplist[0]

    return pdep


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


def logger(param_report, finfo={}, depends_on={}, pdict={}, is_flat=True, log_arrays={}, kind=None, dynamic=None, fit_on=None, pc_map=None):

    wts, y, yhat = log_arrays['wts'], log_arrays['y'], log_arrays['yhat']
    if is_flat:
        fit_on = ' '.join([fit_on, 'FLAT'])
    else:
        fit_on = ' '.join([fit_on, 'FULL'])
    pkeys = depends_on.keys()
    pvals = depends_on.values()
    fname = '_'.join(['./' + kind, '_'.join(pkeys) + '.txt'])
    model_id = "MODEL: %s" % kind
    if 'x' in kind:
        model_id = ' ('.join([model_id, dynamic]) + ')'
    dep_id = "%s DEPENDS ON %s" % (pvals[0], str(tuple(pkeys)))
    wts_str = 'wts = array([' + ', '.join(str(elem)[:6] for elem in wts) + '])'
    yhat_str = 'yhat = array([' + ', '.join(str(elem)[:6] for elem in yhat) + '])'
    y_str = 'y = array([' + ', '.join(str(elem)[:6] for elem in y) + '])'
    with open(fname, 'a') as f:
        f.write('\n\n')
        f.write('==' * 30 + '\n\n')
        f.write(str(fit_on) + '\n')
        f.write(str(model_id) + '\n')
        f.write(str(dep_id) + '\n\n')
        f.write(wts_str + '\n\n')
        f.write(yhat_str + '\n\n')
        f.write(y_str + '\n\n')
        f.write('--' * 30 + '\n')
        f.write("FIT REPORT")
        f.write('\n' + '--' * 30 + '\n')
        f.write(param_report)
        f.write('\n' + '--' * 30 + '\n')
        f.write('='.join(['popt', repr(pdict)]) + '\n')
        f.write('AIC: %.8f' % finfo['AIC'] + '\n')
        f.write('BIC: %.8f' % finfo['BIC'] + '\n')
        f.write('chi: %.8f' % finfo['chi'] + '\n')
        f.write('rchi: %.8f' % finfo['rchi'] + '\n')
        f.write('converged: %s' % finfo['cnvrg'] + '\n\n')
        f.write('==' * 30 + '\n\n\n')
    return finfo
