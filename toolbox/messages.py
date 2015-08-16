#!/usr/bin/env python
from numpy.random import randint
from lmfit import fit_report

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

      print """
      Model is prepared to fit %s model %s to %s data,
      allowing %s to vary across levels of %s (%s)

      %s \n""" % strings

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
            deplist.append('dynamic_v')

      if len(pdeps)>1:
            pdep = ' and '.join(deplist)
      else:
            pdep = deplist[0]

      return pdep



def logger(optmod, finfo={}, depends_on={}, log_arrays={}, kind=None, dynamic=None, fit_id=None, xbasin=None):

      wts, y, yhat = log_arrays['wts'], log_arrays['y'], log_arrays['yhat']

      finfo['chi'] = optmod.chisqr
      finfo['rchi'] = optmod.redchi
      finfo['CNVRG'] = optmod.pop('success')
      finfo['nfev'] = optmod.pop('nfev')
      finfo['AIC']=optmod.aic
      finfo['BIC']=optmod.bic

      pkeys = depends_on.keys()
      pvals = depends_on.values()
      model_id = "MODEL: %s" % kind
      if 'x' in kind:
            model_id = ' ('.join([model_id, dynamic])+')'
      dep_id = "%s DEPENDS ON %s" % (pvals[0], str(tuple(pkeys)))
      wts_str = 'wts = array(['+ ', '.join(str(elem)[:6] for elem in wts)+'])'
      yhat_str = 'yhat = array(['+ ', '.join(str(elem)[:6] for elem in yhat)+'])'
      y_str = 'y = array(['+ ', '.join(str(elem)[:6] for elem in y)+'])'

      if xbasin:
            write_basin=True
            bpopt_str = 'basin = array(['+ ', '.join(str(elem) for elem in xbasin)+'])'
      else:
            write_basin=False
      with open('fit_report.txt', 'a') as f:
            if write_basin:
                  f.write('--'*20+'\n')
                  f.write('BASINHOPPING RESULTS\n')
                  f.write(bpopt_str+'\n')
                  f.write('--'*20+'\n')
            f.write('=='*20+'\n')
            f.write(str(fit_id)+'\n')
            f.write(str(model_id)+'\n')
            f.write(str(dep_id)+'\n')
            f.write('--'*20+'\n')
            f.write(wts_str+'\n')
            f.write(yhat_str+'\n')
            f.write(y_str+'\n')
            f.write('--'*20+'\n')
            try:
                  f.write(fit_report(optmod)+'\n\n')
            except Exception:
                  pass
            f.write('AIC: %.8f' % optmod.aic + '\n')
            f.write('BIC: %.8f' % optmod.bic + '\n')
            f.write('chi: %.8f' % optmod.chisqr + '\n')
            f.write('rchi: %.8f' % optmod.redchi + '\n')
            f.write('Converged: %s' % finfo['CNVRG'] + '\n')
            f.write('=='*20+'\n\n')
