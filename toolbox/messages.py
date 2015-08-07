#!/usr/bin/env python
from numpy.random import randint

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
      if 'a' in pdeps:
            deplist.append('Boundary Height')
      if 'tr' in pdeps:
            deplist.append('Onset Time')
      if 'v' in pdeps:
            deplist.append('Drift-Rate')
      if 'xb' in pdeps:
            deplist.append('Dynamic Drift')

      if len(pdeps)>1:
            pdep = ' and '.join(deplist)
      else:
            pdep = deplist[0]
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
