#!/usr/bin/env python
from numpy.random import randint

def get_one():

      msgs = ["Optimize On, Wayne",
      "Optimize On, Garth",
      "See, it's not that simplex...",
      "I wish you a steep gradient, and a happy Nature paper",
      "It'll probably work this time",
      "'They dont think it be like it is, but it do' -Oscar Gamble"]

      return msgs[randint(0, len(msgs))]

def saygo(depends_on={}, labels=[], kind='radd', fit_on='subjects'):

      pdeps = depends_on.keys()
      deplist = []
      if 'a' in pdeps:
            deplist.append('Boundary Height')
      if 'tr' in pdeps:
            deplist.append('Onset Time')
      if 'v' in pdeps:
            deplist.append('Drift-Rate')

      if len(pdeps)>1:
            pdep = ' and '.join(deplist)
      else:
            pdep = deplist[0]

      dep = depends_on.values()[0]
      lbls = ', '.join(labels)
      msg = get_one()
      strings = (fit_on, kind, pdep, dep, lbls, msg)

      print """
      Model is prepared to fit on %s %s data, allowing
      %s to vary across levels of %s (%s) \n
      %s """ % strings

      return True
