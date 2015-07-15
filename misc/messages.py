#!/usr/bin/env python
from numpy.random import randint

def get_one():

      msgs=["Optimize On, Wayne",
      "Optimize On, Garth",
      "May the Nelder be with you",
      "I wish you a slippery gradient, and a happy Nature paper",
      "See it's not that Simplex, wait that's an oxymoron!",
      "Go to bed",
      "It'll probably work this time",
      '"They Don’t Think It Be Like It Is, But It Do"\n-Oscar Gamble']

      return randint(0, len(msgs)-1)
