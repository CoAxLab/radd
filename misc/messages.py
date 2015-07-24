#!/usr/bin/env python
from numpy.random import randint

def get_one():

      msgs = ["Optimize On, Wayne",
      "Optimize On, Garth",
      "I wish you a steep gradient, and a happy Nature paper",
      "See it's not that Simplex, wait that's an oxymoron!",
      "Go to bed",
      "It'll probably work this time",
      "'They dont think it be like it is, but it do' -Oscar Gamble"]

      return msgs[randint(0, len(msgs))]
