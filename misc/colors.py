#!/usr/local/bin/env python
from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from radd import RADD, boldfx, fitre, utils
from scipy.stats.mstats import mquantiles as mq

sns.set(font="Helvetica")


def style_params(context='notebook'):
      colors = ["#1abc9c","#16a085","#2ecc71","#27ae60",
      "#3498db", "#2980b9", "#4168B7", "#3A539B",
      "#9B59B6", "#8E44AD", "#663399", "#674172",
      "ghostwhite", '#95A5A6', '#6C7A89', "#34495e", "#2c3e50",
      '#E26A6A', "#e16256", "#e74c3c", "#ca4440", "#c0392b",
      "#f1c40f", "#f39c12", "#e67e22"]

      return {'colors':colors,'greens':colors[:4], 'blues':colors[4:8], 'purples':colors[8:12], 'grays':colors[12:17], 'reds':colors[17:22], 'yellows':colors[22:]}
