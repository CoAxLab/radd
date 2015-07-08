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

def flatui():

      return {"t1":"#1abc9c","t2":"#16a085","g1":"#2ecc71","g2":"#27ae60",
                "b1":"#2980b9","b2":"#4168B7","p1":"#9B59B6","p2":"#674172",
                "m1":"#34495e","m2":"#2c3e50","y1":"#f1c40f","y2":"#f39c12",
                "o1":"#e67e22","o2":"#d35400","r1":"#e74c3c","r2":"#c0392b",
                "gr1":"#ecf0f1", "gr2":"#bdc3c7","a1":"#95a5a6","a2":"#7f8c8d" }

def get_hsv(hexrgb):
    hexrgb = hexrgb.lstrip("#")   # in case you have Web color specs
    r, g, b = (int(hexrgb[i:i+2], 16) / 255.0 for i in xrange(0,5,2))

    return colorsys.rgb_to_hsv(r, g, b)

def style_params(style='ticks', context='notebook'):

	if style=='ticks':
		rcdict={'axes.axisbelow': True,'axes.edgecolor': '.15','axes.facecolor': 'white','axes.grid': False,'axes.labelcolor': '.15',
		'axes.linewidth': 1.2,'font.family': 'Helvetica','grid.color': '.8','grid.linestyle': '-','image.cmap': 'Greys',
		'legend.frameon': False,'legend.numpoints': 1,'legend.scatterpoints': 1,'lines.solid_capstyle': 'round','pdf.fonttype': 42,
		'text.color': '.15','xtick.color': '.15','xtick.direction': 'out','xtick.major.size': 6,'xtick.minor.size': 3,'ytick.color': '.15',
		'ytick.direction': 'out','ytick.major.size': 6,'ytick.minor.size': 3}

	colors=['ghostwhite', '#95A5A6', '#6C7A89',
	'#3498db', '#4168B7', '#5C97BF', '#34495e', '#3A539B', '#4B77BE',
	(0.21568627450980393, 0.47058823529411764, 0.7490196078431373),
 	(0.23137254901960785, 0.3568627450980392, 0.5725490196078431),
	'#2ecc71', '#009B76', '#00B16A',"mediumseagreen", '#16A085', '#019875',
	(0.5098039215686274, 0.37254901960784315, 0.5294117647058824),
	'#674172', '#9B59B6', '#8E44AD', '#BF55EC', '#663399', '#9A12B3',
	(0.996078431372549, 0.7019607843137254, 0.03137254901960784),
	'#F27935','#E26A6A', '#F62459',
	(0.8509803921568627, 0.32941176470588235, 0.30196078431372547),
	'#D91E18', '#F64747', '#e74c3c','#CF000F']

	return {'style':rcdict, 'colors':colors, 'reds':colors[-9:],
		'purples':colors[-16:-9], 'greens':colors[-22:-16], 'grays':colors[:3],'blues':colors[3:11]}
