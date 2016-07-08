#!/usr/local/bin/env python
from __future__ import division
import os
from future.utils import listvalues
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font="Helvetica")

def get_cpals(name='all', aslist=False):
    rpal = lambda nc: sns.blend_palette(['#e88379', '#c0392b'], n_colors=nc)
    bpal = lambda nc: sns.blend_palette(['#81aedb', '#3A539B'], n_colors=nc)
    gpal = lambda nc: sns.blend_palette(['#65b88f', '#27ae60'], n_colors=nc)
    ppal = lambda nc: sns.blend_palette(['#848bb6', "#9B59B6"], n_colors=nc)
    heat = lambda nc: sns.blend_palette(['#f39c12', '#c0392b'], n_colors=nc)
    cool = lambda nc: sns.blend_palette(["#4168B7", "#27ae60"], n_colors=nc)
    slate = lambda nc: sns.blend_palette(['#95A5A6', "#6C7A89"], n_colors=nc)
    color_dict = {'bpal': bpal, 'gpal': gpal, 'rpal': rpal, 'ppal': ppal, 'heat': heat, 'cool': cool, 'slate': slate}
    if name=='all':
        if aslist:
            return listvalues(color_dict)
        return color_dict
    else:
        return color_dict[name]


def style_params(context='notebook'):
    colors = ["#1abc9c", "#16a085", "#2ecc71", "#27ae60",
              "#3498db", "#2980b9", "#4168B7", "#3A539B",
              "#9B59B6", "#8E44AD", "#663399", "#674172",
              "ghostwhite", '#95A5A6', '#6C7A89', "#34495e", "#2c3e50",
              '#E26A6A', "#e16256", "#e74c3c", "#ca4440", "#c0392b", "#bd4a4c",
              "#f1c40f", "#f39c12", "#e67e22"]

    return {'colors': colors, 'greens': colors[:4], 'blues': colors[4:8], 'purples': colors[8:12], 'grays': colors[12:17], 'reds': colors[17:22], 'yellows': colors[22:]}


def get_cmaps():
    block = ['Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd', 'PuBu',
             'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd'],
    seq = ['afmhot', 'autumn', 'bone', 'cool', 'copper',
           'gist_heat', 'gray', 'hot', 'pink', 'summer', 'winter']
    div = ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
           'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral', 'seismic']
    qual = ['Accent', 'Dark2', 'Paired', 'Pastel1',
            'Pastel2', 'Set1', 'Set2', 'Set3']

    misc = ['gist_earth', 'terrain', 'ocean', 'gist_stern', 'brg', 'CMRmap', 'cubehelix', 'gnuplot',
            'gnuplot2', 'gist_ncar', 'nipy_spectral', 'jet', 'rainbow', 'gist_rainbow', 'hsv', 'flag', 'prism']
    cmaps = {'block': block, 'div': div,
             'seq': seq, 'qual': qual, 'misc': misc}
    return cmaps
