#!/usr/local/bin/env python
from __future__ import division
import os
from future.utils import listvalues
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import randint
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore", module="matplotlib")

def get_cpals(name='all', aslist=False, random=False):
    rpal = lambda nc: sns.blend_palette(['#e88379', '#de143d'], n_colors=nc)
    bpal = lambda nc: sns.blend_palette(['#81aedb', '#3572C6'], n_colors=nc)
    gpal = lambda nc: sns.blend_palette(['#65b88f', '#27ae60'], n_colors=nc)
    ppal = lambda nc: sns.blend_palette(['#9B59B6', "#663399"], n_colors=nc)
    heat = lambda nc: sns.blend_palette(['#f39c12', '#e5344a'], n_colors=nc)
    cool = lambda nc: sns.blend_palette(["#4168B7", "#27ae60"], n_colors=nc)
    slate = lambda nc: sns.blend_palette(['#95A5A6', "#6C7A89"], n_colors=nc)
    wet = lambda nc: sns.blend_palette(['#34495e', "#99A4AE"], n_colors=nc)
    fire = lambda nc: sns.blend_palette(['#e5344a', "#f39c12"], n_colors=nc)
    bupu = lambda nc: sns.blend_palette(['#8E44AD', "#3498db"], n_colors=nc)
    color_dict = {'bpal': bpal, 'gpal': gpal, 'rpal': rpal, 'ppal': ppal, 'heat': heat, 'cool': cool, 'slate': slate, 'wet': wet, 'fire':fire, 'bupu': bupu}
    if random:
        pals = listvalues(color_dict)
        i = randint(0, len(pals), 1)
        return pals[i]
    if name=='all':
        if aslist:
            return listvalues(color_dict)
        return color_dict
    else:
        return color_dict[name]

def style_params():
    colors = ["#1abc9c", "#16a085", "#2ecc71", "#27ae60", '#009e07', '#94c273', '#83a83b',
              "#3498db", "#2980b9", '#3572C6', "#4168B7", "#3A539B",
              '#8172b2', "#9B59B6", "#8E44AD", "#674172", "#663399",
              "#95A5A6", '#6C7A89', "#34495e", "#2c3e50",
              '#E26A6A', "#e16256", "#e74c3c", "#e5344a", '#de143d', "#c0392b",
              "#f1c40f", "#f39c12", "#e67e22", "#ff914d", "#ff711a"]
    return {'colors': colors, 'greens': colors[:4], 'blues': colors[4:8], 'purples': colors[8:12], 'grays': colors[12:17], 'reds': colors[17:22], 'yellows': colors[22:]}

def param_color_map(param='all'):
    param_color_map = {'a': "#375ee1", 'tr': "#f19b2c", 'v': "#27ae60", 'xb': "#16a085", 'ssv': "#e5344a", 'ssv_v': "#3498db", 'sso': "#e941cd", 'z': '#ff711a', 'all': '#6C7A89', 'flat': '#6C7A89', 'C': '#009e07', 'B': '#de143d', 'Beta': "#ff711a", 'v_ssv': "#9B59B6"}
    if param=='all':
        return param_color_map
    if param in list(param_color_map):
        return param_color_map[param]
    elif '_' in param:
        params = param.split('_')
        blended = [param_color_map[p] for p in params]
        return sns.blend_palette(blended, n_colors=6)[3]
    elif param not in list(param_color_map):
        clrs = assorted_list()
        ix = np.random.randint(0, len(clrs))
        return clrs[ix]

def assorted_list():
    return ['#3572C6',  '#c44e52', '#8172b2', '#83a83b', "#3498db", "#e5344a", '#94c273', '#6C7A89', "#8E44AD", "#16a085", "#f39c12", "#4168B7", '#34495e', "#27ae60", "#e74c3c", "#ff711a", "#ff914d"]

def random_colors(n):
    colornames = list(sns.crayons)
    r_ints = randint(0, high=len(colornames), size=n)
    cnames = [colornames[i] for i in r_ints]
    return [sns.crayons[name] for name in cnames]

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
