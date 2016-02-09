#! /usr/local/bin/env python
import matplotlib as mpl
from matplotlib.pyplot as plt
import seaborn as sns
from radd.tools import colors
from radd import analyze
import prettyplotlib as pl


class FSaver():

    def __init__(self, fig=None, style='white'):

        self.fig = fig
        self.style = style
        self.set_style_params()

        self.rpal = lambda nc: sns.blend_palette(['#e88379', '#c0392b'], n_colors=nc)
        self.bpal = lambda nc: sns.blend_palette(['#81aedb', '#3A539B'], n_colors=nc)
        self.gpal = lambda nc: sns.blend_palette(['#65b88f', '#27ae60'], n_colors=nc)
        self.ppal = lambda nc: sns.blend_palette(['#848bb6', '#663399'], n_colors=nc)
        self.heat = lambda nc: sns.blend_palette(['#f39c12', '#c0392b'], n_colors=nc)
        self.cool = lambda nc: sns.blend_palette(["#4168B7", "#27ae60"], n_colors=nc)
        self.slate = lambda nc: sns.blend_palette(['#95A5A6', "#6C7A89"], n_colors=nc)

    def set_style_params(self, style=None):

        if style != None:
            self.style = style

        if self.style == 'white':
            fc = 'white'
            tc = 'black'
        else:
            fc = '#39414F'
            tc = 'white'

        sns.set(style='ticks', font='Helvetica', rc={'text.color': tc, 'axes.labelcolor': tc, 'figure.facecolor': fc, 'axes.facecolor': fc, 'figure.edgecolor': tc, 'axes.edgecolor': tc, 'xtick.color': tc, 'ytick.color': tc, 'lines.linewidth': 2})

    def savefig(self, fig=None):

        if fig is None:
            fig = self.fig

        for ax in fig.axes():
            ax.patch.set_facecolor('#39414F')
        fig.savefig(''.join([savestr, '.png']), facecolor=white, edgecolor='k')
        fig.patch.set_facecolor('#39414F')
        fig.patch.set_alpha(0.7)
        ax = fig.add_subplot(111)
        ax.plot(range(10))
        ax.patch.set_facecolor('red')
        ax.patch.set_alpha(0.5)

        # If we don't specify the edgecolor and facecolor for the figure when
        # saving with savefig, it will override the value we set earlier!
        fig.savefig('temp.png', facecolor=white, edgecolor='k', )
