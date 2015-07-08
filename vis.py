#!/usr/local/bin/env python
from __future__ import division
import os
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
from radd import utils
from scipy.stats.mstats import mquantiles as mq
sns.set(font='Helvetica')

def scurves(lines=[], task='ssRe', yerr=[], pstop=.5, ax=None, linestyles=None, colors=None, labels=None):

      if len(lines[0])==6:
                task='pro'

      if ax is None:
		f, ax = plt.subplots(1, figsize=(6,5))
      if colors is None:
            colors=sns.color_palette('husl', n_colors=len(lines))
            labels=['']*len(lines)
            linestyles = ['-']*len(lines)

      lines=[np.array(line) if type(line)==list else line for line in lines]
      pse=[];

      if 'Re' in task:
            x=np.array([400, 350, 300, 250, 200], dtype='float')
            xtls=x.copy()[::-1]; xsim=np.linspace(15, 50, 10000);
            yylabel='P(Stop)'; scale_factor=100; xxlabel='SSD'; xxlim=(18,42)
      else:
            x=np.array([100, 80, 60, 40, 20, 0], dtype='float')
            xtls=x.copy()[::-1]; xsim=np.linspace(-5, 11, 10000)
            yylabel='P(NoGo)'; scale_factor=100; xxlabel='P(Go)'; scale_factor=10

      x=utils.res(-x, lower=x[-1]/10, upper=x[0]/10)
      for i, yi in enumerate(lines):

            y=utils.res(yi, lower=yi[-1], upper=yi[0])
            p_guess=(np.mean(x),np.mean(y),.5,.5)
            p, cov, infodict, mesg, ier = optimize.leastsq(utils.residuals, p_guess, args=(x,y), full_output=1, maxfev=5000, ftol=1.e-20)
            x0,y0,c,k=p
            xp = xsim
            pxp=utils.sigmoid(p,xp)
            idx = (np.abs(pxp - pstop)).argmin()

            pse.append(xp[idx]/scale_factor)

            # Plot the results
            if yerr!=[]:
                  #ax.errorbar(x, y[i], yerr=yerr[i], color=colors[i], marker='o', elinewidth=2, ecolor='k')
                  ax.errorbar(x, y, yerr=yerr[i], color=colors[i], ecolor=colors[i], capsize=0, lw=0, elinewidth=3)
            ax.plot(xp, pxp, linestyle=linestyles[i], lw=3.5, color=colors[i], label=labels[i])
            pse.append(xp[idx]/scale_factor)

      plt.setp(ax, xlim=xxlim, xticks=x, ylim=(-.05, 1.05), yticks=[0, 1])
      ax.set_xticklabels([int(xt) for xt in xtls], fontsize=18); ax.set_yticklabels([0.0, 1.0], fontsize=18)
      ax.set_xlabel(xxlabel, fontsize=18); ax.set_ylabel(yylabel, fontsize=18)
      ax.legend(loc=0)
      return np.array(pse)


def gen_pro_traces(ptheta, bias_vals=[], bias='v', integrate_exec_ss=False, return_exec_ss=False, pgo=np.arange(0, 1.2, .2)):

      dvglist=[]; dvslist=[]

      if bias_vals==[]:
            deplist=np.ones_like(pgo)

      for val, pg in zip(bias_vals, pgo):
            ptheta[bias] = val
            ptheta['pGo'] = pg
            dvg, dvs = RADD.run(ptheta, ntrials=10, tb=.565)
            dvglist.append(dvg[0])

      if pg<.9:
            dvslist.append(dvs[0])
      else:
            dvslist.append([0])

      if integrate_exec_ss:
            ssn = len(dvslist[0])
            traces=[np.append(dvglist[i][:-ssn],(dvglist[i][-ssn:]+ss)-dvglist[i][-ssn:]) for i, ss in enumerate(dvslist)]
            traces.append(dvglist[-1])
            return traces

      elif return_exec_ss:
            return [dvglist, dvslist]

      else:
            return dvglist


def gen_re_traces(rtheta, integrate_exec_ss=False, ssdlist=np.arange(.2, .45, .05)):

      dvglist=[]; dvslist=[]
      rtheta['pGo']=.5
      rtheta['ssv']=-abs(rtheta['ssv'])
      #animation only works if tr<=ssd
      rtheta['tr']=np.min(ssdlist)-.001

      for ssd in ssdlist:
            rtheta['ssd'] = ssd
            dvg, dvs = RADD.run(rtheta, ntrials=10, tb=.650)
            dvglist.append(dvg[0])
            dvslist.append(dvs[0])

      if integrate_exec_ss:
            ssn = len(dvslist[0])
            traces=[np.append(dvglist[i][:-ssn],(dvglist[i][-ssn:]+ss)-dvglist[i][-ssn:]) for i, ss in enumerate(dvslist)]
            traces.append(dvglist[-1])
            return traces

      ssi, xinit_ss = [], []
      for i, (gtrace, strace) in enumerate(zip(dvglist, dvslist)):
            leng = len(gtrace)
            lens = len(strace)
            xinit_ss.append(leng - lens)
            ssi.append(strace[0])
            dvslist[i] = np.append(gtrace[:leng-lens], strace)
            dvslist[i] = np.append(dvslist[i], np.array([0]))

      return [dvglist, dvslist, xinit_ss, ssi]


def build_decision_axis(theta, gotraces):

      # init figure, axes, properties
      f, ax = plt.subplots(1, figsize=(7,4))

      w=len(gotraces[0])+50
      h=theta['a']
      start=-100

      plt.setp(ax, xlim=(start-1, w+1), ylim=(0-(.01*h), h+(.01*h)))

      ax.hlines(y=h, xmin=-100, xmax=w, color='k')
      ax.hlines(y=0, xmin=-100, xmax=w, color='k')
      ax.hlines(y=theta['z'], xmin=start, xmax=w, color='Gray', linestyle='--', alpha=.7)
      ax.vlines(x=w-50, ymin=0, ymax=h, color='r', linestyle='--', alpha=.5)
      ax.vlines(x=start, ymin=0, ymax=h, color='k')

      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.set_xticks([])
      ax.set_yticks([])

      sns.despine(top=True, right=True, bottom=True, left=True)

      return f, ax


def re_animate(i, x, dvg_traces, dvg_lines, dvs_traces, dvs_lines, rtheta, xi, yi):

      clist=['#2ecc71']*len(dvg_traces)
      clist_ss = sns.light_palette('#e74c3c', n_colors=6)[::-1]

      for nline, (gl, g) in enumerate(zip(dvg_lines, dvg_traces)):
            if g[i]>=rtheta['a'] or dvs_traces[nline][i]<=0:
                  continue
            gl.set_data(x[:i+1], g[:i+1])
            gl.set_color(clist[nline])

            if dvs_traces[nline][i]>0:
                  ssi = len(g) - len(dvs_traces[nline]) + 1
                  dvs_lines[nline].set_data(x[xi[nline]:i+1], dvs_traces[nline][xi[nline]:i+1])
                  dvs_lines[nline].set_color(clist_ss[nline])

      return dvs_lines, dvg_lines


def pro_animate(i, x, protraces, prolines):

      clist = sns.color_palette('autumn', n_colors=6)[::-1]

      for nline, (pline, ptrace) in enumerate(zip(prolines, protraces)):
            pline.set_data(x[:i+1], ptrace[:i+1])
            pline.set_color(clist[nline])

      return prolines,


def plot_traces(DVg, DVs, sim_theta, tau=.0005, tb=.5451, cg='Green', cr='Red'):

      f,ax=plt.subplots(1,figsize=(8,5))
      tr=sim_theta['tr']; a=sim_theta['a']; z=sim_theta['z']; ssd=sim_theta['ssd']

      for i, igo in enumerate(DVg):
            plt.plot(np.arange(tr, tb, tau), igo, color=cg, alpha=.1, linewidth=.5)
            if i<len(DVs):
                  plt.plot(np.arange(ssd, tb, tau), DVs[i], color=cr, alpha=.1, linewidth=.5)

      plt.setp(ax, xlim=(tr, tb), ylim=(0,a))
      ax.hlines(y=z, xmin=tr, xmax=tb, linewidth=3, linestyle='--', color="#6C7A89")
      sns.despine(top=False,bottom=False, right=True, left=False)
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.set_xticks([])
      ax.set_yticks([])

      return ax
