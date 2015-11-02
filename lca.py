import random
from numpy.random import sample as rs
from numpy import hstack as hs
from numpy import newaxis as na
from radd import theta
from scipy.stats.distributions import norm, uniform
from scipy.misc import factorial
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns

temporal_dynamics = lambda p, t: np.cosh(p['xb'][:,na]*t)
resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=2)*dt
ss_resp_up = lambda trace, a: np.argmax((trace.T>=a).T, axis=3)*dt
resp_lo = lambda trace: np.argmax((trace.T<=0).T, axis=3)*dt
RT = lambda ontime, rbool: ontime[:,na]+(rbool*np.where(rbool==0, np.nan, 1))
RTQ = lambda zpd: map((lambda x:mq(x[0][x[0]<x[1]], prob)), zpd)
prop = lambda n, k: factorial(n)/(factorial(k)*factorial(n-k))
likelihood = lambda fact, prior, n, k: (fact*prior**k)*(1-prior)**(n-k)


def plot_decision_network(Id=3.8, Ii=2.6, Io=4.5, g=12, b=34, rmax=60, wid=.21, wdi=.21):

      sns.set(style='white', font_scale=2.)
      f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,13))

      rd1, ri1, dv1 = decision_network(Id=Id, Ii=Ii, Io=4.5, g=g, b=b, rmax=rmax, wid=wid, wdi=wdi)
      rd2, ri2, dv2 = decision_network(Id=Id, Ii=Ii, Io=3.6, g=g, b=b, rmax=rmax, wid=wid, wdi=wdi)
      rd3, ri3, dv3 = decision_network(Id=Id, Ii=Ii, Io=2.8, g=g, b=b, rmax=rmax, wid=wid, wdi=wdi)
      data=[[rd1, ri1, dv1, 1, True], [rd2, ri2, dv2, .5, False], [rd3, ri3, dv3, .25, False]]
      xlim=(0,len(rd1))
      for i, dat in enumerate(data):
            dual_space_plot(ax2, ax1, dat[0], dat[1], dat[2], rts=IoRT, alpha=dat[3], isfirst=dat[4], i=i, xlim=xlim)



def decision_network(Id=6, Ii=3, Io=2, wdi=.22, wid=.22, k=.85, si=2.3, dt=.001, tau=.05, tmax=1.5, rmax=70, b=35, g=15, ntrials=10, y=1, Z=20, IoMax=4.5):

      rd = np.zeros(ntp)
      ri = np.zeros(ntp)
      dv = np.zeros(ntp)

      NInput = lambda x, r: rmax/(1+np.exp(-(x-b)/g))-r
      dspace = lambda rd, ri: (rd-ri)/np.sqrt(2)

      Ed=si*np.sqrt(dt/tau)*rs(ntp)
      Ei=si*np.sqrt(dt/tau)*rs(ntp)
      x=200
      rd[:x], ri[:x] = [v[0][:x]+Io+v[1][:x] for v in [[rd,Ed],[ri,Ei]]]

      subZ=True
      IIi, IId = [deepcopy(ii) for ii in [Id, Ii]]
      for i in xrange(x, ntp):

            rd[i] = rd[i-1]+dt/tau*(NInput(Id + y*Io + k*rd[i-1] + -wid*ri[i-1], rd[i-1])) + Ed[i]
            ri[i] = ri[i-1]+dt/tau*(NInput(Ii + y*Io + k*ri[i-1] + -wdi*rd[i-1], ri[i-1])) + Ei[i]

            if dv[i-1]<Z and subZ:
                  dv[i] = dspace(rd[i-1], ri[i-1])
            elif subZ:
                  Id,Ii,Io = -Id*Io, -Ii*Io, Io
                  wdi, wid=0, 0
                  NInput = lambda x, r: Io/(1+np.exp(-(x-b)/g))-r-IoMax
                  subZ=False
            elif not subZ and rd[i]<(IoMax+1):
                  x = len(rd[i:])
                  rd0=hs(rd[:200].tolist()*3)
                  ri0=hs(ri[:200].tolist()*3)
                  rd, ri=hs([rd[:i], rd0]), hs([ri[:i], ri0])
                  break

      return rd, ri, dv[:dv[dv<Z].argmax()]



def dual_space_plot(ax1, ax2, rd, ri, dv, rts=None, Z=20, alpha=1, isfirst=True, xlabel=False, i=0, xlim=None):

      rt=len(dv)

      if isfirst:
            labels=['Direct', 'Indirect']
            ax1.set_xlabel('Time (ms)')
      else:
            labels=[None, None]

      # Neural Space
      ax1.plot(rd, label=labels[0], color=colors[3], alpha=alpha)
      ax1.plot(ri, label=labels[1], color=colors[6], alpha=alpha)
      ax1.vlines(rt, ymin=ri[rt], ymax=rd[rt], color=colors[-2], linestyles='--', alpha=alpha)

      # Decision Space
      ax2.plot(dv, color=colors[-2], alpha=alpha)

      if xlabel:
            ax1.set_xlabel('Time (ms)')

      ax1.set_ylabel('Firing Rate (Hz)')
      ax2.set_ylabel('Decision Evidence ($\Theta$)')
      ax1.set_yticks([0, int(hs([rd, ri]).max())+5])
      ax1.set_yticklabels([])#[0, int(hs([rd, ri]).max())+5])
      ax2.set_yticklabels([])
      ax1.set_xticklabels([])
      ax2.set_xticklabels([])
      ax2.set_ylim(-1,Z)

      ax1.legend(loc=2)
      ax2.hlines(Z, 0, ax2.get_xlim()[1], linestyle='--')

      if rts is not None:
            divider = make_axes_locatable(ax2)
            axx = divider.append_axes("top", size=1.6, pad=0.01, sharex=ax1)
            sns.distplot(rts[0], kde=True, ax=axx, kde_kws={'shade':True, "color": colors[-2], "alpha":.8}, hist=False);
            sns.distplot(rts[1], kde=True, ax=axx, kde_kws={'shade':True, "color": colors[-2], "alpha":.5}, hist=False);
            sns.distplot(rts[2], kde=True, ax=axx, kde_kws={'shade':True, "color": colors[-2], "alpha":.2}, hist=False);
            axx.spines['top'].set_visible(False)
            axx.spines['left'].set_visible(False)
            axx.spines['bottom'].set_visible(False)
            axx.spines['right'].set_visible(False)

            axx.set_xticklabels([]);
            axx.set_yticklabels([])

      f.subplots_adjust(hspace=0.1)
      sns.despine(ax=ax1)
      sns.despine(top=True, ax=ax2)
      if xlim is not None:
            ax1.set_xlim(xlim[0], xlim[1])
            ax2.set_xlim(xlim[0], xlim[1])
            axx.set_xlim(xlim[0], xlim[1])
