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
sp=utils.style_params()
colors=sp['colors']
greens=sp['greens']; purples=sp['purples']
blues=sp['blues']; reds=sp['reds']
sns.set_style('ticks', rc=sp['style'])
sns.set_context('paper', rc=sp['context'])

def sigmoid(p,x):
        x0,y0,c,k=p
        y = c / (1 + np.exp(k*(x-x0))) + y0
        return y

def residuals(p,x,y):
        return y - sigmoid(p,x)

def res(arr,lower=0.0,upper=1.0):
        arr=arr.copy()
        if lower>upper: lower,upper=upper,lower
        arr -= arr.min()
        arr *= (upper-lower)/arr.max()
        arr += lower
        return arr

def scurves(lines=[], task='ssRe', linestyles=[], pstop=.5, sxdata=None, pse_only=False, labels=[], colors=[], yerr=[], all_solid=False, title='stop_curves', plot_data=False):

        if len(lines[0])==6:
                task='pro'
        if linestyles==[]:
                linestyles=['-']*len(lines)
	style_params=utils.style_params(style='ticks', context='paper')
	sns.set_style(rc=style_params['style'])
	sns.set_context(rc=style_params['context'])
	sns.set_context(rc={'lines.markeredgewidth': 0.1})

        lines=[np.array(line) if type(line)==list else line for line in lines]

	if colors==[]:
		colors=sns.color_palette(sns.color_palette(),len(lines))
	if labels==[]:
		labels=["C%i"%i for i in np.arange(len(lines))]

	f, ax = plt.subplots(1, figsize=(6., 6.5))
	pse=[]; xsim=np.linspace(-5, 110, 10000)

	if 'Re' in task:
		x=np.array([400, 350, 300, 250, 200], dtype='float')
		xsim=np.linspace(15, 50, 10000)
		scale_factor=100
		xxlim=(17, 42)
		xxlabel='SSD (ms)'
		xxticks=np.arange(20, 45, 5)
		xxticklabels=np.arange(200, 450, 50)
		yylabel="P(Stop)"
		c=colors; ms=4.5
	else:
		x=np.array([100, 80, 60, 40, 20, 0], dtype='float')
		xxticks=np.arange(0, 12, 2)#x/scale_factor
		xxticklabels=np.arange(0.0, 1.2, .20)
		xxlim=(-1, 11)
		xxlabel='P(Go)'
		xsim=np.linspace(-5, 11, 10000)
		scale_factor=10
		yylabel="P(NoGo)"
		c=["#545454","#545454"]; ms=4.5

	if all_solid or pse_only:
		linestyles=['-']*len(lines)

	x=res(-x,lower=x[-1]/10, upper=x[0]/10)

	for i, yi in enumerate(lines):

		y=res(yi, lower=yi[-1], upper=yi[0])
		p_guess=(np.mean(x),np.mean(y),.7,.7)
		p, cov, infodict, mesg, ier = optimize.leastsq(
		    residuals,p_guess,args=(x,y), full_output=1, maxfev=5000, ftol=1.e-10)
		x0,y0,c,k=p
		xp = xsim
		pxp=sigmoid(p,xp)
		idx = (np.abs(pxp - pstop)).argmin()

	        #use fitted values to initiate and refit for better accuracy
		next_guess=(np.mean(xsim), xp[idx], .75, 1.)
		p, cov, infodict, mesg, ier = optimize.leastsq(
		    residuals,next_guess,args = (x,y), full_output=1, maxfev=5000, ftol=1.e-10)
		x0,y0,c,k=p
		xp = xsim
		pxp=sigmoid(p,xp)
		idx = (np.abs(pxp - pstop)).argmin()

		# Plot the results
		ax.plot(xp, pxp, linestyle=linestyles[i], lw=4, color=colors[i], label=labels[i])

		if yerr != []:
			ax.errorbar(x, y, yerr=yerr[i], color=colors[i], ecolor=colors[i], capsize=0, lw=0, elinewidth=3)
		else:
			ax.plot(x, y, color=colors[i], lw=0)

		pse.append(xp[idx]/scale_factor)

	if sxdata is not None:
                for i, cond in enumerate(sxdata):
                        for idx in cond.iterrows():
		                ax.plot(x, idx[1].values, marker='o', color=colors[i], ms=ms, mec=colors[i], lw=0, alpha=.1)

	plt.setp(ax, xlim=xxlim, xticks=xxticks, ylim=(-.05, 1.05), yticks=[0, .5, 1])
	ax.set_xticklabels(xxticklabels, fontsize=24); 	ax.set_yticklabels([0.0, .5, 1.0], fontsize=24)
	ax.set_xlabel(xxlabel, fontsize=24); ax.set_ylabel(yylabel, fontsize=24)
	if 1<len(lines)<10:
		ax.legend(loc=0, fontsize=22); sns.despine(); plt.tight_layout()
        if pse_only:
		plt.close('all')
		return pse
	return ax



def prort(bars, lines=[], berr=[], labels=['Data', 'Drift', 'Onset'], colors=['#2d2d2d', '#1abd80', purples[5]], dist_list=[]):

        f, ax=plt.subplots(1, figsize=(5.5,6))
        x=np.arange(20,120,20)

        if berr==[]:
                berr=np.zeros_like(bars)

        ax.bar(x, bars, yerr=berr, width=10, align='center', color=colors[0], error_kw=dict(elinewidth=2, capsize=0, ecolor='k'), label=labels[0], alpha=.95)


        if lines!=[]:

                for i, line in enumerate(lines):
                        ax.plot(x, np.array(line), color=colors[i+1], lw=4, label=labels[i+1])

        ax.legend(loc=0, fontsize=17)
        yylim=[490, 560]
        plt.setp(ax, ylim=(yylim[0], yylim[1]), yticks=yylim, xticks=x)
        ax.set_yticklabels(yylim, fontsize=22)
        ax.set_xticklabels([str(int(xt)) for xt in x], fontsize=22)
        ax.set_xlabel('P(Go)', fontsize=24)
        ax.set_ylabel('RT (ms)', fontsize=24)
        ax.set_xlim(10, 110)

        plt.tight_layout()
        sns.despine()

        return ax


def plot_bar_lines(bars, lines, berr, lerr, labels=['Data', 'Drift', 'Onset'], colors=['#2d2d2d', '#1abd80', purples[5]], dist_list=[], plot_lines=False):

        f, ax=plt.subplots(1, figsize=(6,5))
        x=np.arange(20,120,20)
        #for i, m in enumerate(means):

        if berr==[]:
                berr=np.zeros_like(bars)
        if np.any(bars<1):
                bars=np.array(bars)*1000
                berr=np.array(berr)*1000
        elif np.any(bars<10):
                bars=np.array(bars)*100
                berr=np.array(berr)*100


        ax.bar(x, bars, yerr=berr, width=10, align='center', color=colors[0], error_kw=dict(elinewidth=2, capsize=0, ecolor='k'), label=labels[0], alpha=1)

        if lerr==[]:
                lerr=np.zeros_like(lines)
        if np.any(lines[0]<1):
                lines=[np.array(l)*1000 for l in lines]
                lerr=[np.array(l)*1000 for l in lerr]
                dist_list=[[np.array(l)*1000 for l in d] for d in dist_list]
        elif np.any(lines[0])<10:
                lines=[np.array(l)*100 for l in lines]
                lerr=[np.array(l)*100 for l in lerr]
                dist_list=[[np.array(l)*100 for l in d] for d in dist_list]

        if plot_lines:
                ax.errorbar(x, lines[0], yerr=lerr, color=colors[1], lw=2, label=labels[1])
                ax.errorbar(x, lines[1], yerr=lerr, color=colors[2], lw=2, label=labels[2])

        if dist_list!=[]:
                dist1, dist2=dist_list[:]
                for n1 in dist1:
                        ax.plot(x, n1, color=colors[1], lw=0, alpha=.3, markersize=5, marker='o')
                for n2 in dist2:
                        ax.plot(x, n2, color=colors[2], lw=0, alpha=.3, markersize=5, marker='o')

        if not plot_lines:
                ax.plot(x, n1,color=colors[1], lw=0, alpha=1, markersize=5, marker='o', label=labels[1])
                ax.plot(x, n2, color=colors[2], lw=0, alpha=1, markersize=5, marker='o', label=labels[2])


        ax.legend(loc=0, fontsize=20)

        yylim=np.array([490, 560])
        plt.setp(ax, ylim=(yylim[0],yylim[1]), yticks=yylim, xticks=x)
        ax.set_yticklabels((yylim[0],yylim[1]), fontsize=22)
        ax.set_xticklabels([str(int(xt)) for xt in x], fontsize=22)
        ax.set_ylabel('RT (ms)', fontsize=24)
        ax.set_xlim(10, 110)
        plt.tight_layout()
        sns.despine()

        return ax


def plot_re_bar(y, ysim=[], yerr=[], ysimerr=[], sxdata=None, colors=[], ylabel='RT (ms)', yylim=[550,580], save=False, savepath="./"):

	f, ax = plt.subplots(1, figsize=(5.5, 6.5))
	x=np.array([1,2]); xsim=np.array([1.1, 1.9]);

	if colors==[]:
		colors=['#4E4E8B', '#AD3333', "#074feb", "#f81447"]

	if sxdata is not None:
                offset=-.32
		for i, cond in enumerate(sxdata):
                        for idxdata in cond:
                                if i>0: offset=.32
			        ax.plot(x[i]+offset, idxdata, color=colors[i], marker='o', ms=4.5, linewidth=0, alpha=.35)
        if ysim!=[]:

	        ax.errorbar(xsim, ysim, yerr=ysimerr, marker='o', mfc=None, ms=10, lw=3, color='k', ecolor='k', elinewidth=3)
                ax.plot(xsim[0], ysim[0], marker='o', color=colors[2], ms=14, label='Baseline')
                ax.plot(xsim[1], ysim[1], marker='o', color=colors[3], ms=14, label='Caution')

	ax.bar(x, y, width=.5, align='center', yerr=yerr, color=colors, error_kw=dict(elinewidth=3, ecolor='k'), alpha=.9)

	ax.set_ylabel(ylabel, fontsize=30)
	plt.setp(ax, xticks=x, xlim=(0.6, 2.4), ylim=yylim, yticks=yylim)
	ax.set_yticklabels(yylim, fontsize=30)
	ax.set_xticklabels(['Baseline', 'Caution'], fontsize=30)
	sns.despine(); plt.tight_layout()

	if save:
		plt.savefig(savepath+"Figure2_re_rt_idxfits_graygreen.png", rasterized=True, dpi=600)
		plt.savefig(savepath+"Figure2_re_rt_idxfits_graygreen.svg", rasterized=True, dpi=600)

	return ax


def re_rt_viofits(y, ysim, colors=[], labels=[], save=False, savepath="./", bw='scott'):

	f, ax = plt.subplots(1, figsize=(5.5, 6.5))
	x=np.array([.8, 2.2]); xsim=np.array([1.2, 2.4])
	xxlim=(x[0]-.8, x[-1]+.8); yylim=[490, 630]
	xxticklabels=['Easy', 'Difficult']

	if colors==[]:
		colors=['#4E4E8B', '#AD3333', "#074feb", "#f81447"]
	if labels==[]:
		labels=['Easy', 'Difficult', 'Easy RADD','Difficult RADD']

	sns.violinplot(y, color=colors[:2], alpha=1, positions=x-.15, label=labels[:2], ax=ax, bw=bw)
	sns.violinplot(ysim, color=colors[2:], alpha=1, positions=x+.15, label=labels[2:], ax=ax, bw=bw)

	ax.set_ylabel('RT (ms)', fontsize=30)
	plt.setp(ax, xlim=xxlim, xticks=x, ylim=yylim, yticks=yylim)
	ax.set_yticklabels(yylim, fontsize=30)
	ax.set_xticklabels(xxticklabels, fontsize=30)


	ax.legend(loc=2)
	sns.despine()
	plt.tight_layout()

	if save:
		plt.savefig(savepath+"re_rt_idx_viofits_graygreen.png", rasterized=True, dpi=600)
		plt.savefig(savepath+"re_rt_idx_viofits_graygreen.svg", rasterized=True, dpi=600)

	return ax



def compare_chi(vboot, tboot, colors=[greens[1], purples[-2]]):

        f,ax=plt.subplots(figsize=(7.5, 5))

        xxticks=np.arange(0,1.2,.2)

        v_chivec=vboot.groupby('pGo').mean()['chi'].values
        t_chivec=tboot.groupby('pGo').mean()['chi'].values
        v_chivec_err=vboot.groupby('pGo').sem()['chi'].values*1.96
        t_chivec_err=tboot.groupby('pGo').sem()['chi'].values*1.96

        #ax.errorbar(xxticks, v_chivec, yerr=v_chivec_err, color=reds[5], ecolor='k', label='Drift Bias', lw=3, elinewidth=2)
        #ax.errorbar(xxticks, t_chivec, yerr=t_chivec_err, color=purples[5], ecolor='k', label='Onset Bias', lw=3, elinewidth=2)

        ax.plot(xxticks, v_chivec, color=colors[0], label='Drift Bias', lw=3)
        ax.plot(xxticks, t_chivec, color=colors[1], label='Onset Bias', lw=3)

        vchi_table=vboot.pivot_table('chi', 'n', 'pGo')
        tchi_table=tboot.pivot_table('chi', 'n', 'pGo')

        for n in xrange(len(vchi_table)):
                plt.plot(xxticks, vchi_table.iloc[n, :], color=colors[0], lw=0, marker='o', ms=4, alpha=.25)
                plt.plot(xxticks, tchi_table.iloc[n, :], color=colors[1], lw=0, marker='o', ms=4, alpha=.25)

        yylim=[0.0, .03]
        ax.set_yticks(yylim)
        ax.set_xticks(xxticks)
        ax.set_xticklabels(xxticks, fontsize=22)
        ax.set_yticklabels(yylim, fontsize=22)
        yylabel=ax.set_ylabel('$ \chi^2$', fontsize=28)
        xxlabel=ax.set_xlabel('P(Go)', fontsize=24)
        ax.set_xlim(-.1, 1.1)
        ax.set_ylim(-.001, yylim[1])
        ax.legend(loc=0, fontsize=22)
        plt.tight_layout()
        sns.despine()

        return ax


def plot_chi_dist(chi, bins=15, labels=['Drift', 'Onset'], rug=False, norm=False, colors=['#1abd80', 'Purple'], hist=True, kde=False, hist_kws=None, rug_kws=None, kde_kws=None):

        f,ax=plt.subplots(1, figsize=(5.5, 4.5))
        if kde==True:
                hist=False
        if labels==[]:
                labels=[None, None]


        for i, ichi in enumerate(chi):
                if i>0 and hist_kws is not None:
                        hist_kws['alpha']=hist_kws['alpha']-.2

                ax=sns.distplot(ichi, bins=bins, rug=rug, color=colors[i], label=labels[i], kde=kde, hist=hist, ax=ax, hist_kws=hist_kws, kde_kws=kde_kws, rug_kws=rug_kws, norm_hist=norm)
        if labels[0]:
                ax.legend(fontsize=20)

        ax.set_xlabel('$\chi^2$', fontsize=28)
        ylim=[0, int(ax.get_yticks()[-1])]
        xlim=[0, ax.get_xticks()[-1]]
        plt.setp(ax, xlim=xlim, xticks=xlim, ylim=ylim, yticks=ylim)

        if norm:
                ytl=[0,1]
                yl='Probability'
        else:
                ytl=ylim
                yl='Count'

        ax.set_ylabel(yl, fontsize=24)
        ax.set_yticklabels(ytl, fontsize=22)
        ax.set_xticklabels(xlim, fontsize=22)

        plt.tight_layout()

        sns.despine()

        return ax

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


def bar_line_evs(means=None, err=None, sxdata=None, colors=['#4E4E8B', '#AD3333', '#59ABE3', '#FF6666'], task='ssRe', ylabel='PSE'):

	f = plt.figure(figsize=(5.5, 6.5))
	ax = f.add_subplot(111)
	sns.despine()
	x=np.array([1.1,1.9])
	xsim=np.array([1.2, 1.8])

	ax.bar(x, means[:2], yerr=err[:2], color=colors[0], align='center',
		error_kw=dict(elinewidth=2, ecolor='black'))

	childrenLS=ax.get_children()
	barlist=filter(lambda x: isinstance(x, matplotlib.patches.Rectangle), childrenLS)
	barlist[1].set_color(colors[1])
	#barlist[1].set_label('Difficult Data')

	ax.errorbar(xsim, means[2:4], yerr=err[2:4], marker='o', mfc=None, ms=14, lw=2, color='k',mew=2, mec='k', ecolor='k', elinewidth=2, capsize=0)
	ax.plot(xsim[0], means[2], marker='o', color=colors[2], ms=14, label='Easy Model')
	ax.plot(xsim[1], means[3], marker='o', color=colors[3], ms=14, label='Difficult Model')

	if sxdata !=None:
		for i, cond in enumerate(sxdata):
			for sx in cond:
				ax.plot(i+1, sx, linewidth=0, marker='o', ms=5, alpha=.4, color="#CCCCCC")
	ax.set_xlim(0.5, 2.5)
	ax.set_xticks(x)
	ax.set_ylabel(ylabel, fontsize=26)
	plt.setp(ax, xticklabels=['Easy', 'Difficult'], ylim=[.32, .4], yticks=np.arange(.32,.4,.02))
	ax.set_xticklabels(['Easy', 'Difficult'], fontsize=26)
	plt.setp(ax.get_yticklabels(), fontsize=24)
	ax.legend(loc=2, fontsize=19)
	plt.tight_layout()


	plt.savefig("PSE_%s%s" % (task, ".png"), format='png', dpi=300)
	plt.savefig("PSE_%s%s" % (task, ".svg"), rasterized=True, dpi=300)
	return ax


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



def plot_npsim_traces(DVg=[], DVs=[], theta={}, tau=.0005, tb=.5451, cg='Green', cr='Red' ):

        f,ax=plt.subplots(1,figsize=(8,5))
        tr=theta['tr']; a=theta['a']; z=theta['z']; ssd=theta['ssd']
        for i, igo in enumerate(DVg):
                plt.plot(np.arange(tr, tb, tau), igo, color=cg, alpha=.1, linewidth=.5)
                if i<len(DVs) and DVs!=[]:
                        plt.plot(np.arange(ssd, tb, tau), DVs[i], color=cr, alpha=.1, linewidth=.5)

        plt.setp(ax, xlim=(tr, tb), ylim=(0,a))
        ax.hlines(y=z, xmin=tr, xmax=tb, linewidth=3, linestyle='--', color="#6C7A89")
        sns.despine(top=False,bottom=False, right=True, left=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        return ax



def plot_thalamus_bold_mean_traces(df=pd.DataFrame, outpath="./", pgo=np.arange(0, 1.25, .25), tr=.272, tlist=None, save=False, task='pro'):

        if tlist is None:
                tlist=[tr]*len(pgo)
                bias='v'
        else:
                bias='tr'

        tlist=[int(t*1000) for t in tlist]

        sparams=utils.style_params()
        sns.set_style('ticks', rc=sparams['style'])
        sns.set_context('paper', rc=sparams['context'])
        d=boldfx.trialwise_integrated_BOLD(df, outcomes=['g', 'ng'])
        gdf=d['g']; ngdf=d['ng']

        f,(ax1,ax2)=plt.subplots(1, 2, figsize=(10,6.5), sharey=True)
        gcolors=sns.blend_palette(["#90C695", '#1E824C'], len(pgo))
        scolors=sns.blend_palette(["#E08283", "#C0392B"], len(pgo))

        duration=[]
        maxy=[]
        for pg in pgo:
                if pg<.5:
                        gdf.replace({'pGo':{pg:.5}}, inplace=True)
                elif pg>.5:
                        ngdf.replace({'pGo':{pg:.5}}, inplace=True)
                else: pass

        gdf.reset_index(inplace=True)
        ngdf.reset_index(inplace=True)
        gtlist=[np.mean(tlist[:3]), tlist[-2], tlist[-1]]
        #g_xlist=[]
        #g_thal_list=[]
        for ix, (pg, pgdf) in enumerate(gdf.groupby('pGo')):
                if pg == .5 or pg == 1.0:
                        pass
                else:
                        continue

                pg_thal=pd.DataFrame({i: pd.Series(pgdf['thalamus'][i]) for i in pgdf.index.values}).mean(axis=1).values

                maxy.append(pg_thal.max())
                duration.append(len(pg_thal))

                x=np.arange(gtlist[ix], gtlist[ix] + duration[-1])
                label=str(int(pg*100))

                if pg==.5:
                        label=label+"-"
                ax1.plot(x, pg_thal, color=gcolors[ix], label=label)
                #g_thal_list.append(pg_thal)
        #ng_xlist=[]
        #ng_thal_list=[]
        ngtlist=[tlist[0], tlist[1], np.mean(tlist[2:])]
        for ix, (pg, pgdf) in enumerate(ngdf.groupby('pGo')):
                if pg == .5 or pg == 0.0:
                        pass
                else:
                        continue
                pg_thal=pd.DataFrame({i: pd.Series(pgdf['thalamus'][i]) for i in pgdf.index.values}).mean(axis=1).values

                maxy.append(pg_thal.max())
                duration.append(len(pg_thal))

                x=np.arange(ngtlist[ix], ngtlist[ix] + duration[-1])
                label=str(int(pg*100))

                if pg==.5:
                        label=label+"+"
                ax2.plot(x,pg_thal, color=scolors[ix], label=label)
                #ng_xlist.append(x)
                #ng_thal_list.append(pg_thal)
        #for xx in ng_thal_list:
        #        if len(xx)>np.min(duration):
        #                pass
        #        else:
        #                np.append(xx, np.zeros(np.max(duration) - np.min(duration)))
        #
        #xlong = len(ng_thal_list[1])
        #xshort = len(ng_thal_list[0])
        #yshort = ng_thal_list[0]
        #ng=ax2.fill_between(xx, ng_thal_list[1], ng_thal_list[1]-(ng_thal_list[1]-ng_thal_list[0]), facecolor='RoyalBlue', alpha=0.2)

        xxticks=np.array([tlist[-1], tlist[-1]+np.max(duration)])
        xxticklabels=[str(xx*.001) for xx in xxticks]

	for ax in [ax1, ax2]:
                #ax.hlines(xmin=xxticks[0], xmax=xxticks[-1], y=pg_thal[0], color='Gray', alpha=.8, lw=2.0, linestyle='--')
                #ax.set_xticklabels=[]
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(-.1*ymax, ymax+.1*ymax)
                ax.hlines(xmin=xmin, xmax=xmax, y=0, color='k', alpha=.8, lw=2.0, linestyle='--')
                plt.setp(ax, yticklabels=[], xticklabels=[])
                #ax.set_xticks(xxticks)
                #ax.set_xlim(xxticks[0], xxticks[1])
                if 're' in task:
                        ax1.set_ylabel(r'$\sum_{t=tr}^{t=end}(\theta_{G}\/-\/\theta_{S})$', fontsize=26)
                else:
                        ax1.set_ylabel(r'$\theta_{G}$', fontsize=30)
		ax.legend(loc=0, fontsize=20)

                ax.set_xlabel("Time (ms)", fontsize=23)
        #ax2.set_xticklabels(xxticklabels, fontsize=18)
        #ymin, ymax = ax.get_ylim()
        #ax.set_ylim(-.1*ymax, ymax+.1*ymax)
        #ax.set_ylim(-.10*np.max(maxy), np.max(maxy)+.20*np.max(maxy))


        plt.tight_layout()
	sns.despine()
        if save:
                f.savefig(outpath+"pro_bold_"+bias+"_mean_traces.png", format='png', dpi=600)
                f.savefig(outpath+"pro_bold_"+bias+"_mean_traces.svg", rasterized=True, format='svg', dpi=600)
        return ax



def plot_bold_manuscript(df=pd.DataFrame, pgo=np.arange(0, 1.25, .25), task='pro', bias='drift', title=None, save=False):

        outpath=utils.find_path()+"FinalFigures_New/"

        d=boldfx.trialwise_integrated_BOLD(df, outcomes=['g', 'ng'])
        gdf=d['g']; ngdf=d['ng']
        #return gdf
	go_cond_ix=[pgo[:-3], pgo[-3:-1], [pgo[-1]]]
	ng_cond_ix=[[pgo[0]], [pgo[1]], pgo[2:]]

	g_mu=[np.mean(gdf.ix[gdf['pGo'].isin(ii), 'mag']) for ii in go_cond_ix]
	g_sem=[scp.stats.sem(gdf.ix[gdf['pGo'].isin(ii), 'mag'])*1.96 for ii in go_cond_ix]

	ng_mu=[np.mean(ngdf.ix[ngdf['pGo'].isin(ii), 'mag']) for ii in ng_cond_ix]
	ng_sem=[scp.stats.sem(ngdf.ix[ngdf['pGo'].isin(ii), 'mag'])*1.96 for ii in ng_cond_ix]

	#PLOTTING
	f,ax=plt.subplots(1,figsize=(6.5,5))
	x=np.arange(len(go_cond_ix+ng_cond_ix))

        ax.bar(x[:3], ng_mu, yerr=ng_sem, color='#bd4a4c', ecolor="k", align='center', label="NoGo")
	ax.bar(x[3:], g_mu, yerr=g_sem, color='#009B76', ecolor="k", align='center', label="Go")

        #INJECTED
        if title is not None:
		ax.set_title(title, fontsize=26)

        yylim=[np.min(ng_mu)-15, np.max(g_mu)+15]
        plt.setp(ax, xlim=(-.75, 5.75), xticks=x, ylim=yylim, yticks=yylim, yticklabels=[str(int(yy)) for yy in yylim])
        xxticklabels=["0", "25", "50+", "50-", "75", "100"]
        ax.set_ylabel(r'$\sum_{t=tr}^{t=end}\theta_{G}$', fontsize=26)
	ax.set_xlabel('P(Go)',  fontsize=22)
        ax.set_xticklabels(xxticklabels, fontsize=20)
	plt.setp(ax.get_yticklabels(), fontsize=20)
	plt.legend(loc=0, fontsize=18)

        plt.tight_layout()
	sns.despine()

        if save:
                f.savefig(outpath+"pro_bold_"+bias+"_manscrpt_allresp_150trials.png", format='png', dpi=600)
                f.savefig(outpath+"pro_bold_"+bias+"_manscrpt_allresp_150trials.svg", rasterized=True, format='svg', dpi=600)

        return ax

def plot_allrt_quants(ntrials=2000, bins=20, sim_hist=False, sim_kde=True, emp_hist=True, emp_kde=False):

        redf = pd.read_csv("/Users/kyle/Dropbox/CoAx/SS/Reactive/Re_Data.csv")
        popt = pd.read_csv("/Users/kyle/Dropbox/CoAx/SS/Reactive/BSL/Boot/RADD/rwr_rebsl_popt_radd.csv", index_col=0)
        popt = popt.mean().to_dict()
        popt['pGo']=.5; prob = np.array([.025, .25, .5, .75, .975])
        simdf = fitre.simple_resim(popt, return_all=True, ntrials=ntrials)

        if sim_hist==True:
                norm_hist=True
        else:
                norm_hist=True
        f, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = np.array(axes).flatten()


        emp_cor = redf.query('trial_type=="go" & choice=="go"').rt.values
        sim_cor = simdf.query('trial_type=="go" & choice==1').GoRT.values
        #emp_all = redf.query('choice=="go"').rt.values
        #sim_all = simdf.query('choice==1').GoRT.values
        for i, (ssd, df) in enumerate(redf.groupby('ssd')):

                if ssd>500:
                        break
                if i==0:
                        labels=['Empirical Go','Simulated Go','Empirical ssGo Err','Simulated ssGo Err']
                        yl = axes[i].get_ylim();
                else:
                        labels=[None]*4
                sgo = df.query('trial_type=="stop" & choice=="go"').rt.values-ssd*.001
                sim_errs = simdf[simdf['SSD'].isin([ssd])].query('trial_type=="stop" & choice==1').GoRT.values-(ssd*.001)

                #plot Correct Go
                sns.distplot(emp_cor, bins=bins, hist=emp_hist, kde=emp_kde, norm_hist=norm_hist, label=labels[0], ax=axes[i], color=greens[3], kde_kws={'shade':True})
                sns.distplot(sim_cor, bins=bins, hist=sim_hist, kde=sim_kde, norm_hist=norm_hist, label=labels[1], ax=axes[i], color=greens[2], kde_kws={'shade':True})
                #sns.kdeplot(emp_cor, label=labels[0], cumulative=True, ax=axes[i], color=greens[3], shade=False, alpha=.8)
                #sns.kdeplot(sim_cor, label=labels[1], cumulative=True, ax=axes[i], color=blues[2], shade=False, alpha=.8)

                try:
                        #plot Empirical SS->Go
                        sns.distplot(sgo, bins=bins, hist=emp_hist, kde=emp_kde, norm_hist=norm_hist, ax=axes[i], color=reds[5], kde_kws={'shade':True}, label=labels[2])
                        #sns.kdeplot(sgo, label=labels[2], cumulative=True, ax=axes[i], color=reds[5], shade=False, alpha=.8)
                except Exception: pass

                try:
                        #plot Simulated SS->Go
                        sns.distplot(sim_errs, bins=bins, hist=sim_hist, kde=sim_kde, norm_hist=norm_hist, ax=axes[i], color=reds[2], kde_kws={'shade':True}, label=labels[3])
                        #sns.kdeplot(sim_errs, label=labels[2], cumulative=True, ax=axes[i], color=reds[0], shade=False, alpha=.8)
                except Exception: pass

                yl = axes[i].get_ylim();
                axes[i].set_xlim(.0, .7)
                axes[i].text(.02, yl[1]*.95, str(ssd)+'ms', fontsize=16)
                sns.despine()

        qg = mq(emp_cor, prob = prob)
        qsim_g = mq(sim_cor, prob = prob)

        all_sgo = redf.query('trial_type=="stop" & choice=="go"').rt
        qsg = mq(all_sgo, prob = prob)
        all_sim_errs = simdf.query('trial_type=="stop" & choice==1').GoRT
        qsim_sg = mq(all_sim_errs, prob = prob)

        pdefect = redf.query('trial_type=="go"').mean()['acc']
        axes[-1].plot(qg, prob*pdefect, marker='o', color=greens[3], label='Emp Correct')
        axes[-1].plot(qsim_g, prob*pdefect, marker='o', linestyle='--', color=greens[2], label='Sim Correct')

        axes[-1].plot(qsg, prob*(1-pdefect), marker='o', color=reds[5], label='Emp Err')
        axes[-1].plot(qsim_sg, prob*(1-pdefect), marker='o', linestyle='--', color=reds[2], label='Sim Err')

        axes[-1].set_ylim(-.05, 1); axes[-1].set_xlim(.43, np.max(qg)+.05);

        axes[-1].legend(loc=0)
        axes[0].legend(loc=1)
        #plt.tight_layout()


def ssgo_go_rts(data):
        """
        plot stop-signal respond and go trial RT distributions
        args:
            data (Pandas DataFrame)
        """

        f, axes = plt.subplots(2, 3, figsize=(15, 7))
        axes = np.array(axes).flatten()
        prob = np.arange(.025, .975, .25)

        for i, (ssd, df) in enumerate(data.groupby('ssd')):

                allgo = data.query('choice=="go"')
                sgo = df.query('trial_type=="stop" & choice=="go"')

                if len(sgo)<1:
                        continue

                sns.distplot(sgo.rt.values, bins=20, hist=False, kde_kws={"shade":True, "alpha":.4},
                ax=axes[i], color=reds[5], label=str(ssd))
                sns.distplot(allgo.rt.values, bins=20, hist=False, kde_kws={"shade":True, "alpha":.4},
                ax=axes[i], color=greens[3])
                sns.despine()

        all_sgo = redf.query('trial_type=="stop" & choice=="go"')
        qg = mq(allgo.rt, prob = prob)
        qsg = mq(all_sgo.rt, prob = prob)
        defect_scalar = redf.query('trial_type=="go"').mean()['acc']
        axes[-1].plot(prob, qg, marker='o', color=greens[3])
        axes[-1].plot(prob, qsg, marker='o', color=reds[5])
        plt.tight_layout()
