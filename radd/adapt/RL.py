#!/usr/local/bin/env python
from __future__ import division
import numpy as np
from numpy import array
from numpy.random import sample as rs
from numpy import newaxis as na
import pandas as pd
from scipy.stats import sem
import seaborn as sns
import matplotlib.pyplot as plt

class Environment(object):

    def __init__(self, cards=None, nblocks=2):
        self.cards = cards
        self.nblocks = nblocks
        self.set_environment()

    def set_environment(self):
        self.trials = np.array([self.cards.index.values]*self.nblocks).flatten()
        self.ntrials = len(self.trials)
        self.nalt = len(self.cards.iloc[1:].columns)
        self.names = np.sort(self.cards.columns.values)


class Agent(Environment):

    def __init__(self, ap=.1, an=.1, b=5, cards=None, niter=10, nblocks=2):

        self.updateQ = lambda q, winner, r, A: q[winner][-1] + A*(r - q[winner][-1])
        self.updateP = lambda q, name, b: np.exp(b*q[name][-1])/np.sum([np.exp(b*q[k][-1]) for k in q.keys()])

        super(Agent, self).__init__(cards=cards, nblocks=nblocks)

        if any(np.size(v)>1 for v in [ap, an, b]):
            self.track_params(niter=niter, ap=ap, an=an, b=b)
        else:
            self.set_params(ap=ap, an=an, b=b)


    def set_params(self, ap, an, b):
        self.ap = ap
        self.an = an
        self.b = b
        self.choices = []


    def track_params(self, niter=10, ap=.1, an=.1, b=5):

        blks = np.arange(niter)
        param_names = ['ap', 'an', 'b']
        param_values = [ap, an, b]

        for i, pvalue in enumerate(param_values):
            if not hasattr(pvalue, '__iter__'):
                param_values[i] = [pvalue]

        apos, aneg, beta = param_values
        perm_param_values = list(itertools.product(apos, aneg, beta))

        nsets = len(perm_param_values)
        blocksdf = pd.DataFrame(perm_param_values, columns=param_names, index=np.arange(nsets))

        self.blocksdf = pd.concat([blocksdf]*niter)
        self.blocksdf.reset_index(inplace=True)
        self.blocksdf.rename(columns={'index':'block'}, inplace=True)
        self.blocksdf['P']=0
        self.blocksdf['Q']=0


    def iter_params(self):
        for i in self.blocksdf.index.values:
            ap, an, b = self.blocksdf.loc[i, ['ap', 'an', 'b']].values
            self.set_params(ap=ap, an=an, b=b)
            P, Q = self.simulate_task(return_scores=True)
            self.blocksdf.loc[i, 'P'] = P
            self.blocksdf.loc[i, 'Q'] = Q


    def simulate_task(self, return_scores=False):

        self.qdict={k:[0] for k in self.names}
        self.choice_prob={k:[1./self.nalt] for k in self.names}
        self.likelihood = []
        for t in self.trials:
            rew_vals = self.cards.iloc[t, :].values
            qvals = np.array([self.qdict[name][-1] for name in self.names])
            pvals = np.array([self.choice_prob[name][-1] for name in self.names])
            winner = np.random.choice(np.arange(self.nalt), p=pvals)
            wname = self.names[winner]

            r = rew_vals[winner]
            q = qvals[winner]
            rpe = r - q

            if rpe>0:
                alpha=self.ap
            else:
                alpha=self.an

            Qup = q + (alpha * rpe)
            self.qdict[wname].append(Qup)
            self.choice_prob[wname].append(self.updateP(self.qdict, wname, self.b))
            self.likelihood.append(self.choice_prob[wname][-1])
            for loser in self.names[self.names!=wname]:
                self.qdict[loser].append(self.qdict[loser][-1])
                self.choice_prob[loser].append(self.updateP(self.qdict, loser, self.b))
            self.choices.append(winner)

        if return_scores:
            return self.igt_scores()

    def igt_scores(self):

        ch = np.asarray(self.choices)
        A = ch[ch==0].size
        B = ch[ch==1].size
        C = ch[ch==2].size
        D = ch[ch==3].size
        # payoff (P) score
        P = (C+D) - (A+B)
        # sensitivity (Q) score
        Q = (B+D) - (A+C)
        return [P, Q]

    def plot_summary(self):
        sns.set(style='darkgrid', context='paper', font_scale=1.4)
        titles=['Order of Choices','Number of Choices per Card', 'Change in Q(card)',
            'Change in P(card)']
        f, axes = plt.subplots(2, 2, figsize=(10,8))
        a1, a2, a3, a4 = axes.flatten()
        choice_names = [n.upper() for n in self.names]
        a1.plot(self.choices, lw=0, marker='o')
        a1.set_ylim(-.5, 3.5); a1.set_yticks(np.arange(self.nalt))
        a1.set_yticklabels(choice_names)

        a2.hist(np.asarray(self.choices))
        a2.set_xticks(np.arange(self.nalt))
        a2.set_xticklabels(choice_names)

        for n in self.names:
            a3.plot(self.qdict[n], label=n.upper())
            a4.plot(self.choice_prob[n], label=n.upper())

        a3.legend()
        a4.legend()

        for i, ax in enumerate(axes.flatten()):
            ax.set_title(titles[i])

        f.subplots_adjust(hspace=.35)
