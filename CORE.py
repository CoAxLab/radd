#!/usr/local/bin/env python
from __future__ import division
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from numpy import array
from radd import analyze, theta
from scipy.stats.mstats import mquantiles as mq
from radd.tools.messages import saygo


class RADDCore(object):

    """ Parent class for constructing shared attributes and methods
    of Model & Optimizer objects. Not meant to be used directly.

    Contains methods for building dataframes, generating observed data vectors
    that are entered into cost function during fitting, calculating summary measures
    and weight matrix for weighting residuals during optimization.

    TODO: COMPLETE DOCSTRINGS
    """

    def __init__(self, data=None, kind='dpm', inits=None, fit_on='average', depends_on=None, niter=50, fit_whole_model=True, tb=None, fit_noise=False, pro_ss=False, dynamic='hyp', split=50, include_zero_rts=False, *args, **kws):

        self.data = data
        self.kind = kind
        self.fit_on = fit_on
        self.dynamic = dynamic
        self.fit_whole_model = fit_whole_model

        # BASIC MODEL STRUCTURE (kind)
        if 'pro' in self.kind:
            self.nudge_dir = 'up'
            self.data_style = 'pro'
            if depends_on is None:
                depends_on = {'v': 'pGo'}
            self.ssd = np.array([.450])
            if tb == None:
                tb = .555
            self.split = split
            if isinstance(self.split, int):
                self.nrt_cond = 2
            elif isinstance(self.split, list):
                self.nrt_cond = len(self.split)

            self.pGo = sorted(self.data.pGo.unique())
            self.include_zero_rts = include_zero_rts
        else:
            self.nudge_dir = 'down'
            self.data_style = 're'
            if tb == None:
                tb = .65
            if depends_on is None:
                depends_on = {'v': 'Cond'}
            ssd = data[data.ttype == "stop"].ssd.unique()
            self.pGo = len(data[data.ttype == 'go']) / len(data)
            self.delays = sorted(ssd.astype(np.int))
            self.ssd = array(self.delays) * .001

        # CONDITIONAL PARAMETERS
        self.depends_on = depends_on
        self.cond = self.depends_on.values()[0]
        self.labels = np.sort(data[self.cond].unique())
        self.ncond = len(self.labels)

        # index to split pro. rts during fit
        # if split!=None (is set during prep in
        # analyze.__make_proRT_conds__())
        self.rt_cix = None
        # Get timebound
        if tb != None:
            self.tb = tb
        else:
            self.tb = data[data.response == 1].rt.max()
        # PARAMETER INITIALIZATION
        if inits is None:
            self.__get_default_inits__()
        else:
            self.inits = inits
        self.__check_inits__(fit_noise=fit_noise, pro_ss=pro_ss)

        # DEFINE ITERABLES
        if self.fit_on == 'bootstrap':
            self.indx = range(niter)
        else:
            self.indx = list(data.idx.unique())

    def set_fitparams(self, ntrials=10000, tol=1.e-5, maxfev=5000, niter=500, disp=True, prob=np.array([.1, .3, .5, .7, .9]), get_params=False, **kwgs):

        if not hasattr(self, 'fitparams'):
            self.fitparams = {}
        self.fitparams = {'ntrials': ntrials, 'maxfev': maxfev,
                          'disp': disp, 'tol': tol, 'niter': niter, 'nudge_dir': self.nudge_dir,
                          'prob': prob, 'tb': self.tb, 'ssd': self.ssd, 'flat_y': self.flat_y,
                          'avg_y': self.avg_y, 'avg_wts': self.avg_wts, 'ncond': self.ncond,
                          'pGo': self.pGo, 'flat_wts': self.flat_wts, 'depends_on': self.depends_on,
                          'dynamic': self.dynamic, 'fit_whole_model': self.fit_whole_model,
                          'rt_cix': self.rt_cix, 'data_style': self.data_style, 'labels': self.labels}
        if get_params:
            return self.fitparams

    def set_basinparams(self, nrand_inits=2, interval=10, niter=40, stepsize=.05, nsuccess=20, is_flat=True, method='TNC', btol=1.e-3, maxiter=20, get_params=False, bdisp=False):

        if not hasattr(self, 'basinparams'):
            self.basinparams = {}

        self.basinparams = {'nrand_inits': nrand_inits, 'interval': interval, 'niter': niter, 'stepsize': stepsize, 'nsuccess': nsuccess, 'method': 'TNC', 'tol': btol, 'maxiter': maxiter, 'disp': bdisp}
        if get_params:
            return self.basinparams


    def __extract_popt_fitinfo__(self, finfo=None):
        """ takes optimized dict or DF of vectorized parameters and
        returns dict with only depends_on.keys() containing vectorized vals.
        Is accessed by fit.Optimizer objects after optimization routine.

        ::Arguments::
              finfo (dict/DF):
                    finfo is dict if self.fit_on is 'average'
                    and DF if self.fit_on is 'subjects' or 'bootstrap'
                    contains optimized parameters
        ::Returns::
              popt (dict):
                    dict with only depends_on.keys() containing
                    vectorized vals
        """
        if finfo is None:
            try:
                finfo = self.fitinfo.mean()
            except Exception:
                finfo = self.fitinfo
        finfo = dict(deepcopy(finfo))
        popt = dict(deepcopy(self.inits))
        pc_map = self.pc_map
        for pkey in popt.keys():
            if pkey in self.depends_on.keys():
                popt[pkey] = np.array([finfo[pc] for pc in pc_map[pkey]])
                continue
            popt[pkey] = finfo[pkey]

        return popt

    def rangl_data(self, data, kind='dpm', prob=np.array([.1, .3, .5, .7, .9])):
        """ wrapper for analyze.rangl_data
        """
        rangled = analyze.rangl_data(
            data, data_style=self.data_style, kind=kind, prob=prob, tb=self.tb)
        return rangled

    def resample_data(self, data):
        """ wrapper for analyze.resample_data
        """
        resampled = analyze.resample_data(data, n=100, data_style=self.data_style, tb=self.tb, kind=self.kind)
        return resampled

    def rt_quantiles(self, data, split_col='HL', prob=np.array([.1, .3, .5, .7, .9])):
        """ wrapper for analyze.rt_quantiles
        """
        if not hasattr(self, "prort_conds_prepared"):
            self.__make_proRT_conds__()
        rtq = analyze.rt_quantiles(data, include_zero_rts=self.include_zero_rts, split_col=split_col, prob=prob, nrt_cond=self.nrt_cond, tb=self.tb)
        return rtq

    def assess_fit(self, finfo=None):
        """ wrapper for analyze.assess_fit calculates and stores
        rchi, AIC, BIC and other fit statistics
        """
        return analyze.assess_fit(finfo)

    def params_io(self, p={}, io='w', iostr='popt'):
        """ read // write parameters dictionaries
        """
        if io == 'w':
            pd.Series(p).to_csv(''.join([iostr, '.csv']))
        elif io == 'r':
            ps = pd.read_csv(''.join([iostr, '.csv']), header=None)
            p = dict(zip(ps[0], ps[1]))
            return p

    def fits_io(self, fits=[], io='w', iostr='fits'):
        """ read // write y, wts, yhat arrays
        """
        if io == 'w':
            if np.ndim(self.avg_y) > 1:
                y = self.avg_y.flatten()
                fits = fits.flatten()
            else:
                y = self.avg_y
            index = np.arange(len(fits))
            df = pd.DataFrame({'y': y, 'wts': self.avg_wts, 'yhat': fits}, index=index)
            df.to_csv(''.join([iostr, '.csv']))

        elif io == 'r':
            df = pd.read_csv(''.join([iostr, '.csv']), index_col=0)
            return df

    def __nudge_params__(self, p, lim=(.98, 1.02)):
        """
        nudge params so not all initialized at same val
        """

        if self.fitparams['nudge_dir'] == 'up':
            lim = [np.min(lim), np.max(lim)]
        else:
            lim = [np.max(lim), np.min(lim)]

        for pkey in p.keys():
            bump = np.linspace(lim[0], lim[1], self.ncond)
            if pkey in self.pc_map.keys():
                if pkey in ['a', 'tr']:
                    bump = bump[::-1]
                p[pkey] = p[pkey] * bump
        return p

    def slice_bounds_global(self, inits, pfit):

        b = theta.get_bounds(kind=self.kind, tb=self.fitparams['tb'])
        pclists = []
        for pkey, pcl in self.pc_map.items():
            pfit.remove(pkey)
            pfit.extend(pcl)
            # add bounds for all
            # pkey condition params
            for pkc in pcl:
                inits[pkc] = inits[pkey]
                b[pkc] = b[pkey]

        pbounds = tuple([slice(b[pk][0], b[pk][1], .25 * np.max(np.abs(b[pk]))) for pk in pfit])
        params = tuple([inits[pk] for pk in pfit])

        return pbounds, params

    def __make_dataframes__(self, qp_cols):
        """ Generates the following dataframes and arrays:
        ::Arguments::
              qp_cols:
                    header for observed/fits dataframes
        ::Returns::
              None (All dataframes and vectors are stored in dict and assigned
              as <dframes> attr)
        observed (DF):
              Contains Prob and RT quant. for each subject
              used to calc. cost fx weights
        fits (DF):
              empty DF shaped like observed DF, used to store simulated
              predictions of the optimized model
        fitinfo (DF):
              stores all opt. parameter values and model fit statistics
        dat (ndarray):
              contains all subject/boot. y vectors entered into costfx
        avg_y (ndarray):
              average y vector for each condition entered into costfx
        flat_y (1d array):
              average y vector used to initialize parameters prior to fitting
              conditional model. calculated collapsing across conditions
        """

        cond = self.cond
        ncond = self.ncond
        data = self.data
        indx = self.indx
        labels = self.labels

        ic_grp = data.groupby(['idx', cond])
        c_grp = data.groupby([cond])
        i_grp = data.groupby(['idx'])

        if self.fit_on == 'bootstrap':
            self.dat = np.vstack([i_grp.apply(self.resample_data, kind=self.kind).values for i in indx]).unstack()

        if self.data_style == 're':
            datdf = ic_grp.apply(self.rangl_data, kind=self.kind).unstack().unstack()
            indxx = pd.Series(indx * ncond, name='idx')
            obs = pd.DataFrame(np.vstack(datdf.values), columns=qp_cols[1:], index=indxx)
            obs.insert(0, qp_cols[0], np.sort(labels * len(indx)))

            self.observed = obs.sort_index().reset_index()
            self.avg_y = self.observed.groupby(cond).mean().values[:, 1:]
            self.flat_y = self.observed.mean().values[1:]
            axis0, axis2 = self.observed.loc[:, qp_cols[1]:].values.shape
            dat = self.observed.loc[:, qp_cols[1]:].values.reshape(int(axis0 / ncond), ncond, axis2)
            fits = pd.DataFrame(np.zeros((len(indxx), len(qp_cols))), columns=qp_cols, index=indxx)

        elif self.data_style == 'pro':
            datdf = ic_grp.apply(self.rangl_data, kind=self.kind).unstack()
            rtdat = pd.DataFrame(np.vstack(i_grp.apply(self.rt_quantiles).values), index=indx)
            rtdat[rtdat < .1] = np.nan
            rts_flat = pd.DataFrame(np.vstack(i_grp.apply(self.rt_quantiles, split_col=None).values), index=indx)
            self.observed = pd.concat([datdf, rtdat], axis=1)
            self.observed.columns = qp_cols
            self.avg_y = self.observed.mean().values
            self.flat_y = np.append(datdf.mean().mean(), rts_flat.mean())
            dat = self.observed.values.reshape((len(indx), len(qp_cols)))
            fits = pd.DataFrame(np.zeros_like(dat), columns=qp_cols, index=indx)

        fitinfo = pd.DataFrame(columns=self.infolabels, index=indx)
        self.dframes = {'data': self.data, 'flat_y': self.flat_y, 'avg_y': self.avg_y,                        'fitinfo': fitinfo, 'fits': fits, 'observed': self.observed, 'dat': dat}

    def __prep_basin_data__(self):

        fp = self.fitparams
        if 'pro' in self.kind:
            # regroup y vectors into conditions
            wtsp = fp['avg_wts'].flatten()[:fp['ncond']].reshape(2, int(fp['ncond'] / 2))
            nogo = self.avg_y.flatten()[:fp['ncond']].reshape(2, int(fp['ncond'] / 2))
            rts = self.avg_y[fp['ncond']:].reshape(2, 5)
            wtsq = fp['avg_wts'][fp['ncond']:].reshape(2, 5)

            upper = [np.append(ng, rts[0]) for ng in nogo[0]]
            lower = [np.append(ng, rts[1]) for ng in nogo[1]]
            upperwts = [np.append(wtp, wtsq[0]) for wtp in wtsp[0]]
            lowerwts = [np.append(wtp, wtsq[1]) for wtp in wtsp[1]]

            cond_data = np.vstack([upper, lower])
            cond_wts = np.vstack([upperwts, lowerwts])
        else:
            cond_data = self.avg_y
            cond_wts = self.avg_wts
        return cond_data, cond_wts

    def get_wts(self):
        """ wtc: weights applied to correct rt quantiles in cost f(x)
              * P(R | No SSD)j * sdj(.5Qj, ... .95Qj)
        wte: weight applied to error rt quantiles in cost f(x)
              * P(R | SSD) * sd(.5eQ, ... .95eQ)
        """

        nc = self.ncond
        cond = self.cond
        nssd = self.nssd
        if self.data_style == 're':
            rprob = self.data.groupby(['ttype', 'Cond']).mean()['response']
            qwts = analyze.reactive_mj_quanterr(df=self.data, cond=cond)
            # multiply by prob. of response on cor. and err. trials
            wtd_qwts = np.vstack(rprob.unstack().unstack().values) * qwts.reshape(nc * 2, 5)
            wtd_qwts = wtd_qwts.reshape(nc, 10)

            perr = self.observed.groupby(cond).std().iloc[:, 1:7].values
            pwts = np.median(perr, axis=1)[:, None] / perr
            self.avg_wts = np.array([np.append(pw, qw) for pw, qw in zip(pwts, wtd_qwts)]).flatten()
            self.flat_wts = self.avg_wts.reshape(nc, 16).mean(axis=0)

        elif self.data_style == 'pro':
            upper = self.data[self.data['HL'] == 1].mean()['response']
            lower = self.data[self.data['HL'] == 2].mean()['response']
            qwts = analyze.proactive_mj_quanterr(df=self.data, split='HL', tb=self.tb)
            wtqwts = np.hstack(np.array([upper, lower])[:, None] * qwts)
            perr = self.observed.std()[:nc]
            pwts = np.median(perr) / perr
            self.avg_wts = np.hstack([pwts, wtqwts])
            nogo = self.avg_wts[:nc].mean()
            quant = self.avg_wts[nc:].reshape(2, 5).mean(axis=0)
            self.flat_wts = np.hstack([nogo, quant])
        self.avg_wts, self.flat_wts = analyze.ensure_numerical_wts(self.avg_wts, self.flat_wts)

    def __remove_outliers__(self, sd=1.5, verbose=False):
        self.data = analyze.remove_outliers(self.data, sd=sd, verbose=verbose)

    def __get_header__(self, params=None, data_style='re', labels=[], prob=np.array([.1, .3, .5, .7, .9]), cond='Cond'):
        if not hasattr(self, 'delays'):
            self.delays = self.ssd
        qp_cols = analyze.get_header(params=params, data_style=self.data_style, labels=self.labels, prob=prob, delays=self.delays, cond=cond)
        if params is not None:
            self.infolabels = qp_cols[1]
        return qp_cols[0]

    def __get_default_inits__(self):
        self.inits = theta.get_default_inits(kind=self.kind, dynamic=self.dynamic, depends_on=self.depends_on)

    def __get_optimized_params__(self, include_ss=False, fit_noise=False):
        params = theta.get_optimized_params(kind=self.kind, dynamic=self.dynamic, depends_on=self.depends_on)
        return params

    def __check_inits__(self, inits=None, pro_ss=False, fit_noise=False):
        if inits is None:
            inits = dict(deepcopy(self.inits))
        self.inits = theta.check_inits(inits=inits, pdep=self.depends_on.keys(), kind=self.kind, pro_ss=pro_ss, fit_noise=fit_noise)

    def mean_pgo_rts(self, p={}, return_vals=True):
        """ Simulate proactive model and calculate mean RTs
        for all conditions rather than collapse across high and low
        """
        if not hasattr(self, 'simulator'):
            self.make_simulator()
        DVg = self.simulator.simulate_pro(p, analyze=False)
        gdec = self.simulator.resp_up(DVg, p['a'])
        rt = self.simulator.RT(p['tr'], gdec)

        mu = np.nanmean(rt, axis=1)
        ci = pd.DataFrame(rt.T).sem() * 1.96
        std = pd.DataFrame(rt.T).std()

        self.pgo_rts = {'mu': mu, 'ci': ci, 'std': std}
        if return_vals:
            return self.pgo_rts

    def __make_proRT_conds__(self):
        self.data, self.rt_cix = analyze.make_proRT_conds(self.data, self.split)
        self.prort_conds_prepared = True

    def __rename_bad_cols__(self):
        self.data = analyze.rename_bad_cols(self.data)
