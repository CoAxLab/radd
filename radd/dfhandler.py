#!usr/bin/env python
from __future__ import division
from future.utils import listvalues
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy import array
from radd.tools import analyze
from scipy.stats.mstats_extras import mjci


class DataHandler(object):

    def __init__(self, model, max_wt=10):

        self.model = model
        self.data = model.data
        self.inits = model.inits
        self.idx = model.idx
        self.nidx = model.nidx

        self.max_wt = max_wt

        self.kind = model.kind
        self.fit_on = model.fit_on
        self.percentiles = model.percentiles
        self.groups = model.groups
        self.depends_on = model.depends_on

        self.conds = model.conds
        self.nconds = model.nconds
        self.levels = model.levels
        self.nlevels = model.nlevels

        self.nrows = self.nidx * self.nlevels * self.nconds
        self.grpData = self.data.groupby(self.groups)


    def make_dataframes(self):
        """ Generates the following dataframes and arrays:
        ::Arguments::
              qp_cols:
                    header for observed/fits dataframes
        ::Returns::
              None (All dataframes and vectors are stored in dict and assigned
              as <dframes> attr)

        observed (list of ndarrays):
            list containing ndarrays (ncond x n_observed_data_points) entered into thec
            costfx during fitting. If self.fit_on=='average', self.observed will
            contain a single ndarray (ncond x average_data). If self.fit_on=='subjects',
            self.observed will contain 1 ndarray for each subject and fits will be performed
            iteratively
        observedDF (DF):
              Contains Prob and RT quant. for each subject
              used to calc. cost fx weights
        fits (DF):
              empty DF shaped like observed DF, used to store simulated
              predictions of the optimized model
        fitinfo (DF):
              stores all opt. parameter values and model fit statistics
        """

        self.make_observed_groupDFs()

        if self.fit_on=='subjects':
            idxdf = lambda idx: self.observedDF[self.observedDF['idx']==idx]
            observed = [idxdf(idx).dropna(axis=1).loc[:, 'acc':].values for idx in self.idx]
            observedflat = [idxdf(idx).dropna(axis=1).mean().loc['acc':].values for idx in self.idx]

        elif self.fit_on=='average':
            observed = [self.observedDF.groupby(self.conds).mean().loc[:, 'acc':].values]
            observedflat = [self.observedDF.mean().loc['acc':].values]

        # elif self.fit_on == 'bootstrap':
        #     observed = self.idxData.apply(self.resample_data)

        # Get rid of any extra dimensions
        self.observed = [obs_i.squeeze() for obs_i in observed]
        self.observed_flat = [obsF_i.squeeze() for obsF_i in observedflat]


    def make_observed_groupDFs(self):
        """ concatenate all idx data vectors into a dataframe
        """

        #idx_cols, obsdf_cols, finfo_cols = self.__make_headers__()
        masterDF_header = self.__make_headers__()
        data = self.data
        ncols = len(masterDF_header)
        nrows = self.nrows
        index = range(nrows)
        nandata = np.zeros((nrows, ncols))*np.nan

        self.datdf = self.grpData.apply(self.model.rangl_data).sortlevel(0)
        self.dfvals = [self.datdf.values[i].astype(float) for i in index]
        self.observedDF = pd.DataFrame(nandata, columns=masterDF_header, index=index)
        self.observedDF.loc[:, self.groups] = self.datdf.reset_index()[self.groups].values
        self.fits = self.observedDF.copy()
        self.wtsDF = self.observedDF.copy()

        for rowi in range(nrows):
            self.observedDF.loc[rowi, self.idx_cols[rowi]] = self.dfvals[rowi]

        # GENERATE DF FOR FIT RESULTS
        self.fitinfo = pd.DataFrame(columns=self.finfo_cols, index=index)
        idx_qwts, idx_pwts = self.estimate_cost_weights()
        # fill wtsDF with idx quantile wts (ratios)
        self.wtsDF.loc[:, self.q_cols] = idx_qwts
        # fill wtsDF with resp. probability wts (ratios)
        self.wtsDF.loc[:, self.p_cols] = idx_pwts

    def __make_headers__(self, ssd_list=None):

        gcols = self.groups
        if hasattr(self.model, 'ssd'):
            ssd_list = self.model.set_ssd(scale=1, get=True)

        self.make_idx_cols(ssd_list)
        masterDF_header = self.groups + self.p_cols + self.q_cols
        return masterDF_header

    def estimate_cost_weights(self):
        """ calculate weights using observed variability
        across subjects (model.observedDF)
        """

        data = self.data
        nsplits = self.nlevels * self.nconds
        percents = self.percentiles
        nquant = percents.size

        # estimate quantile weights
        idx_qwts = self.mj_quanterr()
        # estimate resp. probability weights
        if self.model.fit_on=='subjects':
            idx_pwts = self.pwts_idx_error_calc()
        elif self.model.fit_on=='average':
            # repeat for all rows in wtsDF (idx x ncond x nlevels)
            idx_pwts = self.pwts_group_error_calc()

        return idx_qwts, idx_pwts


    def mj_quanterr(self):
        """ calculates weight vectors for reactive RT quantiles by
        first estimating the SEM of RT quantiles for corr. and err. responses.
        (using Maritz-Jarrett estimatation: scipy.stats.mstats_extras.mjci).
        Then representing these variances as ratios.
        e.g.
              QSEM = mjci(rtvectors)
              wts = median(QSEM)/QSEM
        """
        idx_mjci = lambda x: mjci(x.rt, prob=self.percentiles)
        nidx = self.nidx
        nquant = self.percentiles.size
        groups = self.groups
        nsplits = self.nlevels * self.nconds
        # get all trials with response recorded
        godf = self.data.query('response==1')
        # sort by ttype first so go(acc==1) occurs before stop(acc==0)
        ttype_ordered_groups = np.hstack([groups, 'ttype', 'acc']).tolist()
        godf_grouped = godf.groupby(ttype_ordered_groups)
        # apply self.idx_mjci() to estimate quantile CI's
        quant_err = np.vstack(godf_grouped.apply(idx_mjci).values)
        # reshape [nidx   x   ncond * nquant * nacc]
        idx_qerr = quant_err.reshape(nidx, nsplits * nquant * 2)
        # calculate subject median across all conditions quantiles and accuracy
        # this implicitly accounts for n_obs as the mjci estimate of sem will be lower
        # for conditions with greater number of observations (i.e., more acc==1 in godf)
        idx_medians = np.nanmedian(idx_qerr, axis=1)
        idx_qratio = idx_medians[:, None] / idx_qerr
        # set extreme values to max_wt arg val
        idx_qratio[idx_qratio >= self.max_wt] = self.max_wt
        # reshape to fit in wtsDF[:, q_cols]
        return idx_qratio.reshape(nidx * nsplits, nquant * 2)

    def pwts_group_error_calc(self):
        """ get stdev across subjects (and any conds) in observedDF
        weight perr by inverse of counts for each resp. probability measure
        """
        groupedDF = self.observedDF.groupby(self.conds)
        perr = groupedDF.agg(np.nanstd).loc[:, self.p_cols].values
        counts = groupedDF.count().loc[:, self.p_cols].values
        nsplits = self.nlevels * self.nconds
        ndata = len(self.p_cols)
        # replace stdev of 0 with next smallest value in vector
        perr[perr==0.] = perr[perr>0.].min()
        p_wt_bycount = perr * (1./counts)
        # set wts equal to ratio --> median_perr / all_perr_values
        pwts_ratio = np.nanmedian(p_wt_bycount, axis=1)[:, None] / p_wt_bycount
        # set extreme values to max_wt arg val
        pwts_ratio[pwts_ratio >= self.max_wt] = self.max_wt
        # shape pwts_ratio to conform to wtsDF
        idx_pwts = np.array([pwts_ratio]*self.nidx)
        return idx_pwts.reshape(self.nidx * nsplits, ndata)

    def pwts_idx_error_calc(self, extra_group=None):
        """ count number of observed responses across levels transform into ratios
        (np.median(counts_at_each_level) / counts_at_each_level) for weight
        subject-level p(response) values in cost function.

        ::Arguments::
            df (DataFrame): group-level dataframe
            var (str): column header for variable to count responses
            conds (list): depends_on.values()
        """
        df = self.data.copy()
        if hasattr(self.model, 'ssd'):
            df = df[df.ttype == 'stop'].copy()
            extra_group = 'ssd'

        def calc_pwts(df_i):
            countdf = df_i.groupby(extra_group).count()
            # get counts across levels except for last (correct Go//SSD==1000)
            lvl_counts = countdf.reset_index()['response'].values
            # calculate count ratio for all values
            varwts = np.median(lvl_counts)/lvl_counts
            # append 1 to front as weight of correct P(Go|No_SS)
            acc_var_wts = np.append(1., varwts)
            return acc_var_wts

        ic_grp = df.groupby(self.groups)
        idx_pwts = [calc_pwts(cdf) for c, cdf in ic_grp]
        return idx_pwts


    def make_idx_cols(self, ssd_list=None):
        """ make idx-specific headers in event of missing data
        make all other headers if not done yet
        """
        if not hasattr(self, 'q_cols'):
            self.make_q_cols()
        if not hasattr(self, 'p_cols'):
            self.make_p_cols(ssd_list)
        if not hasattr(self, 'finfo_cols'):
            self.make_finfo_cols()
        idx_pcols = list(self.p_cols[0])*self.nrows
        idx_qcols = list(self.q_cols)*self.nrows
        if ssd_list:
            self.idx_cols = [[self.p_cols[0]] + issd + self.q_cols for issd in ssd_list]
        else:
            self.idx_cols = [self.p_cols + self.q_cols]*self.nrows

    def make_q_cols(self):
        cq = ['c' + str(int(n * 100)) for n in self.percentiles]
        eq = ['e' + str(int(n * 100)) for n in self.percentiles]
        self.q_cols = cq + eq

    def make_p_cols(self, ssd_list=None):
        self.p_cols = ['acc']
        if ssd_list:
            self.p_cols = self.p_cols + np.unique(ssd_list).tolist()

    def make_finfo_cols(self):
        params = np.sort(list(self.inits))
        if not self.model.is_flat:
            dep_keys = list(self.model.pc_map)
            cond_param_names = listvalues(self.model.pc_map)
            params = np.hstack([params, np.squeeze(cond_param_names)]).tolist()
            _ = [params.remove(pname) for pname in dep_keys]
        fit_cols = ['nfev', 'nvary', 'df', 'chi', 'rchi', 'logp', 'AIC', 'BIC', 'cnvrg']
        self.finfo_cols = np.hstack([['idx'], params, fit_cols]).tolist()


    def finfo_to_params(self, finfo='./finfo.csv', pc_map=None):

        pnames = set(['a', 'tr', 'v', 'ssv', 'z', 'xb', 'si', 'sso'])
        if isinstance(finfo, str):
            finfo = pd.read_csv(finfo, header=None, names=['id', 'vals'], index_col=0)
            finfo = pd.Series(finfo.to_dict()['vals'])
        elif isinstance(finfo, dict):
            finfo = pd.Series(finfo)
        elif isinstance(finfo, pd.Series):
            pass

        plist = list(pnames.intersection(finfo.index))
        params = {k: finfo[k] for k in plist}

        if pc_map != None:
            for pkey, pclist in pc_map.items():
                params[pkey] = array([params[pkc] for pkc in pclist])
        return params


    def rwr(self, X, get_index=False, n=None):
        """
        Modified from http://nbviewer.ipython.org/gist/aflaxman/6871948
        """
        if isinstance(X, pd.Series):
            X = X.copy()
            X.index = range(len(X.index))
        if n == None:
            n = len(X)
        resample_i = np.floor(np.random.rand(n) * len(X)).astype(int)
        X_resample = (X[resample_i])
        if get_index:
            return resample_i
        else:
            return X_resample


    def resample_data(self, n=120, groups=['ssd']):
        """ generates n resampled datasets using rwr()
        for bootstrapping model fits
        """

        df = self.data.copy()
        tb = self.tb

        bootlist = list()
        if n == None:
            n = len(df)

        for level, level_df in df.groupby(groups):
            boots = level_df.reset_index(drop=True)
            orig_ix = np.asarray(boots.index[:])
            resampled_ix = self.rwr(orig_ix, get_index=True, n=n)
            bootdf = level_df.irow(resampled_ix)
            bootlist.append(bootdf)

        # concatenate and return all resampled conditions
        return self.model.rangl_data(pd.concat(bootlist))


    def extract_popt_fitinfo(self, finfo=None):
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
                finfo = self.model.fitinfo.mean()
            except Exception:
                finfo = self.model.fitinfo

        finfo = dict(deepcopy(finfo))
        popt = dict(deepcopy(self.inits))
        pc_map = self.model.pc_map

        for pkey in list(popt):
            if pkey in list(self.depends_on):
                popt[pkey] = np.array([finfo[pc] for pc in pc_map[pkey]])
                continue
            popt[pkey] = finfo[pkey]

        return popt
