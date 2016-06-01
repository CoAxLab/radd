#!usr/bin/env python
from __future__ import division
import pandas as pd
import numpy as np
from numpy import array

class DataHandler(object):

    def __init__(self, model):

        self.model = model
        self.data = model.data
        self.inits = model.inits

        self.kind = model.kind
        self.fit_on = model.fit_on
        self.percentiles = model.percentiles

        self.conds = model.conds
        self.nconds = model.nconds
        self.levels = model.levels
        self.nlevels = model.nlevels
        self.idx = model.idx
        self.nidx = model.nidx


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
        avg_y (ndarray):
              average y vector for each condition entered into costfx
        flat_y (1d array):
              average y vector used to initialize parameters prior to fitting
              conditional model. calculated collapsing across conditions
        """

        data = self.data

        idx_list = self.idx
        nidx = self.nidx
        conds = self.conds
        nconds = self.nconds
        levels = self.levels
        nlevels = self.nlevels
        nsplits = nlevels * nconds
        nrows = nidx * nlevels * nconds

        idx_conds = np.hstack(['idx', conds]).tolist()
        params = sorted(self.inits.keys())
        self.__get_headers__(params)
        self.make_observed_groupDFs()

        self.avg_y = self.observedDF.groupby(conds).mean().loc[:, 'acc':].values
        self.flat_y = self.observedDF.mean().loc['acc':].values

        if self.fit_on=='subjects':
            idxdf = lambda idx: self.observedDF[self.observedDF['idx']==idx]
            self.observed = [idxdf(idx).dropna(axis=1).loc[:, 'acc':].values for idx in idx_list]
            self.observed_flat = [idxdf(idx).dropna(axis=1).mean()['acc':].values for idx in idx_list]

        elif self.fit_on == 'bootstrap':
            i_grp = data.groupby(['idx'])
            self.observed = np.vstack([i_grp.apply(self.resample_data, kind=self.kind).values for i in idx_list]).unstack()

        elif self.fit_on=='average':
            self.observed = [self.avg_y]
            self.observed_flat = [self.flat_y]


    def make_observed_groupDFs(self):
        """ concatenate all idx data vectors into a dataframe
        """
        data = self.data
        nperc = self.percentiles.size
        all_qp_cols = self.all_qp_cols
        idx_qp_cols = self.idx_qp_cols

        nrows = self.nidx * self.nlevels * self.nconds
        ncols = len(all_qp_cols)

        idx_conds = np.hstack(['idx', self.conds]).tolist()
        ic_grp = self.data.groupby(idx_conds)
        i_grp = self.data.groupby('idx')

        self.datdf = ic_grp.apply(self.model.rangl_data).unstack().unstack().sortlevel(1)
        self.dfvals = [self.datdf.values[i].astype(float) for i in xrange(nrows)]

        self.observedDF = pd.DataFrame(np.zeros((nrows, ncols))*np.nan, columns=all_qp_cols)
        self.observedDF.loc[:, idx_conds] = self.datdf.reset_index()[idx_conds].values

        for rowi in xrange(nrows):
            self.observedDF.loc[rowi, idx_qp_cols[rowi]] = self.dfvals[rowi]

        # GENERATE DF FOR FIT RESULTS
        self.fits = pd.DataFrame(np.zeros((nrows, ncols)), columns=all_qp_cols, index=self.observedDF.index)
        self.fitinfo = pd.DataFrame(columns=self.infolabels, index=self.observedDF.index)


    def get_ssds(self):
        """ set model attr "ssd" as list of np.arrays
        ssds to use when simulating data during optimization
        """
        idx_conds = np.hstack(['idx', self.conds]).tolist()
        stopdf = self.data[self.data.ttype=='stop']
        ic_stopdf = stopdf.groupby(idx_conds)

        self.idx_ssd_ids = [np.sort(df.ssd.unique().astype(int)).tolist() for _,df in ic_stopdf]
        self.all_ssd_ids = np.sort(stopdf.ssd.unique().astype(np.int))

        if self.fit_on == 'subjects':
            self.model.ssd = [np.asarray(ixssd)*.001 for ixssd in self.idx_ssd_ids]
        else:
            self.model.ssd = [self.all_ssd_ids * .001]


    def bin_idx_ssd(self):
        """ bin trials by ssd and collapse trials into nearest SSD if
        trial_count<10.

        Example: Perform for All Subjects in DF
            gDF = DF.groupby('idx')
            newDF = gDF.apply(bin_idx_ssd).reset_index(drop=True)
        """

        df = self.data.copy()

        # insert order column to preserve trial order
        # when rejoining stop and go trials
        df['order']=np.arange(df.shape[0])

        # split by trial type
        dfg = df[df.ttype=='go'].copy()
        dfx = df[df.ttype=='stop'].copy()

        # bin and get trial_counts per ssd
        counts = dfx.groupby('ssd').count().mean(axis=1)
        lowest = counts[counts>=10].index.min()
        highest = counts[counts>=10].index.max()

        # apply cut-offs (collapse ssds with <10 trials)
        dfx.loc[dfx.ssd<lowest, 'ssd']=lowest
        dfx.loc[dfx.ssd>highest, 'ssd']=highest

        # rejoin stop and go, sort, and drop "order" col
        dfnew = pd.concat([dfg, dfx])
        dfnew.sort_values(by='order', inplace=True)
        dfnew.drop('order', axis=1, inplace=True)

        return dfnew


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


    def __get_headers__(self, params=None, percentiles=np.array([.1, .3, .5, .7, .9])):

        conds = self.conds
        nrows = self.nidx * self.nlevels * self.nconds
        idx_conds_acc = np.hstack(['idx', conds, 'acc']).tolist()

        cq = ['c' + str(int(n * 100)) for n in percentiles]
        eq = ['e' + str(int(n * 100)) for n in percentiles]
        qcols = cq + eq

        if 'ssd' in self.data.columns:
            self.get_ssds()
            self.idx_qp_cols = [['acc'] + idx_ssds + qcols for idx_ssds in self.idx_ssd_ids]
            self.all_qp_cols = idx_conds_acc + self.all_ssd_ids.tolist() + qcols
        else:
            self.model.ssd = None
            self.idx_qp_cols = [np.hstack(['acc', qcols]).tolist()]*nrows
            self.all_qp_cols = idx_conds_acc + qcols

        if params:
            info = ['nfev', 'nvary', 'df', 'chi',
                    'rchi', 'logp', 'AIC', 'BIC', 'cnvrg']
            self.infolabels = params + info


    def remove_outliers(df, sd=1.5, verbose=False):

        ssdf = df[df.response == 0]
        godf = df[df.response == 1]
        bound = godf.rt.std() * sd
        rmslow = godf[godf['rt'] < (godf.rt.mean() + bound)]
        clean_go = rmslow[rmslow['rt'] > (godf.rt.mean() - bound)]

        clean = pd.concat([clean_go, ssdf])
        if verbose:
            pct_removed = len(clean) * 1. / len(df)
            print "len(df): %i\nbound: %s \nlen(cleaned): %i\npercent removed: %.5f" % (len(df), str(bound), len(clean), pct_removed)

        return clean


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

        for pkey in popt.keys():
            if pkey in self.depends_on.keys():
                popt[pkey] = np.array([finfo[pc] for pc in pc_map[pkey]])
                continue
            popt[pkey] = finfo[pkey]

        return popt
