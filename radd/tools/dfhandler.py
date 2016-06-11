#!usr/bin/env python
from __future__ import division
from future.utils import listvalues
import pandas as pd
import numpy as np
from numpy import array

class DataHandler(object):

    def __init__(self, model):

        self.model = model
        self.data = model.data
        self.inits = model.inits
        self.idx = model.idx
        self.nidx = model.nidx

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

        idxdf_cols, obsdf_cols, infodf_cols = self.__get_headers__()

        data = self.data
        ncols = len(obsdf_cols)
        nrows = self.nrows
        index = range(nrows)

        self.datdf = self.grpData.apply(self.model.rangl_data).sortlevel(0)
        self.dfvals = [self.datdf.values[i].astype(float) for i in index]

        self.observedDF = pd.DataFrame(np.zeros((nrows, ncols))*np.nan, columns=obsdf_cols, index=index)
        self.observedDF.loc[:, self.groups] = self.datdf.reset_index()[self.groups].values
        self.fits = self.observedDF.copy()

        for rowi in range(nrows):
            self.observedDF.loc[rowi, idxdf_cols[rowi]] = self.dfvals[rowi]

        # GENERATE DF FOR FIT RESULTS
        self.fitinfo = pd.DataFrame(columns=infodf_cols, index=index)


    def __get_headers__(self):

        obsdf_cols = np.hstack([self.groups + ['acc']]).tolist()
        cq = ['c' + str(int(n * 100)) for n in self.percentiles]
        eq = ['e' + str(int(n * 100)) for n in self.percentiles]
        qcols = cq + eq

        if hasattr(self.model, 'ssd'):
            ssd_list = [np.asarray(issd*1000, dtype=np.int) for issd in self.model.ssd]
            idxdf_cols = [['acc'] + issd.tolist() + qcols for issd in ssd_list]
            all_ssds = np.unique(np.hstack(ssd_list))
            obsdf_cols = obsdf_cols + all_ssds.tolist()
        else:
            idxdf_cols = [np.hstack(['acc'], qcols).tolist()]*nrows

        obsdf_cols = obsdf_cols + qcols
        params = np.sort(list(self.inits))
        if not self.model.is_flat:
            dep_keys = list(self.model.pc_map)
            cond_param_names = listvalues(self.model.pc_map)
            params = np.hstack([params, np.squeeze(cond_param_names)]).tolist()
            _ = [params.remove(pname) for pname in dep_keys]
        fit_cols = ['nfev', 'nvary', 'df', 'chi', 'rchi', 'logp', 'AIC', 'BIC', 'cnvrg']
        fitdf_cols = np.hstack([['idx'], params, fit_cols]).tolist()

        return idxdf_cols, obsdf_cols, fitdf_cols


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
