from ._iteration import Iteration

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted

from numba import jit  # to speed up some routines

import warnings
warnings.filterwarnings('ignore')


class TClust(ClusterMixin, BaseEstimator):
    """
    General Trimming Approach to Robust Cluster Analysis
    
    TClust searches for k (or less) clusters with different covariance structures in a data matrix x.

    To make the estimation robust, a proportion alpha of observations may be trimmed.
    
    This iterative algorithm initializes k clusters randomly and performs "concentration steps" in order to improve
    the current cluster assignment. The number of maximum concentration steps to be performed is given by iter_max.
    For approximately obtaining the global optimum, the system is initialized n_inits times and concentration
    steps are performed until convergence or ksteps is reached.
    When processing more complex data sets, higher values of n_inits and ksteps have to be specified
    (obviously implying extra computation time). However, if more than half of the iterations do not converge,
    a warning message is issued, indicating that n_inits has to be increased.
    
    The parameter restr_cov_var defines the cluster’s shape restrictions, which are applied to all clusters
    during each iteration.  Options "eigen"/"deter" restrict the ratio between the maximum and minimum
    eigenvalue/determinant of all cluster’s covariance structures to parameter restr_fact.
    Setting restr_fact=1 yields the strongest restriction, forcing all eigenvalues/determinants to be equal
    and so the method looks for similarly scattered (respectively spherical) clusters.
    Option "sigma" is a simpler restriction, which averages the covariance structures during each
    iteration (weighted by cluster sizes) in order to get similar (equal) cluster scatters.
    
    
    .. note::
        The trimmed k-means method (tkmeans) can be obtained by setting parameters restr="eigen",
        restr_fact=1 and equal_weights = True.

    Parameters
    ----------
    k : int, default=2
        The number of clusters
        
    alpha : float, default=0.05
        The proportion of observations to be trimmed
        
    n_inits : int, default=20
        The number of random intializations to be performed
        
    ksteps : int, default=40
        The maximum number of concentration steps to be performed
        
    restr_cov_value : string, default='eigen'
        The type of restriction to be applied on the cluster scatter matrices.
        Valid values are {"eigen", "deter", "sigma"}.
        
    equal_weights : bool, default=False
        Specifying whether equal cluster weights are equal.
        
    zero_tol : float, default=1e-16
        The zero tolerance used.
        
    maxfact_e : float, default=5
        Level of eigen constraints.
    
    maxfact_d : float, default=5
        Level of determinant constraints.
        
    m : float, default=2.
        Fuzzy power parameter.
        
    opt : string, default='hard'
        type of assignment. Accepted values are {'hard', 'mixture', 'fuzzy'}

    sol_ini : object of class Iteration, default=None
        Initial solution provided by the user.
        
    tk : bool, default=False
        Whether to use tkmeans initialization.
        
    verbose : bool, default=True
        Whether to print the progress of the objective function throughout the iterations.

    
    Example
    --------
    >>> from tclust import TClust
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> clustering = TClust(k=2).fit(X)
    >>> clustering.labels_
    array([2, 2, 2, 1, 1, 1], dtype=int64)
    >>> clustering.iter.center
    array([[10.,  2.], [ 1.,  2.]])
    """
    
    def __init__(self, k=2, alpha=0.05, n_inits=20, ksteps=10, equal_weights=False, restr_cov_value='eigen',
                 maxfact_e=5., maxfact_d=5, m=2., zero_tol=1e-16, opt='hard', sol_ini=None,
                 tk=False, verbose=True):
        """
        Initialise
        :param k: Number of clusters initially searched for.
        :param alpha: Proportion of observations to be trimmed.
        :param n_inits: Number of random initializations to be performed.
        :param ksteps: Maximum number of concentration steps to be performed.
                    The concentration steps are stopped whenever two consecutive steps lead to the same data partition.
        :param equal_weights: Specifies whether equal cluster weights (True) or not (False) shall be considered in the concentration and assignment steps.
        :param restr_cov_value: Type of restriction to be applied on the cluster scatter matrices. Valid values are "eigen" (default) , "deter" and "sigma".
        :param maxfact_e: level of eigen constraints
        :param maxfact_d: level of determinant constraints
        :param m:
        :param zero_tol: The zero tolerance used. Default = 1e-16.
        :param verbose: Defines the verbosity level (default = True). If True, it gives additional information on the iteratively decreaseing objective function's value.
        :param opt:
        :param sol_ini:
        :param tk:
        """
        self.k = k  # number of clusters
        self.alpha = alpha  # level of trimming
        self.n_inits = n_inits  # number of random initializations
        self.ksteps = ksteps  # number of iterations within each initialization
        self.equal_weights = equal_weights  # equal population proportions  for all clusters
        self.zero_tol = zero_tol  # zero tolerance	(to be implemented)
        self.m = m  # fuzzy power parameter
        self.restr_deter = None
        self.restr_cov_value = restr_cov_value
        self.maxfact_e = maxfact_e  # level eigen constraints
        self.maxfact_d = maxfact_d  # level determinant constraints
        assert opt.lower() in ['mixture', 'hard', 'fuzzy'], "opt must be 'mixture' for a mixture model, " \
                                                            "'hard' for a hard clustering assignment, " \
                                                            "or 'fuzzy' for fuzzy clustering"
        self.opt = opt.lower()  # estimated model ('mixture' for a mixture model; 'hard' for hard assignment; 'fuzzy' for fuzzy clustering)
        self.sol_ini = sol_ini  # initial solution (provided by the user); needs to be an Iteration object
        self.tk = tk  # tkmeans
        self.iter = None
        self.best_iter = Iteration()
        self.no_trim = None
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Compute tclust clustering.
        
        Parameters
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. The observations should be given row-wise.

        y : Ignored
            Not used, present for API consistency with scikit-learn by convention.
        
        Returns
        --------
        self
            Fitted estimator.
        """
        
        n_samples, n_features = X.shape  # number of observations (n_samples); number of dimensions (n_features)
        self._check_params(n_samples, n_features)
        # Start algorithm
        for j in range(self.n_inits):
            if self.verbose and j > 0 and (j + 1) % 50 == 0:
                print('Initialisation = {}/{}'.format(j + 1, self.n_inits))
            if self.sol_ini is not None:  # if user provided initial solution
                self.n_inits = 1
                self.iter = self.sol_ini
            else:  # random initial solution
                self.init_clusters(X)
            for i in range(self.ksteps):
                self.f_restr(n_features)  # restricting the clusters' scatter structure
                if self.iter.code == 0:  # all eigenvalues are 0
                    if i > 0:
                        self.calc_obj_function(X)
                        self.treatSingularity()
                        return
                    else:
                        for k in range(self.k):
                            self.iter.sigma[:, :, k] = np.identity(n_features)
                if self.opt == 'fuzzy':  # estimates the cluster's assignment and TRIMMING (fuzzy assignment)
                    self.findFuzzyClusterLabels(X)
                else:  # estimates the cluster's assignment and TRIMMING (mixture models and hard assignment)
                    self.findClusterLabels(X)
                
                if self.iter.code == 2 or i == self.ksteps - 1:
                    # if findClusterLabels returned 1, meaning that the cluster assignment has not changed or we're in the last concentration step:
                    break  # break the for loop: we finished this iteration! Don't re-estimate cluster parameters this time
                self.estimClustPar(X)  # estimates the cluster's parameters
            self.calc_obj_function(X)  # calculates the objetive function value
            if self.iter.obj > self.best_iter.obj:
                if self.verbose:
                    print('(%d/%d) New iteration is better! Old obj: %.2f; New obj: %.2f' % (
                        j + 1, self.n_inits, self.best_iter.obj, self.iter.obj))
                self.best_iter.update(obj=self.iter.obj, labels_=self.iter.labels_, csize=self.iter.csize,
                                      cw=self.iter.cw,
                                      sigma=self.iter.sigma, center=self.iter.center, z_ij=self.iter.z_ij,
                                      lambd=self.iter.lambd, code=self.iter.code)
        self.labels_ = self.best_iter.labels_.copy()
        return self
        
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
        :return: labels_ : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        
        check_is_fitted(self)
        if self.opt in ['hard', 'mixture']:
            self.findClusterLabels(X)
        else:
            self.findFuzzyClusterLabels(X)
        return iter.labels_
    
    def fit_predict(self, X, y=None):
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Convenience method; equivalent to calling fit(X) followed by predict(X).
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.
        
        Returns
        -------
        labels_ : ndarray of shape (n_samples,)
                Index of the cluster each sample belongs to.
        """
        
        return self.fit(X).labels_
    
    def calc_obj_function(self, X):
        """
        Calculates the objective function value for mixture, hard, and fuzzy assignments
        
        :param X: array of shape=(nsamples, nfeatures) containing the data
        :return: N/A
        """
        
        ww_m = np.zeros(shape=(X.shape[0], 1))  # mixture
        ww = np.zeros(shape=(X.shape[0], 1))  # hard or fuzzy
        for k in range(self.k):  # for each cluster
            if self.tk:
                w_m = self.iter.cw[k] * dmnorm_tk(x=X, mu=self.iter.center[k, :], lambd=self.iter.lambd[k])
            else:
                w_m = self.iter.cw[k] * dmnorm(x=X, mu=self.iter.center[k, :],
                                               sigma=np.asarray(self.iter.sigma[:, :, k]))
            w_m = w_m.reshape((-1, 1))  # column vector, 1 value per sample
            ww_m += w_m * (w_m >= 0)   # calculates each individual contribution for the obj funct (mixture assignment)
            if self.opt == 'hard':
                w = w_m * (self.iter.labels_ == (k + 1)).reshape((-1, 1))
                w = w.reshape((-1, 1))
                ww += w * (w >= 0)  # calculates each individual contribution for the obj funct (hard assignment)
            elif self.opt == 'fuzzy':
                z_ij_flattened = self.iter.z_ij[:, k].reshape((-1, 1))
                w = z_ij_flattened * np.log(w_m * (w_m >= 0) + 1 * (z_ij_flattened == 0))
                ww += w  # calculates each individual contribution for the obj funct (fuzzy assignment)
            assert ww.shape[-1] == ww_m.shape[-1] == 1
        
        ww_m = ww_m * (ww_m >= 0)
        if self.opt == 'mixture':
            self.iter.obj = np.sum(np.log(ww_m[self.iter.labels_ > 0]))
        elif self.opt == 'hard':
            ww = ww * (ww >= 0)
            self.iter.obj = np.sum(np.log(ww[self.iter.labels_ > 0]))
        elif self.opt == 'fuzzy':
            self.iter.obj = np.sum(ww[self.iter.labels_ > 0])
        return
    
    def estimClustPar(self, X):
        """
        Function to estimate model parameters
        
        :param X: array of shape=(nsamples, nfeatures) containing the data
        :return: N/A
        """
        
        for k in range(self.k):  # for each cluster
            if self.iter.csize[k] > self.zero_tol:  # if cluster size is > 0
                self.iter.center[k, :] = (self.iter.z_ij[:, k].T).dot(X) / self.iter.csize[k]
                X_c = X - self.iter.center[k, :]
                if not self.tk:  # CHECKED OK
                    self.iter.sigma[:, :, k] = \
                        np.matmul(np.multiply(X_c, self.iter.z_ij[:, k].reshape(-1, 1)).T, X_c) / self.iter.csize[k]
                else:
                    self.iter.lambd[k] = \
                        np.mean(np.sum(np.matmul(self.iter.z_ij[:, k].T, X_c ** 2) / self.iter.csize[k], axis=0))
            else:  # this cluster's size is 0
                if self.tk:
                    self.iter.lambd[k] = 0
                else:
                    self.iter.sigma[:, :, k] = np.zeros((self.iter.sigma.shape[0], self.iter.sigma.shape[1]))

    ####################################################################################
    ########### Functions for obtaining the assignment and trimming: ###################
    ########### - findClusterLabels: for mixture models and hard assignment ############
    ########### - findFuzzyClusterLabels: for fuzzy assignment #########################
    ####################################################################################

    def findClusterLabels(self, X):
        """
        Obtain the cluster assignment and trimming in the non-fuzzy case (i.e., mixture and hard assignments)
        
        :param X: array of shape=(nsamples, nfeatures) containing the data
        :return: N/A
        """
        
        ll = self.get_ll(X)
        old_labels_ = self.iter.labels_.copy()
        self.iter.labels_ = np.argmax(ll, axis=1) + 1  # searching the cluster which fits each observation best
        pre_z_h = np.max(ll, axis=1)
        pre_z_m = np.sum(ll, axis=1)
        pre_z_ = np.tile(pre_z_m.reshape(-1, 1), (1, self.k))
        assert pre_z_.shape[0] == X.shape[0]
        assert pre_z_.shape[1] == self.k
        # TODO: if we want to use this function to predict on new samples, we need to pass a new alpha to avoid/change
        #  the trimming or an option to trim=False?
        # To obtain the trimming: tc_set is the non-trimming indicator
        if self.opt == 'mixture':
            tc_set = np.argsort(pre_z_m.argsort()) >= (np.floor(X.shape[0] * self.alpha))
        elif self.opt == 'hard':
            tc_set = np.argsort(pre_z_h.argsort()) >= (np.floor(X.shape[0] * self.alpha))
        # To obtain the self.iter.z_ij matrix (which contains the assignment and trimming):
        self.iter.labels_ = (np.argmax(ll, axis=1) + 1) * tc_set  # hard assignnment including trimming (labels_ E [1,K])
        self.iter.z_ij = ll / (pre_z_ + (pre_z_ == 0)) * tc_set.reshape((-1, 1))  # mixture assignment including trimming
        # Obtain the size of the clusters and the estimated weight of each population
        if self.opt == 'hard':
            self.iter.z_ij = 0 * self.iter.z_ij
            self.iter.z_ij[np.arange(X.shape[0]), 1 * (self.iter.labels_ == 0) + (self.iter.labels_ - 1)] = 1
            self.iter.z_ij[~tc_set, :] = 0  # return 0 for trimmed samples
            self.iter.code = 2 * np.all(old_labels_ == self.iter.labels_)  # setting the code, signaling whether the labels have changed --- it's the only stopping rule implemented
            self.iter.csize = np.asarray([self.iter.labels_.tolist().count(cl + 1) for cl in range(self.k)])
        elif self.opt == 'mixture':
            self.iter.csize = np.sum(self.iter.z_ij, axis=0)
        if not self.equal_weights:
            self.iter.cw = self.iter.csize / np.sum(self.iter.csize)  # obtain cluster weights
    
    def findFuzzyClusterLabels(self, X):
        """
        Obtain assignment and trimming in the fuzzy case
       
        :param X: array of shape=(nsamples, nfeatures) containing the data
        :return: N/A
        """
        
        n = X.shape[0]
        ll = self.get_ll(X)
        # Obtain the cluster assignnment (self.iter.labels_)
        self.iter.labels_ = np.argmax(ll, axis=1) + 1  # searching the cluster which fits best for each observation
        ll_ = np.max(ll, axis=1)
        # Obtain the cluster assignment matrix (self.iter.z_ij)
        yy = np.nan * np.ones((n, self.k, self.k))  # Initialization
        ll_log = np.log(ll)
        for ii1 in range(self.k):
            for ii2 in range(self.k):
                yy[:, ii1, ii2] = (ll_log[:, ii1] / ll_log[:, ii2]) ** (1 / (self.m - 1))
        yy_ = 1 / np.sum(yy, axis=2)  # [nsamples, self.k]
        nm = self.k - np.sum(np.invert(np.isnan(yy_)), axis=1)
        
        where = (0 < nm) & (nm < self.k)
        if np.any(where):
            yy1_ = yy_[where, :].copy()
            yy1_[np.isnan(yy_[where, :])] = 0
            yy_[where, :] = yy1_
        yy_[nm == self.k, :] = 1 / self.k
        
        self.iter.z_ij = np.zeros((n, self.k))  # Initialization
        self.iter.z_ij[np.arange(n), self.iter.labels_ - 1] = 1
        self.iter.z_ij[ll_ <= 1, :] = yy_[ll_ <= 1, :] ** self.m
        
        # Obtain the trimming:  tc_set is the non trimming indicator
        pre_z = np.nansum(self.iter.z_ij * ll_log, axis=1)
        tc_set = np.argsort(pre_z.argsort()) >= np.floor(n * self.alpha)
        
        # Obtain the assignment iter$labels_  iter$z_ij including trimming
        self.iter.labels_ = (np.argmax(ll, axis=1) + 1) * tc_set
        self.iter.z_ij[~tc_set, :] = 0
        
        # Obtain the size of the clusters and the estimated weight of each population
        self.iter.csize = np.sum(self.iter.z_ij, axis=0)
        if not self.equal_weights:
            self.iter.cw = self.iter.csize / np.sum(self.iter.csize)
    
    ################################################################
    ########### Functions for random initialization ################
    ################################################################
    
    def getini(self):
        """
        Calculates the initial cluster sizes
        
        :return: array, number of samples in each cluster
        """
        
        if self.k == 1:
            return np.array(self.no_trim)
        pi_ini = np.random.uniform(low=0, high=1, size=self.k)  # sample from random uniform distribution
        ni_ini = np.random.choice(self.k, self.no_trim, replace=True, p=pi_ini / np.sum(pi_ini)) + 1
        return np.asarray([ni_ini.tolist().count(cl + 1) for cl in range(self.k)])
    
    def init_clusters(self, X):
        """
        Calculates the initial cluster assignment and initial values for the parameters
        
        :param X: 2D array of data [samples, dimensions]
        :return: N/A
        """
        
        n, p = X.shape
        for k in range(self.k):
            # Select observations randomly for the current initialisation cluster
            idx = np.random.choice(range(n), size=p + 1, replace=False)
            X_ini = X[idx, :]
            self.iter.center[k] = np.mean(X_ini, axis=0)  # calculate the center
            if self.tk:
                X_c = X_ini - self.iter.center[k, :]  # .reshape(p + 1, p)
                self.iter.lambd[k] = np.mean(np.sum(np.ones(shape=(1, p + 1)).dot(X_c ** 2) / (p + 1), axis=0))
            else:
                cc = p / (p + 1) * np.cov(X_ini.T)  # calculating sigma (cc = current cov)
                self.iter.sigma[:, :, k] = cc
        if self.equal_weights:  # if we're considering equal weights, cw is set here AND NEVER CHANGED
            self.iter.csize = np.asarray([self.no_trim / self.k] * self.k)
        else:
            self.iter.csize = self.getini()
        self.iter.cw = self.iter.csize / self.no_trim
    
    def treatSingularity(self):
        """
        To manage singular situations.
        
        :return: N/A
        """
        
        if self.restr_deter or self.restr_cov_value == 'sigma':
            print("WARNING: Points in the dataset are concentrated in k subspaces after trimming")
        else:
            print("WARNING: Points in the dataset are concentrated in k points after trimming")
    
    def get_ll(self, X):
        """
        Extracted this function to avoid repetition in the code
        
        :param X: array; 2D array of data, of shape [nsamp, nfeat]
        :return: array of shape (nsamp, k); ll
        """
        
        ll = np.nan * np.ones((X.shape[0], self.k))
        if self.tk:
            for k in range(self.k):
                ll[:, k] = self.iter.cw[k] * dmnorm_tk(x=X, mu=self.iter.center[k, :], lambd=self.iter.lambd[k])
        else:
            for k in range(self.k):
                ll[:, k] = self.iter.cw[k] * dmnorm(x=X, mu=self.iter.center[k, :],
                                                    sigma=np.asarray(self.iter.sigma[:, :, k]))
        return ll

    #####################################################################################
    ########### Functions to apply constraints to covariance matrices ###################
    #####################################################################################
    
    def restr2_eigenv(self, autovalues, ni_ini, factor_e, zero_tol):
        """
        Function for applying eigen constraints. These are the typical constraints.
        
        :param autovalues: matrix containing eigenvalues
        :param ni_ini: current sample size of the clusters
        :param factor_e: level of the constraints
        :param zero_tol: tolerance level
        :return: ?
        """
        
        assert factor_e > 0
        # Initialization
        d = np.copy(autovalues.T)
        p, k = autovalues.shape
        n = np.sum(ni_ini)
        if p > k:
            nis = np.tile(np.array(ni_ini).reshape((1, -1)), (k, p))[:k, :p]
        else:
            nis = np.tile(np.array(ni_ini).reshape((-1, 1)), (k, p))[:k, :p]

        # d_ is the ordered set of values in which the restriction objective function change the definition
        # points in d_ correspond to the frontiers for the intervals in which this objective function
        # has the same definition
        # ed is a set with the middle points of these intervals
        d_ = list(np.sort(np.concatenate((d.flatten(), d.flatten() / factor_e))))
        ed = (np.asarray(d_ + [2 * d_[-1]]) + np.asarray([0] + d_)) / 2
        dim = ed.shape[0]
        
        # The only relevant eigenvalues are those that belong to clusters with sample size > 0.
        # Eigenvalues corresponding to clusters with 0 individuals have no influence in the objective function.
        # If all the eigenvalues are 0 during the smart initialization, we assign to all the eigenvalues the value 1.
        if np.max(d[nis > 0]) <= zero_tol:
            return np.zeros((p, k))  # solution corresponds to matrix of 0s
        # Check if the eigenvalues verify the restrictions
        if np.min(d[nis > 0]) == 0:  # avoiding runtime warning when dividing by 0
            denom = 1e-16
        else:
            denom = np.min(d[nis > 0])
        if np.abs(np.max(d[nis > 0]) / denom) <= factor_e:
            d[nis == 0] = np.mean(d[nis > 0])
            return d.T  # the solution corresponds to the input because it verifies the constraints
        
        # we build the sol array, which contains the critical values of the interval functions which define the objective function
        # we use the centers of the interval to get a definition for the function in each interval
        # this set with the critical values (in the array sol) contains the optimum m value
        t = np.zeros((k, dim))
        s = np.zeros((k, dim))
        r = np.zeros((k, dim))
        sol = np.zeros(dim)
        sal = np.zeros(dim)
        for mp_ in range(dim):
            r[:, mp_] = np.sum(d < ed[mp_], axis=1) + np.sum(d > (ed[mp_] * factor_e), axis=1)
            s[:, mp_] = np.sum(d * (d < ed[mp_]), axis=1)
            t[:, mp_] = np.sum(d * (d > (ed[mp_] * factor_e)), axis=1)
            sol[mp_] = np.sum(ni_ini / n * (s[:, mp_] + t[:, mp_] / factor_e)) / (np.sum(ni_ini / n * r[:, mp_]))  # this can be NAN
            e = sol[mp_] * (d < sol[mp_]) + d * (d >= sol[mp_]) * (d <= factor_e * sol[mp_]) + (factor_e * sol[mp_]) * (d > factor_e * sol[mp_])
            sal[mp_] = np.sum(-0.5 * nis / n * (np.log(e) + d / e))
        # m is the optimum value for the eigenvalues procedure
        m = sol[np.nanargmax(sal)]
        # based on the m value we get the restricted eigenvalues
        temp = m * (d < m) + d * (d >= m) * (d <= factor_e * m) + (factor_e * m) * (d > factor_e * m)
        return temp.T  # the return value
    
    
    def restr2_deter_(self, autovalues, ni_ini, factor_d, factor_e, zero_tol=1e-16):
        """
        Function for applying constraints to the determinants.
        
        Used when p>1 (multivariate case) -- in the univariate case the constraints can be obtained with restr2_eigenv()
        In order to avoid the instability in the current release of this function implemented in the CRAN,
        it is better to apply these constraints, at the desired level,
        joint to eigenvalue constraints at very low level (factor_e=1e10).
        In this way eigenvalues are not constrained in practice, but numerical issues are avoided.

        :param autovalues: matrix containing eigenvalues
        :param ni_ini: current sample size of the clusters
        :param factor_d: constraint level for the determinants
        :param factor_e: constraint level for the eigenvalues
        :param zero_tol: tolerance level
        :return: ?
        """
        
        nrows, ncols = autovalues.shape
        autovalues[autovalues < 1e-16] = 0  # TODO: it's like this in the R code, but should probably be passed as a variable instead
        autovalues_ = np.copy(autovalues)
        for k_ in range(autovalues.shape[1]):
            autovalues_[:, k_] = self.restr2_eigenv(autovalues=autovalues[:, k_].reshape(-1, 1), ni_ini=1,
                                                    factor_e=factor_e, zero_tol=zero_tol).reshape(-1)
        es = np.prod(autovalues_, axis=0)
        es[es == 0] = 1
        temp = np.tile((es ** (1 / nrows)).reshape(1, -1), (nrows, 1))
        assert temp.shape[0] == nrows
        assert temp.shape[1] == ncols
        gm = autovalues_ / temp
        d_ = np.sum(autovalues / gm, axis=0).reshape(1, -1) / nrows
        d_[np.isnan(d_)] = 0
        dfin = self.restr2_eigenv(autovalues=d_, ni_ini=ni_ini, factor_e=factor_d ** (1 / nrows), zero_tol=zero_tol)
        d__ = np.tile(dfin.reshape(1, -1), (nrows, 1)) * (gm * (gm > 0) + 1 * (gm == 0))
        return d__
    
    def restr_diffax(self, p):
        """
        Function which manages the application of constraints (deter, eigen)
        
        :param p: int, number of features of the data
        :return: N/A
        """
        
        u = np.nan * np.ones((p, p, self.k))
        d = np.nan * np.ones((p, self.k))
        if not self.tk:
            for k in range(self.k):
                d[:, k], u[:, :, k] = np.linalg.eig(self.iter.sigma[:, :, k])
        else:
            d = np.tile(self.iter.lambd, (d.shape[0], 1))
        d[d < 0] = 0  # all eigenvalues < 0 are assigned to 0, this happens sometimes due to numerical errors
        if self.restr_deter and p > 1:
            d = self.restr2_deter_(autovalues=d, ni_ini=self.iter.csize, factor_d=self.maxfact_d,
                                       factor_e=1 if self.tk else self.maxfact_e, zero_tol=self.zero_tol)
        else:
            d = self.restr2_eigenv(autovalues=d, ni_ini=self.iter.csize, factor_e=self.maxfact_e,
                                   zero_tol=self.zero_tol)
        # Checking for singularity in all clusters:
        self.iter.code = np.max(d) > self.zero_tol
        if not self.iter.code:
            return
        if not self.tk:
            for k in range(self.k):  # recomposing the sigmas
                self.iter.sigma[:, :, k] = np.dot(np.dot(u[:, :, k], np.diag(d[:, k])), u[:, :, k].T)
        else:
            self.iter.lambd = d[0, :].copy()
    
    def restr_avgcov(self, p):
        """
        Restricts the clusters' covariance matrices to be equal.
        Simple function to get the pooled within group covariance matrix.
        
        :param p: int, number of dimensions of the data
        :return: N/A
        """
        
        s_all = np.zeros((p, p))
        for k in range(self.k):
            s_all += self.iter.sigma[:, :, k] * self.iter.csize[k] / np.sum(self.iter.csize)
        for k in range(self.k):
            self.iter.sigma[:, :, k] = s_all.copy()
        self.iter.code = int(np.sum(np.diag(s_all)) > self.zero_tol)
    
    
    def _check_params(self, n, p):
        """
        Checks the parameters for the execution and completes missing variables.
        
        :param n: int, number of observations
        :param p: int, number of dimensions
        :return: N/A
        """
        
        self.iter = Iteration().fill(nobs=n, ndim=p, k=self.k)
        if self.restr_cov_value == 'sigma':
            self.f_restr = self.restr_avgcov
            self.restr_deter = False
        if p == 1:
            self.f_restr = self.restr_diffax
            self.restr_deter = False
        else:
            if self.restr_cov_value == 'eigen':
                self.f_restr = self.restr_diffax
                self.restr_deter = False
            elif self.restr_cov_value == 'deter':
                self.f_restr = self.restr_diffax
                self.restr_deter = True
        self.no_trim = int(np.floor(n * (1 - self.alpha)))  # number of observations which are considered not outliers


######################################################
########### Miscelaneous functions ###################
######################################################

@jit(nopython=True)
def dmnorm(x, mu, sigma):
    """
    Multivariate normal density
    
    :param x:  array of shape=(nsamples, nfeatures) containing the data
    :param mu: center of the cluster [features, ]
    :param sigma:
    :return: ?
    """
    
    centered_array = x - mu
    inv_cov = np.linalg.inv(sigma)
    mahal_dist = []
    for i in range(centered_array.shape[0]):
        mahal_dist.append(np.dot(np.dot(centered_array[i, :], inv_cov), centered_array[i, :]))
    assert len(mahal_dist) == x.shape[0]
    result = ((2 * np.pi) ** (-0.5 * len(mu))) * (np.linalg.det(sigma) ** (-1 / 2)) * np.exp(-0.5 * np.asarray(mahal_dist))
    return result


@jit(nopython=True)
def dmnorm_tk(x, mu, lambd):
    """
    Multivariate normal density sigma=lambd*ID
    
    :param x:  array of shape=(nsamples, nfeatures) containing the data
    :param mu: center of the cluster [features, ]
    :param lambd: one number - diagonal value for tkmeans (whatever that means)
    :return: ?
    """
    
    a = ((2 * np.pi) ** (-0.5 * len(mu))) * (lambd ** (-len(mu) / 2))
    b = (np.multiply(np.ones((x.shape[0], 1)), mu) - x) ** 2
    c = np.exp(-(0.5 / lambd) * np.sum(b, axis=1))
    return a * c
