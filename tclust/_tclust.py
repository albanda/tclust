from ._iteration import Iteration

import numpy as np
from numpy.matlib import repmat
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_is_fitted

from numba import jit

import warnings
warnings.filterwarnings('ignore')



class TClust(ClusterMixin, BaseEstimator, TransformerMixin):
    """
    Perform Tclust clustering from matrix

    Parameters
    ----------
    k : int, default=
        The number of clusters
    alpha : float, default=
        The proportion of observations to be trimmed
    nstart : int, default=
        The number of random intializations to be performed
    niter : int, default=20
        The maximum number of concentration steps to be performed (Ksteps)
    restr_cov_value : string, default='eigen'
        The type of restriction to be applied on the cluster scatter matrices "eigen"  "deter" and "sigma"

    equal_weights : bool, default=False
        Specifying whether equal cluster weights are equal
    zero_tol : float, default=1e-16
        The zero tolerance used

    Attributes
    ----------
    #centers A matrix of size p x k containing the centers columnwise of each cluster. (center)
    #cov An array of size p x p x k containing the covariance matrices of each cluster (sigma)
    #cluster A numerical vector of size n containing the cluster labels_nment for each observation (labels_)
    #obj The value of the objective function of the best solution (obj)
    #size An integer vector of size k, returning the number of observations contained by each cluster (csize)
    #weights A numerical vector of length k, containing the weights of each cluster (cw)

    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    """
    
    def __init__(self, k, alpha=0.05, niter=20, ksteps=10, equal_weights=False, restr_cov_value='eigen',
                 maxfact_e=5., maxfact_d=5, m=2., zero_tol=1e-16, trace=0, opt='hard', sol_ini=None,
                 tk=False, verbose=True):
        """

        :param k: number of clusters
        :param alpha: trimming factor
        :param niter: maximum number of iterations
        :param ksteps: TODO
        :param equal_weights: Specifying whether equal cluster weights are equal
        :param restr_cov_value: The type of restriction to be applied on the cluster scatter matrices "eigen" , "deter" or "sigma"
        :param maxfact_e:
        :param maxfact_d:
        :param m:
        :param zero_tol:
        :param trace:
        :param opt:
        :param sol_ini:
        :param tk:
        """
        self.k = k  # number of clusters
        self.alpha = alpha  # level of trimming
        self.niter = niter  # number of random starts
        self.ksteps = ksteps  # number of iterations
        self.equal_weights = equal_weights  # equal population proportions  for all clusters
        self.zero_tol = zero_tol  # zero tolerance	(to be implemented)
        self.trace = trace  # level of information provided (to be implemented)
        self.m = m  # fuzzy power parameter
        self.restr_deter = None
        self.restr_cov_value = restr_cov_value
        self.maxfact_e = maxfact_e  # level eigen constraints
        self.maxfact_d = maxfact_d  # level determinant constraints
        assert opt.lower() in ['mixture', 'hard', 'fuzzy'], "opt must be 'mixture' for a mixture model, " \
                                                            "'hard' for a hard clustering labels_nment, " \
                                                            "or 'fuzzy' for fuzzy clustering"
        self.opt = opt.lower()  # estimated model ('mixture' for a mixture model; 'hard' for hard labels_nment; 'fuzzy' for fuzzy clustering)
        self.sol_ini = sol_ini  # initial solution
        self.tk = tk  # tkmeans
        self.iter = None
        self.best_iter = Iteration()
        self.no_trim = None
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Computer tclust clustering
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
                    Training instances to cluster.
        :param y: Ignored
            Not used, present for sklearn consistency by convention.
        :return: fitted estimator
        """
        n_samples, n_features = X.shape  # number of observations (n_samples); number of dimensions (n_features)
        self._check_params(X)
        # Start algorithm
        for j in range(self.niter):
            if self.verbose and j > 0 and (j + 1) % 50 == 0:
                print('Iteration = {}/{}'.format(j + 1, self.niter))
            if self.sol_ini is not None:  # needs to be an object of type Iteration; initial solution provided by the user
                self.niter = 1
                self.iter = self.sol_ini
            else:  # random initial solution
                self.iter = self.init_clusters(X)
            for i in range(self.ksteps):
                self.iter = self.f_restr(n_features)  # restricting the clusters' scatter structure
                if self.iter.code == 0:  # all eigenvalues are 0
                    if i > 0:
                        self.calc_obj_function(X)
                        return self.treatSingularity()
                    else:
                        for k in range(self.k):
                            self.iter.sigma[:, :, k] = np.identity(n_features)
                if self.opt == 'fuzzy':  # estimates the cluster's assignment and TRIMMING (FUZZY)
                    self.iter = self.findClusterlabels__fuzzy(X)
                else:  # estimates the cluster's assignment and TRIMMING (mixture models and HARD)
                    self.iter = self.findClusterlabels_(X)
                
                if self.iter.code == 2 or i == self.ksteps - 1:  # if findClusterLabels returned 1, meaning that the cluster assignment has not changed or we're in the last concentration step:
                    break  # break the for loop: we finished this iteration! Don't re-estimate cluster parameters this time
                self.iter = self.estimClustPar(X)  # estimates the cluster's parameters
            self.calc_obj_function(X)  # calculates the objetive function value
            if self.iter.obj > self.best_iter.obj:
                if self.verbose:
                    print('(%d/%d) New iteration is better! Old obj: %.2f; New obj: %.2f' % (
                j + 1, self.niter, self.best_iter.obj, self.iter.obj))
                self.best_iter.update(obj=self.iter.obj, labels_=self.iter.labels_, csize=self.iter.csize,
                                      cw=self.iter.cw,
                                      sigma=self.iter.sigma, center=self.iter.center, z_ij=self.iter.z_ij,
                                      lambd=self.iter.lambd, code=self.iter.code)
        self.labels_ = self.best_iter.labels_
        return self
    
    # TODO?
    def transform(self, X):
        """
        Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster centers.
        Note that even if X is sparse, the array returned by `transform` will typically be dense.
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
                New data to predict.
        :return:
            X_new : ndarray of shape (n_samples, n_clusters)
                    X transformed in the new space.
        """
        # TODO
        check_is_fitted(self)
        return euclidean_distances(X, self.iter.center)
    
    def fit_transform(self, X, y=None):
        """
        Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
        :param y: Ignored
            Not used, present for sklearn consistency by convention.
        :return:
            X_new : ndarray of shape (n_samples, n_clusters)
                X transformed in the new space.
        """
        return self.fit(X).transform(X)
    
    # TODO?
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
        :return: labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        if self.opt in ['hard', 'mixture']:
            iter = self.findClusterlabels_(X)
            return iter.labels_
        else:
            iter = self.findClusterlabels__fuzzy(X)
            return iter.labels_
    
    def fit_predict(self, X, y=None):
        """
        Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by predict(X).
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        :param y: Ignored
            Not used, present here for API consistency by convention.
        :return:
            labels : ndarray of shape (n_samples,)
                Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_
    
    def calc_obj_function(self, X):
        """
        Calculates the objective function value for mixture (obj_m) hard (obj_h) and fuzzy (obj_f)
        :param X: Array of data [nsamples, nfeatures]
        :return:
        """
        ww_m = np.zeros(shape=(X.shape[0], 1))  # mixture
        ww = np.zeros(shape=(X.shape[0], 1))  # hard # fuzzy
        for k in range(self.k):  # TODO should be able to parallelise this
            if self.tk:
                w_m = self.iter.cw[k] * dmnorm_tk(x=X, mu=self.iter.center[k, :], lambd=self.iter.lambd[k])
            else:
                w_m = self.iter.cw[k] * dmnorm(x=X, mu=self.iter.center[k, :],
                                               sigma=np.asarray(self.iter.sigma[:, :, k]))
            w_m = w_m.reshape((-1, 1))  # column vector,  1 value per sample
            ww_m = w_m * (w_m >= 0) + ww_m  # calculates each individual contribution for the obj funct mixture
            if self.opt == 'hard':
                w = w_m * (self.iter.labels_ == (k + 1)).reshape((-1, 1))
                w = w.reshape((-1, 1))
                ww = w * (w >= 0) + ww  # calculates each individual contribution for the obj funct hard
            elif self.opt == 'fuzzy':
                z_ij_flattened = self.iter.z_ij[:, k].reshape((-1, 1))
                w = z_ij_flattened * np.log(w_m * (w_m >= 0) + 1 * (z_ij_flattened == 0))
                ww = w + ww  # calculates each individual contribution for the obj funct fuzzy
            assert ww.shape[-1] == ww_m.shape[-1] == 1
        
        ww_m = ww_m * (ww_m >= 0)
        if self.opt == 'mixture':
            self.iter.obj = np.sum(np.log(ww_m[self.iter.labels_ > 0]))
        elif self.opt == 'hard':
            ww = ww * (ww >= 0)
            self.iter.obj = np.sum(np.log(ww[self.iter.labels_ > 0]))
        elif self.opt == 'fuzzy':
            self.iter.obj = np.sum(ww[self.iter.labels_ > 0])
    
    def estimClustPar(self, X):
        """
        Function to estimate model parameters
        :param X: 2D array of data to cluster [samples, variables]
        :return:
        """
        for k in range(self.k):
            if self.iter.csize[k] > self.zero_tol:  # if cluster size is > 0
                self.iter.center[k, :] = (self.iter.z_ij[:, k].T).dot(X) / self.iter.csize[k]
                X_c = X - self.iter.center[k, :]
                if not self.tk:
                    self.iter.sigma[:, :, k] = np.matmul(np.multiply(X_c, self.iter.z_ij[:, k].reshape(-1, 1)).T, X_c) / self.iter.csize[k]
                else:
                    self.iter.lambd[k] = np.mean(
                        np.sum(np.matmul(self.iter.z_ij[:, k].T, X_c ** 2) / self.iter.csize[k], axis=0))
            else:  # this cluster's size has decreased to 0
                if self.tk:
                    self.iter.lambd[k] = 0
                else:
                    self.iter.sigma[:, :, k] = 0
        return self.iter
    
    ######## FUNCTIONS FOR obtaining the assignment and trimming: findClustlabels_ (mixture models and hard assignment)  findClustlabels__f  (fuzzy assignment)
    def findClusterlabels_(self, X):
        """
        FUNCTION FOR obtaining the cluster assignment and trimming in the non FUZZY CASE (mixture and hard labels_nments)
        
        :param X: data matrix (samples x dimensions)
        :return:
        """
        ll = self.get_ll(X)
        old_labels_ = self.iter.labels_.copy()
        self.iter.labels_ = np.argmax(ll, axis=1) + 1  # searching the cluster which fits each observation best
        pre_z_h = np.max(ll, axis=1)
        pre_z_m = np.sum(ll, axis=1)
        pre_z_ = repmat(pre_z_m.reshape(-1, 1), 1, self.k)
        assert pre_z_.shape[0] == X.shape[0]
        assert pre_z_.shape[1] == self.k
        # TODO: if we want to use this function to predict on new samples, we need to pass a new alpha to avoid/change
        #  the trimming or an option to trim=False?
        # To obtain the trimming: tc_set is the non-trimming indicator
        if self.opt == 'mixture':
            tc_set = np.argsort(pre_z_m) > (np.floor(X.shape[0] * self.alpha))
        elif self.opt == 'hard':
            tc_set = np.argsort(pre_z_h) > (np.floor(X.shape[0] * self.alpha))
        # To obtain the self.iter.z_ij matrix containing the labels_nment and trimming:
        self.iter.labels_ = (np.argmax(ll, axis=1) + 1) * tc_set  # hard labels_nment including trimming
        self.iter.z_ij = ll / (pre_z_ + (pre_z_ == 0)) * tc_set.reshape((-1, 1))  # mixture labels_nment including trimming
        if self.opt == 'hard':
            self.iter.z_ij = 0 * self.iter.z_ij
            self.iter.z_ij[np.arange(X.shape[0]), self.iter.labels_ - 1] = 1
            # self.iter.z_ij[np.arange(n), self.iter.labels_ == 0] == 1
            self.iter.z_ij[~tc_set, :] = 0  # return 0 for trimmed samples
            self.iter.code = 2 * np.all(
                old_labels_ == self.iter.labels_)  # setting the code - parameter, signaling whether the labels_nment is the same than the previous --- is the only stopping rule implemented
        
        # Obtain the size of the clusters and the estimated weight of each population
        if self.opt == 'hard':
            self.iter.csize = np.asarray([self.iter.labels_.tolist().count(cl + 1) for cl in range(self.k)])
        elif self.opt == 'mixture':
            self.iter.csize = np.sum(self.iter.z_ij, axis=0)
        if not self.equal_weights:
            self.iter.cw = self.iter.csize / np.sum(self.iter.csize)  # add cluster weights
        return self.iter
    
    def findClusterlabels__fuzzy(self, X):
        """
        Obtain assignment and trimming in the fuzzy case
       
        :param X:
        :return:
        """
        n = X.shape[0]
        ll = self.get_ll(X)
        # Obtain the cluster assignnment (self.iter.labels_)
        self.iter.labels_ = np.argmax(ll, axis=1) + 1  # searching the cluster which fits best for each observation
        ll_ = np.max(ll, axis=1)
        # Obtain the cluster assignment matrix (self.iter.z_ij)
        yy = np.empty((n, self.k, self.k))  # Initialization
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
        tc_set = np.argsort(pre_z) > np.floor(n * self.alpha)
        
        # Obtain the assignment iter$labels_  iter$z_ij including trimming
        self.iter.labels_ = (np.argmax(ll, axis=1) + 1) * tc_set
        self.iter.z_ij[~tc_set, :] = 0
        
        # Obtain the size of the clusters and the estimated weight of each population
        self.iter.csize = np.sum(self.iter.z_ij, axis=0)
        if not self.equal_weights:
            self.iter.cw = self.iter.csize / np.sum(self.iter.csize)
        return self.iter
    
    ######## FUNCTIONS FOR RANDOM INITIALIZATION:  getini; init_clusters
    def getini(self):
        """
        Calculates the initial cluster sizes
        :return:
        """
        if self.k == 1:
            return np.array(self.no_trim)
        pi_ini = np.random.uniform(low=0, high=1, size=self.k)  # sample from random uniform distribution
        ni_ini = np.random.choice(self.k, self.no_trim, replace=True, p=pi_ini / np.sum(pi_ini)) + 1
        return np.asarray([ni_ini.tolist().count(cl + 1) for cl in range(self.k)])
    
    def init_clusters(self, X):
        """
        Calculates the initial cluster labels_nment and initial values for the parameters
        :param X: 2D array of data [samples, dimensions]
        :return:
        """
        n, p = X.shape
        for k in range(self.k):
            idx = np.random.choice(range(n), size=p + 1, replace=False)
            # Select observations randomly for the current initialisation cluster
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
        return self.iter
    
    def treatSingularity(self):
        '''
        To manage singular situations
        :return:
        '''
        if self.restr_deter or self.restr_cov_value == 'sigma':
            print("WARNING: Points in the data set are concentrated in k subspaces after trimming")
        else:
            print("WARNING: Points in the data set are concentrated in k points after trimming")
        return self.iter
    
    def get_ll(self, X):
        """
        Extracted this function because it was repeated in the code and this is cleaner
        :param X: 2D array of data, of shape [nsamp, nfeat]
        :return: ll
        """
        ll = np.empty((X.shape[0], self.k))
        if self.tk:
            for k in range(self.k):  # TODO should be able to parallelise this
                ll[:, k] = self.iter.cw[k] * dmnorm_tk(x=X, mu=self.iter.center[k, :],
                                                       lambd=self.iter.lambd[k])
        else:
            for k in range(self.k):  # TODO should be able to parallelise this
                ll[:, k] = self.iter.cw[k] * dmnorm(x=X, mu=self.iter.center[k, :],
                                                    sigma=np.asarray(self.iter.sigma[:, :, k]))
        return ll
    
    ## FUNCTIONS FOR APPLYING CONSTRAINTS TO COVARIANCE MATRICES ##
    def restr2_eigenv(self, autovalues, ni_ini, factor_e, zero_tol):
        """
        FUNCTION FOR APPLYING EIGEN CONSTRAINTS. These are the typical constraints
        :param autovalues: matrix containing eigenvalues
        :param ni_ini: current sample size of the clusters
        :param factor_e: level of the constraints
        :param zero_tol: tolerance level
        :return:
        """
        
        # Initialization
        c = factor_e
        assert c > 0
        d = autovalues.T
        p = autovalues.shape[0]
        k = autovalues.shape[1]
        n = np.sum(ni_ini)
        nis = repmat(np.array(ni_ini).reshape(-1, 1), k, p)[:k, :p]
        
        # d_ is the ordered set of values in which the restriction objective function change the definition
        # points in d_ correspond to  the frontiers for the intervals in which this objective function has the same definition
        # ed is a set with the middle points of these intervals
        d_ = list(np.sort(np.concatenate((d.flatten(), d.flatten() / c))))
        ed = (np.asarray(d_ + [2 * d_[-1]]) + np.asarray([0] + d_)) / 2
        dim = ed.shape[0]
        
        # The only relevant eigenvalues are those that belong to clusters with sample size > 0.
        # Eigenvalues corresponding to clusters with 0 individuals have no influence in the objective function.
        # If all the eigenvalues are 0 during the smart initialization we labels_n to all the eigenvalues the value 1.
        #if n == 0:
        #    print('Possible break!', n, d[nis > 0])
        #    print('this should break:',
        #          max(d[nis > 0]))  # "zero-size array to reduction operation maximum which has no identity"
        if np.max(d[nis > 0]) <= zero_tol:
            return np.zeros((p, k))  # solution corresponds to matrix of 0s
        # Check if the eigenvalues verify the restrictions
        if np.min(d[nis > 0]) == 0:  # avoiding runtime warning when dividing by 0
            denom = 1e-16
        else:
            denom = np.min(d[nis > 0])
        if np.abs(np.max(d[nis > 0]) / denom) <= c:
            d[nis == 0] = np.mean(d[nis > 0])
            return d.T  # the solution corresponds to the input because it verifies the constraints
        
        # we build the sol array, which contains the critical values of the interval functions which define the m objective function
        # we use the centers of the interval to get a definition for the function in each interval
        # this set with the critical values (in the array sol) contains the optimum m value
        t = np.zeros((k, dim))
        s = np.zeros((k, dim))
        r = np.zeros((k, dim))
        sol = np.zeros(dim)
        sal = np.zeros(dim)
        for mp_ in range(dim):
            r[:, mp_] = np.sum(d < ed[mp_], axis=1) + np.sum(d > ed[mp_] * c, axis=1)
            s[:, mp_] = np.sum(d * (d < ed[mp_]), axis=1)
            t[:, mp_] = np.sum(d * (d > ed[mp_] * c), axis=1)
            # for i in range(k):
            #    r[i, mp_] = np.sum(d[i, :] < ed[mp_]) + np.sum(d[i, :] > ed[mp_]*c)
            #    s[i, mp_] = np.sum(d[i, :] * (d[i, :] < ed[mp_]))
            #    t[i, mp_] = np.sum(d[i, :] * (d[i, :] > ed[mp_] * c))
            sol[mp_] = np.sum(ni_ini / n * (s[:, mp_] + t[:, mp_] / c)) / (np.sum(ni_ini / n * r[:, mp_]))
            e = sol[mp_] * (d < sol[mp_]) + d * (d >= sol[mp_]) * (d <= c * sol[mp_]) + (c * sol[mp_]) * (
                        d > c * sol[mp_])
            sal[mp_] = np.sum(-0.5 * nis / n * (np.log(e) + d / e))
        # m is the optimum value for the eigenvalues procedure
        m = sol[np.nanargmax(sal)]
        # based on the m value we get the restricted eigenvalues
        temp = m * (d < m) + d * (d >= m) * (d <= c * m) + (c * m) * (d > c * m)
        return temp.T  # the return value
    
    def restr2_deter_(self, autovalues, ni_ini, factor_d, factor_e, zero_tol=1e-16):
        """
        FUNCTION FOR APPLYING DETER CONSTRAINTS
        It is applied when p>1 (multivariate case) because in the univariate case these constraints can be obtained
        with restr2_eigen()
        In order to avoid the instability in the current release of this function implemented in the CRAN,
        it is better to apply these constraints, at the desired level,
        joint to eigenvalue constraints at very low level (factor_e=1e10).
        In this way eigenvalues are not constrained in practice, but numerical issues are avoided.

        :param autovalues: matrix containing eigenvalues
        :param ni_ini: current sample size of the clusters
        :param factor_d: constraint level for the determinants
        :param factor_e: constraint level for the eigenvalues
        :return:
        """
        autovalues[autovalues < zero_tol] = 0
        autovalues_ = autovalues.copy()
        for k_ in range(autovalues.shape[1]):
            autovalues_[:, k_] = self.restr2_eigenv(autovalues=autovalues[:, k_].reshape(-1, 1), ni_ini=1,
                                                    factor_e=factor_e, zero_tol=zero_tol).reshape(-1)
        es = np.prod(autovalues_, axis=0)
        es[es == 0] = 1
        temp = repmat((es ** (1 / autovalues.shape[0])).reshape(1, -1), autovalues.shape[0], 1)
        assert temp.shape[0] == autovalues.shape[0]
        assert temp.shape[1] == autovalues.shape[1]
        gm = autovalues_ / temp
        d_ = np.sum(autovalues / gm, axis=0).reshape(1, -1) / autovalues.shape[0]
        d_[np.isnan(d_)] = 0
        dfin = self.restr2_eigenv(autovalues=d_, ni_ini=ni_ini, factor_e=factor_d ** (1 / autovalues.shape[0]),
                                  zero_tol=zero_tol)
        d__ = repmat(dfin.reshape(1, -1), autovalues.shape[0], 1) * (gm * (gm > 0) + 1 * (gm == 0))
        return d__
    
    def restr_diffax(self, p):
        """
        Function which manages constrains application (deter, eigen)
        :param p: number of dimensions of the data
        :return:
        """
        u = np.empty((p, p, self.k))
        d = np.empty((p, self.k))
        if not self.tk:
            for k in range(self.k):
                d[:, k], u[:, :, k] = np.linalg.eig(self.iter.sigma[:, :, k])
        else:
            # for k in range(self.k):
            d = repmat(self.iter.lambd, d.shape[0], 1)  # self.iter.lambd[k].copy()
        d[d < 0] = 0  # all eigenvalue < 0 are labels_ned to 0, this happens sometimes due to numerical errors
        if self.restr_deter and p > 1:
            if not self.tk:
                d = self.restr2_deter_(autovalues=d, ni_ini=self.iter.csize, factor_d=self.maxfact_d,
                                       factor_e=self.maxfact_e, zero_tol=self.zero_tol)
            else:
                d = self.restr2_deter_(autovalues=d, ni_ini=self.iter.csize, factor_d=self.maxfact_d, factor_e=1,
                                       zero_tol=self.zero_tol)
        else:
            d = self.restr2_eigenv(autovalues=d, ni_ini=self.iter.csize, factor_e=self.maxfact_e,
                                   zero_tol=self.zero_tol)
        # Checking for singularity in all clusters:
        self.iter.code = int(np.max(d) > self.zero_tol)
        if self.iter.code == 0:
            return self.iter
        if not self.tk:
            for k in range(self.k):  # recomposing the sigmas
                self.iter.sigma[:, :, k] = u[:, :, k].dot(np.diag(d[:, k])).dot(u[:, :, k].T)
        else:
            self.iter.lambd = d[0, :].copy()
        return self.iter
    
    def restr_avgcov(self, p):
        '''
        Restricts the clusters' covariance matrices to be equal
        Simple function to get the pooled within group covariance matrix
        :param p: number of dimensions of the data
        :return:
        '''
        s_all = np.zeros((p, p))
        for k in range(self.k):
            s_all += self.iter.sigma[:, :, k] * self.iter.csize[k] / np.sum(self.iter.csize)
        self.iter.sigma = s_all
        self.iter.code = int(np.sum(np.diag(s_all)) > self.zero_tol)
        return self.iter
    
    def _check_params(self, X):
        n, p = X.shape
        self.iter = Iteration().fill(nobs=n, ndim=p, k=self.k)
        if self.restr_cov_value == 'sigma':
            self.f_restr = self.restr_avgcov
            self.restr_deter = False
        if p == 1:
            self.f_restr = self.restr_diffax
            self.restr_deter = False
        elif self.restr_cov_value == 'eigen':
            self.f_restr = self.restr_diffax
            self.restr_deter = False
        elif self.restr_cov_value == 'deter':
            self.f_restr = self.restr_diffax
            self.restr_deter = True
        
        self.no_trim = np.int(
            np.floor(n * (1 - self.alpha)))  # number of observations which are considered not outliers


###### MISCELANEOUS FUNCTIONS: dmnorm  dmnorm_tk   ssclmat
@jit(nopython=True)
def dmnorm(x, mu, sigma):
    """
    Multivariate normal density
    :param x: 2D array (samples x dimensions) of data
    :param mu:
    :param sigma:
    :return:
    """
    # R: ((2 * pi) ^ (-length(mu) / 2)) * (det(sigma) ^ (-1 / 2)) * exp(-0.5 * mahalanobis(X, mu, sigma))
    # R's Mahalanobis: x is an (n x p) matrix of data
    #                  mu is the mean vector of the distribution or the second data vector of length p
    #                  sigma is the covariance matrix (p x p) of the distribution
    # D^2 = (x-mu)' * (sigma**-1(x-mu)
    centered_array = x - mu
    inv_cov = np.linalg.inv(sigma)
    mahal_dist = []
    for i in range(centered_array.shape[0]):
        mahal_dist.append(np.dot(np.dot(centered_array[i, :], inv_cov), centered_array[i, :]))
    assert len(mahal_dist) == x.shape[0]
    result = ((2 * np.pi) ** (-0.5 * len(mu))) * (np.linalg.det(sigma) ** (-1 / 2)) * np.exp(
        -0.5 * np.asarray(mahal_dist))
    return result


@jit(nopython=True)
def dmnorm_tk(x, mu, lambd):
    """
    Multivariate normal density sigma=lambd*ID
    :param x: 2D array of data [samples, dimensions]
    :param mu: center of the cluster [dimensions, ]
    :param lambd: one number - diagonal value for tkmeans (whatever that means)
    :return:
    """
    a = ((2 * np.pi) ** (-0.5 * len(mu))) * (lambd ** (-len(mu) / 2))
    b = (np.multiply(np.ones((x.shape[0], 1)), mu) - x) ** 2  # we cannot use numba for this function while using matmul
    c = np.exp(-(0.5 / lambd) * np.sum(b, axis=1))
    return a * c
