import numpy as np


class Iteration(object):
    """

    """
    
    def __init__(self):
        self.obj = -np.inf
        self.code = np.nan
        self.labels_ = None
        self.csize = None
        self.cw = None
        self.sigma = None
        self.center = None
        self.z_ij = None
        self.lambd = None
    
    def fill(self, nobs, ndim, k):
        """

        :param nobs: Number of observations in the sample
        :param ndim: Number of dimensions
        :param k: Number of clusters
        """
        self.obj = np.nan  # The value of the objective function of the best solution
        self.labels_ = np.zeros(
            nobs)  # A numerical vector of size n containing the cluster labels_nment for each observation
        self.csize = np.nan * np.ones(
            (1, k))  # An integer vector of length k, returning the number of observations contained by each cluster
        self.cw = np.asarray([np.nan] * k)  # A numerical vector of length k, containing the weights of each cluster
        self.sigma = np.nan * np.ones(
            (ndim, ndim, k))  # An array of size ndim x ndim x k containing the covariance matrices of each cluster
        self.center = np.nan * np.ones(
            (k, ndim))  # A matrix of size k x ndim containing the centers of each cluster columnwise
        self.code = np.nan  # this is a return code supplied by functions like findClustLabels_
        self.z_ij = np.zeros((nobs, k))  # cluster assignment given by 0/1 columns (for hard)
        self.lambd = np.asarray([np.nan] * k)  # diagonal values for tkmeans  # lambda in the R code
        return self
    
    def update(self, obj=None, labels_=None, csize=None, cw=None, sigma=None, center=None, z_ij=None, lambd=None,
               code=None):
        if obj is not None:
            self.obj = obj
        if labels_ is not None:
            self.labels_ = labels_
        if csize is not None:
            self.csize = csize
        if cw is not None:
            self.cw = cw
        if sigma is not None:
            self.sigma = sigma
        if center is not None:
            self.center = center
        if z_ij is not None:
            self.z_ij = z_ij
        if lambd is not None:
            self.lambd = lambd
        if code is not None:
            self.code = code
        return self
