import numpy as np


class Iteration(object):
    """
    Helper class for TClust package.
    """
    
    def __init__(self):
        self.obj = -np.inf
        self.labels_ = None
        self.csize = None
        self.cw = None
        self.sigma = None
        self.center = None
        self.code = np.nan
        self.z_ij = None
        self.lambd = None
    
    def fill(self, nobs, ndim, k):
        """

        :param nobs: Number of observations in the sample
        :param ndim: Number of dimensions
        :param k: Number of clusters
        """
        print("Creating Iteration object with nobs=%d, ndim=%d, k=%d" % (nobs, ndim, k))
        self.obj = np.nan  # The value of the objective function of the best solution
        self.labels_ = np.zeros(nobs)  # Numerical vector of size n containing the cluster assignment for each observation
        self.csize = np.nan * np.ones((1, k))  # Integer vector of length k,
                                               # returning the number of observations contained by each cluster
        self.cw = np.asarray([np.nan] * k)  # Numerical vector of length k, containing the weights of each cluster
        self.sigma = np.nan * np.ones((ndim, ndim, k))  # Array of size ndim x ndim x k containing
                                                        # the covariance matrices of each cluster
        self.center = np.nan * np.ones((k, ndim))  # Matrix of size k x ndim containing the centers
                                                    # of each cluster columnwise
        self.code = np.nan  # Return code supplied by functions like findClusterLabels
        self.z_ij = np.zeros((nobs, k))  # Cluster assignment given by 0/1 columns (for hard)
        self.lambd = np.asarray([np.nan] * k)  # Diagonal values for tkmeans  # lambda in the R code
        return self
    
    def update(self, obj=None, labels_=None, csize=None, cw=None, sigma=None, center=None, z_ij=None, lambd=None,
               code=None):
        if obj is not None:
            self.obj = obj
        if labels_ is not None:
            self.labels_ = np.copy(labels_)
        if csize is not None:
            self.csize = np.copy(csize)
        if cw is not None:
            self.cw = np.copy(cw)
        if sigma is not None:
            self.sigma = np.copy(sigma)
        if center is not None:
            self.center = np.copy(center)
        if z_ij is not None:
            self.z_ij = np.copy(z_ij)
        if lambd is not None:
            self.lambd = np.copy(lambd)
        if code is not None:
            self.code = np.copy(code)
        return self
