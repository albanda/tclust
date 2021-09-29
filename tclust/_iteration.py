import numpy as np


class Iteration(object):
    """
    Helper class for TClust package.
    
    Attributes
    ----------
    obj : float
        Value of the objective function of the current solution.
    
    labels_ : ndarray
        Cluster assignment for each observation. Shape=(nobs, ).
    
    csize : ndarray
        Number of observations that belong to each cluster. Shape=(1, k).
    
    cw : ndarray
        Weights of each cluster. Shape=(k, )
    
    sigma : ndarray
        Covariance matrices of each cluster. Shape=(ndim, ndim, k).
    
    center : ndarray
        Centers of each cluster (column-wise). Shape=(k, ndim).
    
    code : int
        Return code (supplied by functions like findClusterLabels).
    
    z_ij : ndarray
        Cluster assignment for each sample. Shape=(nobs, k).
    
    lambd : ndarray
        Diagonal values for tkmeans. Shape=(k, ).
    
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
        
        Parameters
        __________
        nobs: int
            Number of observations in the sample.
            
        ndim : int
            Number of dimensions.
            
        k : int
            Number of clusters
        """
        #print("Creating Iteration object with nobs=%d, ndim=%d, k=%d" % (nobs, ndim, k))
        self.obj = np.nan
        self.labels_ = np.zeros(nobs)
        self.csize = np.nan * np.ones((1, k))
        self.cw = np.asarray([np.nan] * k)
        self.sigma = np.nan * np.ones((ndim, ndim, k))
        self.center = np.nan * np.ones((k, ndim))
        self.code = np.nan
        self.z_ij = np.zeros((nobs, k))
        self.lambd = np.asarray([np.nan] * k)
        return self
    
    def update(self, obj=None, labels_=None, csize=None, cw=None, sigma=None, center=None, z_ij=None, lambd=None,
               code=None):
        """
        Updates the values of the Iteration object.
        
        :param obj: value of the objective function
        :param labels_: cluster assignment
        :param csize: cluster sizes
        :param cw: cluster weights
        :param sigma: covariance matrices of each cluster
        :param center: centers of each cluster (column-wise)
        :param z_ij: cluster assignment
        :param lambd: diagonal values for tkmeans
        :param code: return code
        :return: self
        """
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
