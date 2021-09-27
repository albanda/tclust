from tclust import TClust
import numpy as np
#from numpy.matlib import repmat
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import pandas as pd


def tkmeans(X, k, alpha=0.05, niter=20, ksteps=10, equal_weights=False, maxfact_d=5, m=2., zero_tol=1e-16, trace=0,
            opt="hard", sol_ini=None):
    if X.shape[1] > 1:
        return TClust(k=k, alpha=alpha, n_inits=niter, ksteps=ksteps, equal_weights=equal_weights,
                      restr_cov_value='deter', maxfact_e=1, maxfact_d=maxfact_d, m=m, zero_tol=zero_tol, trace=trace,
                      opt=opt, sol_ini=sol_ini, tk=True).fit(X)
    elif X.shape[1] == 1:
        return TClust(k=k, alpha=alpha, n_inits=niter, ksteps=ksteps, equal_weights=equal_weights,
                      restr_cov_value='deter', maxfact_e=1, maxfact_d=maxfact_d, m=m, zero_tol=zero_tol, trace=trace,
                      opt=opt, sol_ini=sol_ini, tk=False).fit(X)


def main(doX=True, doY=True):
    nsamp = 200
    nfeat = 2
    gauss = np.random.randn
    u = gauss(nsamp * nfeat).reshape(nsamp, nfeat)  # standard normal distribution
    v = np.cov(u.T)
    eig_values, eig_vectors = np.linalg.eig(v)
    
    if doX:
        '''
        x1 = gauss(nsamp * nfeat).reshape(nsamp, nfeat).dot(np.array([[1, 0], [0, 9]])).dot(eig_vectors) + \
             repmat(np.array([20, 20]).reshape(1, -1), nsamp, 1)
        x2 = gauss(nsamp * nfeat).reshape(nsamp, nfeat).dot(np.array([[9, 0], [0, 1]])).dot(eig_vectors) + \
             repmat(np.array([-20, -20]).reshape(1, -1), nsamp, 1)
        x3 = gauss(nsamp * nfeat).reshape(nsamp, nfeat).dot(np.array([[3, 0], [0, 3]])).dot(eig_vectors) + \
             repmat(np.array([0, 0]).reshape(1, -1), nsamp, 1)
        x4 = gauss(nsamp * nfeat).reshape(nsamp, nfeat).dot(np.array([[25, 0], [0, 25]])).dot(eig_vectors) + \
             repmat(np.array([2, 3]).reshape(1, -1), nsamp, 1)
        x = np.concatenate((x1, x2, x3, x4), axis=0)  # shape=[800, 2]
        '''
        x = pd.read_csv('x.csv', header=None).values
        
        clusteringX = TClust(k=3, alpha=0.25, n_inits=200, ksteps=40, equal_weights=False, restr_cov_value='deter',
                             maxfact_e=1e10, maxfact_d=10, m=1.1, zero_tol=1e-16, trace=0, opt='mixture', sol_ini=None,
                             tk=False)
        t0 = time.clock()
        clusteringX.fit(x)
        print('Time elapsed for x = %.2f s' % (time.clock() - t0))
        print(set(clusteringX.best_iter.labels_))
        plt.scatter(x[:, 0], x[:, 1], c=clusteringX.best_iter.labels_)
        label = [0] * nsamp + [1] * nsamp + [2] * nsamp + [3] * nsamp
        print(confusion_matrix(label, clusteringX.best_iter.labels_))
        plt.show()
    
    if doY:
        '''
        y1 = gauss(nsamp * nfeat).reshape(nsamp, nfeat).dot(np.array([[1, 0], [0, 1]])).dot(eig_vectors) + \
             repmat(np.array([2.5, 3]).reshape(1, -1), nsamp, 1)
        y2 = gauss(nsamp * nfeat).reshape(nsamp, nfeat).dot(np.array([[1, 0], [0, 1]])).dot(eig_vectors) + \
             repmat(np.array([-2.5, 3]).reshape(1, -1), nsamp, 1)
        y3 = gauss(nsamp * nfeat).reshape(nsamp, nfeat).dot(np.array([[1, 0], [0, 1]])).dot(eig_vectors) + \
             repmat(np.array([0, 0]).reshape(1, -1), nsamp, 1)
        y4 = gauss(nsamp * nfeat).reshape(nsamp, nfeat).dot(np.array([[8, 0], [0, 8]])).dot(eig_vectors) + \
             repmat(np.array([0, 2]).reshape(1, -1), nsamp, 1)
        y = np.concatenate((y1, y2, y3, y4), axis=0)  # [800, 2]
        '''
        y = pd.read_csv('y.csv', header=None).values
        
        t0 = time.clock()
        clusteringY = tkmeans(X=y, k=3, alpha=0.25, niter=200, ksteps=40, equal_weights=False, maxfact_d=10, m=1.1,
                              zero_tol=1e-16, trace=0, opt='fuzzy', sol_ini=None)
        print('Time elapsed for y = %.2f s' % (time.clock() - t0))
        print(set(clusteringY.best_iter.labels_))
        plt.scatter(y[:, 0], y[:, 1], c=clusteringY.best_iter.labels_)
        plt.show()
        label = [1] * nsamp + [2] * nsamp + [3] * nsamp + [4] * nsamp
        print(confusion_matrix(label, clusteringY.best_iter.labels_))
        plt.show()


if __name__ == '__main__':
    tests = [4]#range(1, 5)
    for t in tests:
        if t == 1:
            print("TEST 1")
            X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
            clustering = TClust(k=2)
            clustering.fit(X)
        elif t == 2:
            print("TEST 2")
            X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
            clustering = tkmeans(X, 2)
        elif t == 3:
            from sklearn.datasets import make_blobs
            print("TEST 3: BLOBS!")
            X, y = make_blobs(n_samples=1500, n_features=5, centers=3, random_state=170)
            print(X.shape)
            clustering = TClust(k=3, alpha=0)
            clustering.fit(X)
            print("y.set", set(y), 'labels set', set(clustering.best_iter.labels_))
            print(confusion_matrix(y, clustering.best_iter.labels_))
        elif t == 4:
            main(doX=True, doY=True)
            # Time elapsed for x: 28.32 / 42.13
            # Time elapsed for y = 35.17 s / 23.22 s
        if t != 4:
            best = clustering.best_iter
            print('Best obj = %.2f' % best.obj)
            print('labels:', clustering.labels_)
            print('csize', best.csize)
            print('cw', best.cw)
            print('code', best.code)
            print('center', best.center)
