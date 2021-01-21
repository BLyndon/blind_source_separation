import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt

import warnings

def sym_decorrelation(W):
    """ Symmetric decorrelation
    W <- (W * W.T) ^{-1/2} * W
    """
    w, v = eigh(np.dot(W, W.conj().T))
    return multi_dot([v * (1. / np.sqrt(w)), v.conj().T, W])


def whitening(X):
    """ Whitening transformation
    Transform random vector X to whitened random vector Y=VX with unit diagonal covariance
    """
    X -= X.mean(1,keepdims=True)
    w, v = eigh(np.cov(X))
    D = np.diag(w)
    V = np.sqrt(inv(D)).dot(v.conj().T)
    Z = V.dot(X)
    return V, Z

def complex_FastICA(Z, U=None, tol=1e-6, max_iter=1000, alpha=0.1, history=True):
    def _g(y, alpha=alpha):
        g = y.conj()/(2 * np.sqrt(np.abs(y)**2 + alpha)) 
        dg = ((1 - 0.5 * (np.abs(y)**2/(np.abs(y)**2 + alpha)))/(2 * np.sqrt(np.abs(y)**2 + alpha))).mean(1)
        return g, dg
    def _logcosh(Y, alpha=alpha):
        Y *= alpha
        g = np.tanh(Y, Y)
        dg = np.empty(Y.shape[0])
        for i, g_i in enumerate(g):
            dg[i] = (alpha * (1 - g_i ** 2)).mean()
        return g, dg

    N, M = Z.shape
    
    if history:
        lims = list()

    if U==None:
        U = np.asarray(np.random.randn(N,N), dtype=Z.dtype)
        U = sym_decorrelation(U)

    for _ in range(max_iter):
        if Z.dtype==np.complex:
            g, dg = _g(U.dot(Z))
        else:
            g, dg = _logcosh(U.dot(Z))

        U1 = np.diag(dg).dot(U) - g.conj().dot(Z.conj().T)/M
        U1 = sym_decorrelation(U1)

        lim = max(abs(abs(np.diag(U1.dot(U.T))) - 1))
        U = U1

        if history:
            lims.append(lim)

        if lim < tol:
            break
    else:
        warnings.warn('FastICA did not converge.')

    if history:
        plt.plot(lims)
        plt.show()

    Y = U.dot(Z)

    return U, Y

def MLE(X, V, U, step_size=1e-5, max_iter=1000, tol=1e-5, history=True):
    def _g(y):
        return y/abs(y)
        
    N, _ = X.shape

    if history:
        lims = list()

    W = U.dot(V)

    for _ in range(max_iter):
        Y = W.dot(X)
        
        if X.dtype==np.complex:
            phi = _g(Y)
        else:
            phi = np.tanh(Y)
        
        step = step_size*(np.eye(N) - phi.dot(Y.conj().T)).dot(W)
        W1 = W + step

        lim = np.max(abs(step))
        W = W1

        if history:
            lims.append(lim)

        if lim < tol:
            break
    else:
        warnings.warn('MLE did not converge.')

    if history:
        plt.plot(lims)
        plt.show()

    Y = W.dot(X)

    return W, Y