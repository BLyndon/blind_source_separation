import numpy as np
from numpy.linalg import *
from auxiliaries import timit, freq_list_decorator

@freq_list_decorator
def TDOA_estimator(Wf, J=0):
    W,f = Wf
    try:
        A = inv(W)
    except:
        A = pinv(W)

    M = A.shape[0]
    a_J = A[J,:]
    A_J = np.repeat([a_J],M,axis=0)

    estimator = - np.angle(A/A_J)/(2*np.pi*f)
    estimator = np.delete(estimator,J,0)

    return estimator

from itertools import permutations

@freq_list_decorator
def find_permutations(r,c):
    def sum_sq(v,c,perm):
        sum=0
        for k in range(c.shape[0]):
            sum += norm(v[perm[k]]-c[k])**2
        return sum
    perms = list(permutations(range(len(c))))
    sums = [sum_sq(r,c,perm) for perm in perms]
    return perms[np.argmin(sums)]

from scipy.sparse import coo_matrix

@freq_list_decorator
def get_permutation_matrix(perm):
    K = len(perm)
    rows = list(range(K))
    cols = perm
    data = np.ones(K)
    return coo_matrix((data,(rows,cols)),shape=(K, K)).toarray()