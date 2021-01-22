import numpy as np
from numpy.linalg import *

def TDOA_estimator(Wf, J=0):
    def single_f_est(wf,J=J): 
        W,f = wf
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
    return np.squeeze(np.asarray(list(map(single_f_est, Wf))))

from itertools import permutations

def find_permutations(r,c):
    def find_single_f_perm(r,c):
        def sum_sq(v,c,perm):
            sum=0
            for k in range(c.shape[0]):
                sum += norm(v[perm[k]]-c[k])**2
            return sum
    
        perms = list(permutations(range(len(c))))
        sums = [sum_sq(r,c,perm) for perm in perms]
        return perms[np.argmin(sums)]
    return np.asarray(list(map(find_single_f_perm, r,c)))

from scipy.sparse import coo_matrix

def get_permutation_matrix(perms):
    def get_single_permutation_matrix(perm):
        K = len(perm)
        rows = list(range(K))
        cols = perm
        data = np.ones(K)
        return coo_matrix((data,(rows,cols)),shape=(K, K)).toarray()
    return np.asarray(list(map(get_single_permutation_matrix,perms)))