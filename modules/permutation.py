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
    rf_est = list(map(single_f_est, Wf))
    return np.squeeze(np.asarray(rf_est))

def init_centers(r):
    N, K = r.shape
    scale = abs(rf_est).mean()
    c = scale*np.random.randn(K,1)
    return np.repeat(c,N, axis=1).T

def find_permutations(r,c)
    def find_single_f_perm(arg):
        r, c = arg
        def sum_sq(v,c,perm):
            sum=0
            for k in range(c.shape[0]):
            sum += np.linalg.norm(v[perm[k]]-c[k])**2
            return sum
    
        perms = list(permutations(range(len(c))))
        sums = [sum_sq(v,c,perm) for perm in perms]
        return perms[np.argmin(sums)]
    
    perms = list(map(find_single_f_perm, [r,c]))
    return np.asarray(perms)