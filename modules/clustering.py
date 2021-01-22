import numpy as np
from permutation import get_permutation_matrix

def init_centers(r):
    N, K = r.shape
    scale = abs(r).mean()
    c = scale*np.random.randn(K,1)
    return np.repeat(c,N, axis=1).T

def update_centroids(r, perms):
    F, K = r.shape
    P = get_permutation_matrix(perms)
    r_perm = np.asarray(list(map(np.dot, P,r)))
    c = np.reshape(r_perm.mean(0),(-1,1))
    return np.repeat(c, F, axis=1).T