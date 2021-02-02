import numpy as np
from permutation import get_permutation_matrix, find_permutations


def init_centers(r):
    N, K = r.shape
    scale = abs(r).mean()
    c = scale*np.random.randn(K, 1)
    return np.repeat(c, N, axis=1).T


def update_centroids(r, perms):
    F, _ = r.shape
    P = get_permutation_matrix(perms)
    r_perm = np.asarray(list(map(np.dot, P, r)))
    c = np.reshape(r_perm.mean(0), (-1, 1))
    return np.repeat(c, F, axis=1).T


def cluster(r, max_iter=100):
    c_ = init_centers(r)
    for i in range(max_iter):
        cc = c_.copy()
        perms = find_permutations(r, c_)
        c_ = update_centroids(r, perms)

        dc = abs(c_-cc)
        if dc.all() < 1e-12:
            print("Converged at itr={}".format(i))
            break
    return perms, c_
