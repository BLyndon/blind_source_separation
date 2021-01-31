import numpy as np
from scipy.sparse import coo_matrix
from auxiliaries import freq_list_decorator
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
    c = init_centers(r)
    for i in range(max_iter):
        c_ = c
        perms = find_permutations(r, c)
        c = update_centroids(r, perms)

        dc = abs(c-c_)
        if dc.all() < 1e-12:
            print("Converged at itr={}".format(i))
            break
    return perms, c
