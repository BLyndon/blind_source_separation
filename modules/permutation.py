import numpy as np
from numpy.linalg import norm
from auxiliaries import timit, freq_list_decorator, get_unmixing
from scipy.sparse import coo_matrix
from itertools import permutations


@freq_list_decorator
def TDOA_estimator(Wf, J=0):
    W, f = Wf
    assert f != 0, print("TDOA estimator: wanishing frequency!")
    A = get_unmixing(W)

    M = A.shape[0]
    a_J = A[J, :]
    A_J = np.repeat([a_J], M, axis=0)

    estimator = - np.angle(A/A_J)/(2*np.pi*f)
    estimator = np.delete(estimator, J, 0)

    return estimator


@freq_list_decorator
def find_permutations(r, c):
    def sum_sq(v, c, perm):
        sum = 0
        for k in range(c.shape[0]):
            sum += norm(v[perm[k]]-c[k])**2
        return sum
    perms = list(permutations(range(c.shape[0])))
    sums = [sum_sq(r, c, perm) for perm in perms]
    return perms[np.argmin(sums)]


@freq_list_decorator
def get_permutation_matrix(perm):
    K = perm.shape[0]
    rows = list(range(K))
    cols = perm
    data = np.ones(K)
    return coo_matrix((data, (rows, cols)), shape=(K, K)).toarray()


@freq_list_decorator
def permute_W(Wf, P):
    W, f = Wf

    W_ = W.dot(P.T)

    Wf_ = [W_, f]
    return Wf_


@freq_list_decorator
def permute_Y(Yf, P):
    Y, f = Yf

    Y_ = P.dot(Y)
    
    Yf_ = [Y_, f]
    return Yf_


@freq_list_decorator
def scale(Wf, Yf, J=0):
    Y, f = Yf

    A = get_unmixing(Wf[0])

    a_J = np.reshape(A[J, :], (-1, 1))
    Y_ = a_J*Y

    return [Y_, f]


def perm_alignment(r, Wf, Yf, perms):
    Pf = get_permutation_matrix(perms)

    Wf_ = permute_W(Wf, Pf)
    Yf_ = permute_Y(Yf, Pf)

    return Wf_, Yf_
