import numpy as np
from separation import whitening
from auxiliaries import freq_list_decorator, get_unmixing, get_Y


@freq_list_decorator
def get_theta(X, Wf):
    W = Wf[0]

    A = get_unmixing(W)
    V, Z = whitening(X)
    B = V.dot(A)

    BZ = B.conj().T.dot(Z)
    BZ_norm = np.sqrt(BZ*BZ.conj())

    B_norm = np.sqrt((B*B.conj()).sum(axis=0, keepdims=True)).T
    Z_norm = np.sqrt((Z*Z.conj()).sum(axis=0, keepdims=True))

    B_Z_norm = B_norm*Z_norm

    phi = BZ_norm/B_Z_norm
    return np.arccos(phi.astype(np.float64))


def mask(Xf_, Yf_, Wf_, theta_max=0.5):
    theta = get_theta(Xf_, Wf_)

    M = np.where(theta < theta_max, 1, 0)
    M = np.swapaxes(M, 0, 1)

    Y = get_Y(Yf_)
    return M*Y
