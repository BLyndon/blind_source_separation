import numpy as np
from numpy.linalg import *

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