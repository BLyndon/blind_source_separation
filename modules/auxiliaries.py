import numpy as np
import time
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt


def freq_list_decorator(func):
    def wrapper(*args, **kwargs):
        return np.squeeze(np.asarray(list(map(func, *args, **kwargs))))
    return wrapper


def timit(func):
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('func:{}\nruntime: {:2.4f} s'.format(func.__name__, te-ts))
        return result
    return wrapper


def print_characteristics(args):
    def printer(arg):
        print("type:\t{}".format(type(arg)))
        if type(arg) == list:
            print('length:\t{}'.format(len(arg)))
        else:
            try:
                print("shape:\t{}".format(arg.shape))
                print("dtype:\t{}".format(arg.dtype))
            except:
                pass
        print()
    if type(args) != list:
        printer(args)
    else:
        for arg in args:
            printer(arg)


def get_unmixing(W):
    try:
        A = inv(W)
    except:
        A = pinv(W)

    return A


def get_Y(Yf):
    F = len(Yf)
    D, T = Yf[0][0].shape
    Y = np.zeros((D, F, T), dtype=np.complex)
    for i in range(F):
        Y[:, i, :] = Yf[i][0]
    return Y


def spectogram(t, f, Z, vmax=None):
    D = Z.shape[0]
    plt.figure(figsize=(8, D*6))
    plt.title('STFT Magnitude')
    for d in range(D):
        if vmax == None:
            vmax = abs(Z[d]).mean()
        plt.subplot(D+1, 1, d+1)
        plt.pcolormesh(t, f, np.abs(Z[d]), vmin=0,
                       vmax=vmax, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
    plt.show()
