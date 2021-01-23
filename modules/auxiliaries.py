import numpy as np
import time

def freq_list_decorator(func):
  def wrapper(*args,**kwargs):
    return np.squeeze(np.asarray(list(map(func, *args, **kwargs))))
  return wrapper

def timit(func):
    def wrapper(*args,**kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        print('func:{}\nruntime: {:2.4f} s'.format(func.__name__, te-ts))
        return result
    return wrapper

def print_characteristics(args):
    def printer(arg):
        print("type:\t{}".format(type(arg)))
        try:
            print("shape:\t{}".format(arg.shape))
            print("dtype:\t{}".format(arg.dtype))
        except:
            pass
        print()
    if type(args)!=list:
        printer(args)
    else:
        for arg in args:
            printer(arg)