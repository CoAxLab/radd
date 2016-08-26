#!/usr/local/bin/env python
# Authors:
#   Kyle Dunovan (ked64@pitt.edu) &
#   Jeremy Huang (jeremyhuang.cmu.edu)
import numpy as np
from numba import boolean
from numba.decorators import jit
from numba import vectorize, float64

@vectorize([float64(boolean, float64, float64)])
def ufunc_where(condition, x, y):
    if condition:
        return x
    else:
        return y

@jit(nopython=True)
def numba_argmax_func(trace, onset, upper, dt):
    i=0
    for val in trace:
        i=i+1*dt
        if val>=upper:
            return onset + i
    return i

@vectorize([float64(float64, float64)])
def multiply(x, y):
    return x * y

@jit(float64[:](float64[:]), nopython=True)
def numba_1d_mutating_cumulative_sum(numpyArray):
    length = numpyArray.shape[0]
    runningSum = 0
    for i in xrange(length):
        runningSum += numpyArray[i]
        numpyArray[i] = runningSum
    return numpyArray

@jit(float64[:](float64[:]))
def numba_cumsum(x):
    return np.cumsum(x)

@jit(float64[:](float64[:], float64[:]))
def numba_argmax(trace, bound):
    return np.argmax(trace.T>=bound, axis=2)
