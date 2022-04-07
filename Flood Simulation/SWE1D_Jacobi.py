import time
import numpy as np
from scipy import sparse
from scipy.linalg import norm
import copy
import matplotlib.pyplot as plt
from Create1DScene import Create1DScene
from numpy import sum


#####################################
# Argüman alma kismi alinma kısmı yok

# function [d, h] = SWE1D_Jacobi(b, d, h, h_prior, k, tau, maximumJacobiIteration, JacobiConvergenceTolerance)

# initialize base and water depth

def SWE1D_Jacobi(b, d, h, h_prior, k, tau, maximumJacobiIteration, JacobiConvergenceTolerance):

    totalVolume = sum(d)

    if totalVolume < 1e-3:
        return

    r = len(h)

    # march fluid

    # y vector
    y = h + (1 - tau) * (h - h_prior)
    # h_prior = copy.deepcopy(h)

    # solve linear system Ah = y
    d_first_element = d[0].reshape(1, 1)
    d_last_element = d[-1].reshape(1, 1)
    d_p = np.vstack((-1 * d_first_element, d[:-1]))
    d_n = np.vstack((d[1:], d_last_element * -1))

    q = sum([d_p, 2 * d, d_n], axis=0)
    w = sum([d, d_n], axis=0)
    e = k * q
    f = -k * w

    # Iterative Weighted Jacobi solver
    # https://en.wikipedia.org/wiki/Jacobi_method
    # A = (sparse.spdiags(np.array(([f, e])).reshape(2, r), [-1, 0], r, r) + sparse.spdiags(np.array(f).reshape(1, r), -1,
    #                                                                                    r, r).T)
    Dinv = sparse.spdiags((1. / (e + 1)).reshape(1, r), 0, r, r)
    L = sparse.spdiags(f.reshape(1, r), -1, r, r)
    U = sparse.spdiags(f.reshape(1, r), -1, r, r).T

    R = L + U
    yd = Dinv * y
    Rd = Dinv * R
    for t in range(maximumJacobiIteration):
        # h_next = (w * Dinv) * (y - R * h) + (1 - w) * h
        w = 2 / 3
        h_next = w * (yd - Rd * h) + (1 - w) * h
        if norm(h - h_next, ord=np.inf) < JacobiConvergenceTolerance:
            break

        # prevent negative volumes
        d_next = h_next - b

        d_next[np.logical_and(np.array(d_next) < 0, np.array(d) < 1e-4)] = 0
        d_next[d_next < 1e-4] = d[d_next < 1e-4] * 0.5
        # d_next[h_next - b < 0] = 0

        # preserve volume
        currentVolume = sum(d_next)
        if currentVolume > 0:
            d = d_next * (totalVolume / currentVolume)
            # calculate new surface height
            h = b + d
        else:
            break

    return d, h_next
