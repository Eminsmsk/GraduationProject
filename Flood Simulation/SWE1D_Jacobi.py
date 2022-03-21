import time
import numpy as np
from scipy import sparse
from scipy.linalg import norm
import copy
import matplotlib.pyplot as plt
from Create1DScene import Create1DScene

# initialize base and water depth
hillHeight = 1
[b, d, hPlot] = Create1DScene(hillHeight)

h = b + d
h_prior = copy.deepcopy(h)
x = np.arange(1000) * 0.01
minx = min(x) - 0.5
maxx = max(x) + 0.5
miny = min(h) - 1.0
maxy = max(h) + 1.0

r = len(h)
totalVolume = sum(d)

# parameters
g = 9.81           # unit: meter / second^2
delta_x = 0.1     # unit: meter
delta_t = 0.1     # unit: second
tau = 0.001        # artificial viscosity

k = g * (delta_t / delta_x)**2

# convergence parameters
maximumJacobiIteration = 100
JacobiConvergenceTolerance = 1e-6
fluidFlowConvergenceTolerance = 1e-4

# march fluid
tic = time.time()
for n in range(10000):
    # y vector
    y = h + (1 - tau) * (h - h_prior)
    h_prior = copy.deepcopy(h)

    # solve linear system Ah = y
    aa = d[0].reshape(1, 1)
    bb = d[-1].reshape(1, 1)
    d_p = np.vstack((-1 * aa, d[:-1]))
    d_n = np.vstack((d[1:], bb * -1))
    from numpy import sum
    q = sum([d_p, 2 * d, d_n], axis=0)
    w = sum([d, d_n], axis=0)
    e = 1.0 + 0.5 * k * q
    f = -0.5 * k * w


    # Iterative Weighted Jacobi solver
    # https://en.wikipedia.org/wiki/Jacobi_method
    # A = (sparse.spdiags(np.array(([f, e])).reshape(2, r), [-1, 0], r, r) + sparse.spdiags(np.array(f).reshape(1, r), -1,
    #                                                                                       r, r).T)
    Dinv = sparse.spdiags((1. / e).reshape(1, r), 0, r, r)
    L= sparse.spdiags(f.reshape(1, r), -1, r, r)
    U = sparse.spdiags(f.reshape(1, r), -1, r, r).T


    R = L + U
    yd = Dinv * y
    Rd = Dinv * R
    for t in range(maximumJacobiIteration):
        # h_next = (w * Dinv) * (y - R * h) + (1 - w) * h
        w = 0.9
        h_next = w * (yd - Rd * h) + (1 - w) * h

        if norm(h - h_next, ord=np.inf) < JacobiConvergenceTolerance:
            break

        # prevent negative volumes
        d = h_next - b
        d[d < 0] = 0

        # preserve volume
        currentVolume = sum(d)
        d = d * (totalVolume / currentVolume)

        #calculate new surface height
        h = b + d

    # visualize fluid flow
    showAnimation = 1
    if (showAnimation):
        plt.figure(1)
        a = plt.plot(x, h, 'b', 'LineWidth', 2)
        plt.plot(x, b, 'k', 'LineWidth', 2)
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.pause(0.005)
        plt.clf()
    if norm(h - h_prior, ord=np.inf) < fluidFlowConvergenceTolerance:
        break

plt.show()
toc = time.time()
totalTime = toc-tic

# end of fluid flow
print('Simulation time is %.4f seconds \n', n * delta_t)
print('Converged in %d iterations \n\n', n)

print('Computation time is %.4f seconds \n', totalTime)
print('Computation time per frame is %.5f seconds \n', totalTime / n)