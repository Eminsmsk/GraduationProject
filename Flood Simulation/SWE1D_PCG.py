# initialize base (b) and water depth (d)
import numpy as np

import Create1DScene
import time
from scipy import sparse
from scipy.linalg import norm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
import ilupp
import copy




hillHeight = 1
b, d, a = Create1DScene.Create1DScene(hillHeight)
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

k = g * ((delta_t / delta_x)**2)


# convergence parameters
maximumJacobiIteration = 100
JacobiConvergenceTolerance = 1e-6
fluidFlowConvergenceTolerance = 1e-4

# march fluid
tic = time.time()
for n in range(10000):
    # y vector
    # y = 2 * h - h_prior;

    y = h + (1 - tau) * (h - h_prior)
    h_prior = copy.deepcopy(h)

    aa = d[0].reshape(1, 1)
    bb = d[-1].reshape(1, 1)
    d_p = np.vstack((-1 * aa, d[:-1]))
    d_n = np.vstack((d[1:], bb * -1))



    from numpy import sum
    q = sum([d_p,2 * d, d_n], axis=0)
    w = sum([d, d_n], axis=0)
    e = 1.0 + 0.5 * k * q
    f = -0.5 * k * w
    print(e.shape)
    print(f.shape)
    A = (sparse.spdiags(np.array(([f, e])).reshape(2, r), [-1, 0], r, r) + sparse.spdiags(np.array(f).reshape(1, r), -1, r, r).T) # trans almak yerine 1 yapıldı
    A = sparse.csr_matrix(A)
    # # Direct Solver
    # # h = A / y
    #
    # Iterative (Preconditioned Conjugate Gradient)

    # M1 = ilupp.ichol0(A)
    M1 = ilupp.IChol0Preconditioner(A)
    # print(M1)
    M2 = M1.T
    solverTolerance = 1e-5
    solverMaxIter = 1000
    h = cg(A, y, h,  tol=solverTolerance, maxiter=solverMaxIter, M=M1)
    h = h[0].reshape(r, 1)

    # prevent negative volumes
    d = h - b
    d[d < 0] = 0

    # Preserve volume
    currentVolume = sum(d)
    d = d * (totalVolume / currentVolume)

    # # Visualise fluid flow
    h = b + d

    showAnimation = 1
    if showAnimation:
        plt.figure(1)
        a = plt.plot(x, h, 'b', 'LineWidth', 2)
        plt.plot(x, b, 'k', 'LineWidth', 2)
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        plt.pause(0.005)
        plt.clf()


    # check for steady state
    if norm(h - h_prior) < fluidFlowConvergenceTolerance:
        break
plt.show()
toc = time.time()
totalTime = toc - tic

print('Simulation time is %.4f seconds \n', n * delta_t)
print('Converged in %d iterations \n\n', n)

print('Computation time is %.4f seconds \n', totalTime)
print('Computation time per frame is %.5f seconds \n', totalTime / n)

