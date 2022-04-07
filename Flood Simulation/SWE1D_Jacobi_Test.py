from Create1DScene import Create1DScene
import copy
import numpy as np
from SWE1D_Jacobi import SWE1D_Jacobi
import matplotlib.pyplot as plt
from scipy.linalg import norm


hillHeight = 1
[b, d, h] = Create1DScene(hillHeight)

# Plotting
x = np.arange(1000) * 0.01
minx = min(x) - 0.5
maxx = max(x) + 0.5
miny = min(h) - 1.0
maxy = max(h) + 1.0

# parameters
g = 9.81           # unit: meter / second^2
delta_x = 0.1     # unit: meter
delta_t = 0.1     # unit: second
tau = 0.0001        # artificial viscosity

k = 0.5 * g * (delta_t / delta_x)**2

# convergence parameters
maximumJacobiIteration = 1000
JacobiConvergenceTolerance = 1e-6
fluidFlowConvergenceTolerance = 1e-4

h_prior = h
maximumIteration = 10000

for n in range(maximumIteration):
    [d, h_next] = SWE1D_Jacobi(b, d, h, h_prior, k, tau, maximumJacobiIteration, JacobiConvergenceTolerance)
    h_prior = h
    h = h_next

    # visualize fluid flow
    showAnimation = 1

    if showAnimation:

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


