import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


def Create1DScene(hillHeight):
    r = 1000

    x = np.arange(r) * 0.01
    b = 0.25 * np.square(x - 5) + 0.1 * np.sin(2 * x) + 0.5
    b[0] = b[0] + 1
    b[-1] = b[-1] + 1
    b[500: 550] = 1.50 + hillHeight
    b[600: 700] = 0.05
    d = np.zeros((r, 1))
    d[100: 200] = 4.5
    d[800: 850] = 3.0
    b = np.array(b).reshape(r, 1)

    # smooth depth
    a = [0.1 * np.ones((r, 1)), 0.8 * np.ones((r, 1)), 0.1 * np.ones((r, 1))]
    D = sparse.spdiags(np.array(a).reshape(3, r), [-1, 0, 1], r, r)
    D = D.tolil()
    D[0, 0] = 1
    D[0, 1] = 0
    D[-1, -1] = 1
    D[-1, -2] = 0
    smoothingCount = 500
    for i in range(smoothingCount):
        b = D * b

    # initialize figure for animation
    h = b + d
    minx = min(x) - 0.5
    maxx = max(x) + 0.5
    miny = min(h) - 1.0
    maxy = max(h) + 1.0

    # plt.figure(1)
    # a = plt.plot(x, h, 'b', 'LineWidth', 2)
    # plt.plot(x, b, 'k', 'LineWidth', 2)
    # plt.xlim([minx, maxx])
    # plt.ylim([miny, maxy])
    # plt.show()
    return b, d, h
