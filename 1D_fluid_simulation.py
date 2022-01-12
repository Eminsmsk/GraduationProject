import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def calcE(one_d_data, k):
    e = np.zeros(len(one_d_data), dtype=np.float64)

    for i in range(0, len(one_d_data)):
        if (i == 0):
            e[i] = 1 + (k * (one_d_data[0] + one_d_data[1]))
        elif (0 < i and i < len(one_d_data) - 1):
            e[i] = 1 + (k * (one_d_data[i - 1] + 2 * one_d_data[i] + one_d_data[i + 1]))
        elif (i == len(one_d_data) - 1):
            e[i] = 1 + (k * (one_d_data[len(one_d_data) - 2] + one_d_data[len(one_d_data) - 1]))

    return e


def calcF(one_d_data, k):
    f = np.zeros(len(one_d_data) - 1, dtype=np.float64)

    for i in range(0, len(one_d_data) - 1):
        f[i] = k * (one_d_data[i] + one_d_data[i + 1]) * -1.0

    return f


def create_env(width):
    h = np.zeros(width, dtype=np.float64)
    b = np.zeros(width, dtype=np.float64)
    d = np.zeros(width, dtype=np.float64)

    x_axis = np.arange(-150, 150)

    for i in range(len(x_axis)):
        b[i] = (x_axis[i] ** 2) / 10000.0 + max(- (x_axis[i] * 0.03) ** 2 + 1, 0)

    for j in range(80, 100):
        d[j] = 3

    for j in range(220, 230):
        d[j] = 4

    return b, d, h


def simulateEq21(b, d, h, max_iter, dt, dx, tau, k):
    volume = sum(d)
    h_prev = copy.deepcopy(h)

    for n in range(2, max_iter):

        e = calcE(d, k)
        f = calcF(d, k)

        A = diags(e) + diags(f, -1) + diags(f, +1)
        A = csr_matrix(A)

        y = h + (1 - tau) * (h - h_prev)

        h_next = spsolve(A, y)
        h_prev = h
        h = h_next

        d = h - b
        d[d < 0] = 0

        d = d / d.sum() * volume
        h = b + d

        if n % 4 == 0:
            x_axis = np.arange(-150, 150)
            plt.plot(x_axis * dx, h, color="blue")
            plt.plot(x_axis * dx, b, color="red")
            plt.ylim(0, 8)

            plt.pause(0.01)
            plt.clf()

    plt.show()


def simulateEq45(b, d, h, max_iter, dt, dx, tau):
    h_prev = copy.deepcopy(h)
    h_next = np.zeros_like(h)
    for n in range(2, max_iter):

        e = calcE(d)
        f = calcF(d)
        y = h + (1 - tau) * (h - h_prev)

        for i in range(0, len(b)):
            if i == 0:
                h_next[i] = (y[i] - f[i] * h[i + 1]) / e[i]
            elif (i == len(b) - 1):
                h_next[i] = (y[i] - f[i - 1] * h[i - 1]) / e[i]
            else:
                h_next[i] = (y[i] - f[i - 1] * h[i - 1] - f[i] * h[i + 1]) / e[i]

        d = h_next - b
        d[d < 0] = 0

        h_next = b + d
        h_prev = copy.deepcopy(h)
        h = copy.deepcopy(h_next)

        x_axis = np.arange(-150, 150)
        plt.plot(x_axis * dx, h, color="blue")
        plt.plot(x_axis * dx, b, color="red")
        plt.ylim(0, 8)

        plt.pause(0.0001)
        plt.clf()
    plt.show()


if __name__ == '__main__':
    b, d, h = create_env(300)
    h = b + d
    tau = 0
    dt = 0.01
    dx = 0.01
    k = 9.8 * (dt ** 2) / (2 * (dx ** 2))
    simulateEq21(b, d, h, 10000, dt, dx, tau, k)
