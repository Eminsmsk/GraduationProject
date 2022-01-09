import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from PIL import Image
import cv2


def horizontal2D(b, d, h, b_height, n, h_prev):
    for i in range(b_height):
        b_i = b[i]
        d_i = d[i]
        h_i = h[i]

        b_r, d_r, h_r, h_prev[i] = simulateEq21_2D(b_i, d_i, h_i, n, h_prev[i])
        b[i] = b_r
        d[i] = d_r
        h[i] = h_r

    return b, d, h, h_prev


def vertical2D(b, d, h, b_width, n, h_prev):
    b2 = b.T
    d2 = d.T
    h2 = h.T
    h_prev2 = h_prev.T

    for i in range(b_width):
        b2[i], d2[i], h2[i], h_prev2[i] = simulateEq21_2D(b2[i], d2[i], h2[i], n, h2[i])
    b = b2.T
    d = d2.T
    h = h2.T
    h_prev = h_prev2.T
    return b, d, h, h_prev


def normalize_d(d):

    if d.sum() > 24500:
        print(d.sum())
        dif = d.sum() - 24500
        # print(dif)
        counter = 0
        for r in range(d.shape[0]):
            for c in range(d.shape[1]):
                if d[r][c] > 0:
                    counter = counter + 1


        fark = (dif / (counter))
        print('dif : ', dif)
        print('fark : ', fark)
        print('number : ', counter)
        print('res : ', (counter) * fark)

        for r in range(d.shape[0]):
            for c in range(d.shape[1]):
                if d[r][c] >= 0:
                    d[r][c] = d[r][c] - fark
                if d[r][c] < 0:
                    d[r][c] = 0

        print('üst sum : ', d.sum())

    return d


def simulateEq21_2D(one_d_b, one_d_d, one_d_h, n, one_d_h_prev):
    tau = 0.01

    e = calcE(one_d_d)
    f = calcF(one_d_d)

    A = diags(e) + diags(f, -1) + diags(f, +1)

    if n == 0:
        y = one_d_b + one_d_d
    else:
        y = one_d_h + (1 - tau) * (one_d_h - one_d_h_prev)

    one_d_h_prev = copy.deepcopy(one_d_h)

    one_d_h = spsolve(A, y)
    one_d_d = one_d_h - one_d_b
    for k in range(len(one_d_d)):
        if (one_d_d[k] < 0):
            one_d_d[k] = 0

    one_d_h = one_d_b + one_d_d

    return one_d_b, one_d_d, one_d_h, one_d_h_prev


def calcE(one_d_data):
    dt = 0.1
    dx = 0.1
    k = 9.8 * (dt ** 2) / (2 * (dx ** 2))
    e = np.zeros(len(one_d_data), dtype=(np.float32))

    for i in range(0, len(one_d_data)):
        if (i == 0):
            e[i] = 1 + (k * (one_d_data[0] + one_d_data[1]))
        elif (0 < i and i < len(one_d_data) - 1):
            e[i] = 1 + (k * (one_d_data[i - 1] + 2 * one_d_data[i] + one_d_data[i + 1]))
        elif (i == len(one_d_data) - 1):
            e[i] = 1 + (k * (one_d_data[len(one_d_data) - 2] + one_d_data[len(one_d_data) - 1]))

    return e


def calcF(one_d_data):
    dt = 0.1
    dx = 0.1
    k = 9.8 * (dt ** 2) / (2 * (dx ** 2))
    f = np.zeros(len(one_d_data) - 1, dtype=(np.float32))

    for i in range(0, len(one_d_data) - 1):
        f[i] = k * (one_d_data[i] + one_d_data[i + 1]) * -1.0

    return f


def simulate2D(b, d, h, h_prev):
    dt = 0.1
    b_width = b.shape[0]
    b_height = b.shape[1]
    for i in range(10000):
        # d = normalize_d(d)
        # print('sum : ', d.sum())

        b_temp = copy.deepcopy(b)
        d_temp = copy.deepcopy(d)
        h_temp = copy.deepcopy(h)
        h_prev_temp = copy.deepcopy(h_prev)
        h_prev_temp_ver = copy.deepcopy(h_prev)



        b_temp, d_temp, h_temp, h_prev_temp = horizontal2D(b_temp, d_temp, h_temp, b_height, i, h_prev_temp)

        b_temp, d_temp, h_temp, h_prev_temp = vertical2D(b_temp, d_temp, h_temp, b_width, i, h_prev_temp_ver)
        h_prev = h
        h = h_temp
        b = b_temp
        d = d_temp

        empthy = np.zeros_like(d)
        res = np.empty((len(d), len(d), 3))
        alpha = d.copy()
        alpha = cv2.normalize(alpha, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        beta = b.copy()
        beta = cv2.normalize(beta, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        res[:, :, 0] = empthy
        res[:, :, 1] = empthy
        res[:, :, 2] = alpha
        plt.figure(1)
        plt.imshow(res)
        plt.figure(2)
        plt.imshow(b, cmap='gray')
        plt.pause(0.1)
    plt.show()


def createEnv(base):
    b = base
    d = np.zeros((b.shape[0], b.shape[1]), dtype=(np.float32))

    for j in range(150, 199):
        for i in range(0, 50):
            d[j][i] = 10.0
    print('aaa : ', d.sum())
    h_prev = b + d
    h = b + d
    return b, d, h, h_prev


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt

    a = np.array(Image.open('Dem.tif'))
    # plt.imshow(a)
    # plt.show()
    b = a[600:799, 600:799]
    print(b.shape)
    b, d, h, h_prev = createEnv(b)
    simulate2D(b, d, h, h_prev)
