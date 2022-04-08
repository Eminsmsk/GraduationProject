def SWE2D_Jacobi(B, D, H, H_prior, k_x, k_y, tau, maximumJacobiIteration=None, JacobiConvergenceTolerance=None):


    r, c = H.shape()
    n = r * c

    B = B.transpose()
    D = D.transpose()
    H = H.transpose()
    H_prior = H_prior.transpose()

    d = D.transpose().flatten()
    totalVolume = sum(d)

    if totalVolume < 1e-3:
        return

    b = B.transpose().flatten()
    h = H.transpose().flatten()
    h_prior = H_prior.transpose().flatten()

    if maximumJacobiIteration is None:
        maximumJacobiIteration = 25

    if JacobiConvergenceTolerance is None:
        JacobiConvergenceTolerance = 1e-5

    y = h + (1 - tau) * (h - h_prior.transpose().flatten())

    D_left = np.vstack((-D[0,:], D[0:-1,:]))
    D_right = np.vstack((D[1:, :], -D[-1, :]))
    D_up = np.hstack((-D[:, 0].reshape(-1,1), D[-1, :]))
    D_down = np.hstack((D[:, 1:-1], -D[:, -1]))

    e_x = k_x * (D_left + 2*D + D_right)
    f_x = -k_x * (D + D_right)
    e_y = k_y * (D_up + 2 * D + D_down)
    f_y = -k_y * (D + D_down)

    # spdiagslar

    # Dinv = spdiags(1. / (e_x(:) + e_y(:) + 1), 0, n, n);
    # L = spdiags(f_x(:), -1, n, n) + spdiags(f_y(:), -c, n, n);

    U = L.transpose()
    R: L + U
    yd = Dinv * y
    Rd = Dinv * R

    for t in range(maximumJacobiIteration):
        w = (2 / 3)
        # % h_next = (w * Dinv) * (y - R * h) + (1 - w) * h;
        h_next = w * (yd - Rd * h) + (1 - w) * h

        if (norm(h - h_next, Inf) < JacobiConvergenceTolerance)
            break

        d_next = h_next - b
        d_next(d_next < 0 & d < 1e-4) = 0;
        d_next(d_next < 1e-4) = d(d_next < 1e-4) * 0.5;
        % d_next = max(h_next - b, 0);


        currentVolume = sum(d_next);
        if (currentVolume > 0)
            d = d_next * (totalVolume / currentVolume);


            h = b + d;
        else
            break



    return D, H




if __name__ == '__main__':

    import numpy as np
    arr = np.array([[1,2,3], [4,5,6], [7,8,9]])

    b = np.hstack((-arr[:,0].reshape(3,1),arr[:,0:-1]))

    # print(b)