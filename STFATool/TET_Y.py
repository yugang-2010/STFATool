import numpy as np

def TET_Y(x, hlength=None):

    x = np.asarray(x).flatten()
    N = x.size
    if N == 0:
        raise ValueError("Input signal x must have non-zero length")

    if hlength is None:
        hlength = int(round(N / 8))
    hlength += 1 - (hlength % 2)

    half = int(round(N / 2))
    t = np.arange(1, N+1)
    ft = np.arange(1, half+1)

    ht = np.linspace(-0.5, 0.5, hlength)
    h = np.exp(-np.pi / 0.32**2 * ht**2)

    th = h * ht
    Lh = (hlength - 1) // 2

    tfr1 = np.zeros((N, N), dtype=complex)
    tfr2 = np.zeros((N, N), dtype=complex)

    for icol in range(N):
        ti = icol
        tau_min = -min(half-1, Lh, ti)
        tau_max =  min(half-1, Lh, N - ti - 1)
        tau = np.arange(tau_min, tau_max + 1)
        idx = (tau + N) % N

        rSig = x[ti + tau]
        win  = h[Lh + tau]
        win_t = th[Lh + tau]

        tfr1[idx, icol] = rSig * np.conjugate(win)
        tfr2[idx, icol] = rSig * np.conjugate(win_t)

    tfr1 = np.fft.fft(tfr1, axis=0)[:half, :]
    tfr2 = np.fft.fft(tfr2, axis=0)[:half, :]

    E = np.mean(np.abs(x))

    GD = np.zeros((half, N))
    for a in range(half):

        GD[a, :] = t + (hlength - 1) * np.real(tfr2[a, :] / tfr1[a, :])

    TEO = np.zeros((half, N))
    for i in range(half):
        for j in range(N):
            if np.abs(tfr1[i, j]) > 12 * E and np.abs(GD[i, j] - (j+1)) < 0.5:
                TEO[i, j] = 1

    tfr = tfr1 / (N / 2)

    Te_t = TEO * tfr

    return Te_t, GD
