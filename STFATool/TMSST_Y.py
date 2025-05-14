import numpy as np


def TMSST_Y(x, hlength, num):
    def _rhalf_up(x):
        return np.sign(x) * np.floor(np.abs(x) + 0.5)

    x = np.asarray(x, dtype=float).flatten()
    N = x.size
    if N == 0:
        raise ValueError("Input x must be non-empty 1-D array")

    t = np.arange(1, N + 1)

    hlength += 1 - (hlength % 2)
    ht = np.linspace(-0.5, 0.5, hlength)
    h  = np.exp(-np.pi / 0.32**2 * ht**2)
    th = h * ht
    Lh = (hlength - 1) // 2

    half = (N + 1) // 2

    tfr1 = np.zeros((N, N), dtype=complex)
    tfr3 = np.zeros_like(tfr1)

    for icol in range(N):
        ti = icol + 1
        tau_min = -min(half - 1, Lh, ti - 1)
        tau_max =  min(half - 1, Lh, N - ti)
        tau      = np.arange(tau_min, tau_max + 1)

        idx   = (tau + N) % N
        rSig  = x[ti - 1 + tau]
        win   = h [Lh + tau]
        win_t = th[Lh + tau]

        tfr1[idx, icol] = rSig * np.conjugate(win)
        tfr3[idx, icol] = rSig * np.conjugate(win_t)

    tfr1 = np.fft.fft(tfr1, axis=0)[:half, :]
    tfr3 = np.fft.fft(tfr3, axis=0)[:half, :]

    neta, nb = tfr1.shape

    omega = (t - 1) + (hlength - 1) * np.real(tfr3 / tfr1)

    if num > 1:
        omega_tmp = omega.copy()
        for _ in range(num - 1):
            omega_new = np.zeros_like(omega_tmp)
            for eta in range(neta):
                for b in range(nb):
                    k2 = int(_rhalf_up(omega_tmp[eta, b]))
                    if 1 <= k2 <= nb:
                        omega_new[eta, b] = omega_tmp[eta, k2 - 1]
            omega_tmp = omega_new
        omega = omega_tmp

    omega = _rhalf_up(_rhalf_up(omega * 2) / 2)

    tfr2 = np.empty_like(tfr1)
    for eta in range(neta):
        tfr2[eta, :] = tfr1[eta, :] * np.exp(
            -1j * 2 * np.pi * (eta + 1) * (t / N)
        )

    Ts = np.zeros_like(tfr2)
    for b in range(nb):
        for eta in range(neta):
            k2 = int(omega[eta, b])
            if 1 <= k2 <= nb:
                Ts[eta, k2 - 1] += tfr2[eta, b]

    Ts   = Ts   / (N / 2)


    return Ts
