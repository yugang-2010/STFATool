import numpy as np


def MSST_Y(x, hlength, num):

    x = np.asarray(x).flatten()
    N = x.size
    if N == 0:
        raise ValueError("Input x must have non-zero length")


    hlength += 1 - (hlength % 2)
    ht = np.linspace(-0.5, 0.5, hlength)
    h = np.exp(-np.pi / (0.32 ** 2) * ht ** 2)
    Lh = (hlength - 1) // 2

    half = int(round(N / 2))
    tfr_mat = np.zeros((N, N), dtype=complex)
    for icol in range(N):
        ti = icol
        tau_min = -min(half - 1, Lh, ti)
        tau_max = min(half - 1, Lh, N - 1 - ti)
        tau = np.arange(tau_min, tau_max + 1)
        idx = (tau + N) % N
        rSig = x[ti + tau]
        win = h[Lh + tau]
        tfr_mat[idx, icol] = rSig * np.conjugate(win)

    tfr = np.fft.fft(tfr_mat, axis=0)[:half, :]

    unwrapped = np.unwrap(np.angle(tfr), axis=1)
    raw_omega = np.diff(unwrapped, axis=1) * (N) / (2 * np.pi)
    omega2 = np.zeros((half, N), dtype=int)
    omega2[:, :-1] = np.round(raw_omega).astype(int)
    omega2[:, -1] = omega2[:, -2]

    omega = omega2.copy()
    if num > 1:
        for _ in range(num - 1):
            new_omega = np.zeros_like(omega)
            for eta in range(half):
                for b in range(N):
                    k = omega[eta, b]
                    if 1 <= k <= half:
                        new_omega[eta, b] = omega[k - 1, b]
            omega = new_omega
        omega2 = omega

    Ts = np.zeros((half, N), dtype=complex)
    for b in range(N):
        for eta in range(half):
            val = tfr[eta, b]
            if abs(val) > 1e-4:
                k = omega2[eta, b]
                if 1 <= k <= half:
                    Ts[k - 1, b] += val


    Ts = Ts / (N / 2)

    return Ts
