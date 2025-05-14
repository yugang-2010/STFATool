import numpy as np

def SET_Y(x, hlength=None):
    def _rhalf_up(x):
        return np.sign(x) * np.floor(np.abs(x) + 0.5)

    x = np.asarray(x, dtype=float).flatten()
    N = x.size
    if N == 0:
        raise ValueError("输入信号不能为空")

    if hlength is None:
        hlength = int(_rhalf_up(N / 8))
    hlength += 1 - (hlength % 2)

    half = (N + 1) // 2
    ft = np.arange(1, half + 1)

    ht = np.linspace(-0.5, 0.5, hlength)
    h = np.exp(-np.pi / 0.32 ** 2 * ht ** 2)
    dh = -2 * np.pi / 0.32 ** 2 * ht * h
    Lh = (hlength - 1) // 2

    tfr1 = np.zeros((N, N), dtype=complex)
    tfr2 = np.zeros_like(tfr1)

    for icol in range(N):
        ti = icol
        tau_min = -min(half - 1, Lh, ti)
        tau_max = min(half - 1, Lh, N - ti - 1)
        tau = np.arange(tau_min, tau_max + 1)

        idx = (tau + N) % N
        rSig = x[ti + tau]
        win = h[Lh + tau]
        win_d = dh[Lh + tau]

        tfr1[idx, icol] = rSig * np.conjugate(win)
        tfr2[idx, icol] = rSig * np.conjugate(win_d)

    tfr1 = np.fft.fft(tfr1, axis=0)[:half, :]
    tfr2 = np.fft.fft(tfr2, axis=0)[:half, :]

    va = N / hlength
    omega = (ft[:, None] - 1) + np.real(
        va * 1j * tfr2 / (2 * np.pi) / tfr1
    )

    E = np.mean(np.abs(x))
    IF = np.zeros_like(tfr1, dtype=float)
    for i in range(half):
        for j in range(N):
            if (np.abs(tfr1[i, j]) > 0.8 * E and
                    np.abs(omega[i, j] - (i + 1)) < 0.5):
                IF[i, j] = 1.0

    tfr = tfr1 / (h.sum() / 2)
    Te = tfr * IF

    return IF, Te
