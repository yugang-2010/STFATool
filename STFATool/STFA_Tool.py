import os
import warnings
import sys
import numpy as np
from PyQt5.QtCore import Qt
from scipy.io import wavfile, loadmat
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QComboBox, QLineEdit, QLabel, QGridLayout, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QFont, QPixmap

os.environ["QT_OPENGL"] = "software"
warnings.filterwarnings("ignore", category=DeprecationWarning, message="sipPyTypeDict")

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 10
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
import numpy as np

def SET_Y(x, hlength=None):
    def _rhalf_up(x):
        return np.sign(x) * np.floor(np.abs(x) + 0.5)

    x = np.asarray(x, dtype=float).flatten()
    N = x.size
    if N == 0:
        raise ValueError("The input signal cannot be empty")

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














class TFAViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sparse Time-Frequency Analysis Tool")
        self.resize(1000, 1000)


        self.samplerate = None
        self.data = None

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        top = QGridLayout()
        top.setContentsMargins(5,5,5,5)
        top.setHorizontalSpacing(10)
        top.setVerticalSpacing(5)

        self.load_btn = QPushButton("Load Data")
        top.addWidget(self.load_btn, 0, 0, 2, 1)
        self.load_btn.clicked.connect(self.load_wav)

        top.addWidget(QLabel("TFA Method"), 0, 1, 1, 1)
        self.combo = QComboBox()
        self.combo.addItems(["SET", "TET", "MSST", "TMSST"])
        self.combo.currentTextChanged.connect(self.on_method_change)
        top.addWidget(self.combo, 1, 1)

        self.hlen_edit = QLineEdit()
        self.hlen_edit.setPlaceholderText("Window Length")
        top.addWidget(self.hlen_edit, 0, 2)
        self.iter_edit = QLineEdit()
        self.iter_edit.setPlaceholderText("Iterations")
        top.addWidget(self.iter_edit, 1, 2)
        self.proc_btn = QPushButton("Process")
        self.proc_btn.clicked.connect(self.process_tfa)
        top.addWidget(self.proc_btn, 0, 3, 2, 1)

        self.start_edit = QLineEdit();
        self.start_edit.setPlaceholderText("Start Time (s)")
        self.end_edit = QLineEdit();
        self.end_edit.setPlaceholderText("End Time (s)")
        self.lowf_edit = QLineEdit();
        self.lowf_edit.setPlaceholderText("Start Freq (Hz)")
        self.highf_edit = QLineEdit();
        self.highf_edit.setPlaceholderText("End Freq (Hz)")
        self.zoom_btn = QPushButton("Zoom Region")
        self.zoom_btn.clicked.connect(self.zoom_region)
        self.save_btn = QPushButton("Save Heatmap")
        self.save_btn.clicked.connect(self.save_heatmap)

        top.addWidget(self.start_edit, 0, 4)
        top.addWidget(self.end_edit, 1, 4)
        top.addWidget(self.lowf_edit, 0, 5)
        top.addWidget(self.highf_edit, 1, 5)

        top.addWidget(self.zoom_btn, 0, 6)


        top.addWidget(self.save_btn, 1, 6)


        self.logo_label = QLabel()
        pixmap = QPixmap("logo2.png")

        self.logo_label.setPixmap(pixmap.scaledToHeight(96, Qt.SmoothTransformation))
        self.logo_label.setAlignment(Qt.AlignCenter)
        top.addWidget(self.logo_label, 0, 7, 2, 1)

        layout.addLayout(top)

        mid = QHBoxLayout()
        self.fig_time = Figure()
        self.canvas_time = FigureCanvas(self.fig_time)
        mid.addWidget(self.canvas_time)
        self.fig_freq = Figure()
        self.canvas_freq = FigureCanvas(self.fig_freq)
        mid.addWidget(self.canvas_freq)
        layout.addLayout(mid, 1)

        self.fig_tfa = Figure(figsize=(5,4))
        self.canvas_tfa = FigureCanvas(self.fig_tfa)
        layout.addWidget(self.canvas_tfa, 3)

        self.on_method_change(self.combo.currentText())

    def on_method_change(self, method):
        enabled = method not in ("SET", "TET")
        self.iter_edit.setEnabled(enabled)
        if not enabled:
            self.iter_edit.clear()

    def load_wav(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WAV or MAT File",
            "",
            "WAV & MAT Files (*.wav *.mat);;Wave Files (*.wav);;MAT Files (*.mat)"
        )

        if not path:
            return

        if path.lower().endswith(".wav"):
            sr, data = wavfile.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)

        elif path.lower().endswith(".mat"):
            try:
                mat_data = loadmat(path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot read MAT file:\n{e}")
                return

            keys = [k for k in mat_data.keys() if not k.startswith("__")]
            if not keys:
                QMessageBox.critical(self, "Error", "MAT file contains no valid variables.")
                return
            arr = mat_data[keys[0]]
            arr = np.asarray(arr)

            if arr.ndim > 1:
                data = arr[:, 0].flatten()
            else:
                data = arr.flatten()


            sr, ok = QInputDialog.getInt(
                self,
                "Input Sampling Rate",
                "Enter sampling rate (Hz):",
                value=12000, min=1, max=1000000, step=1
            )
            if not ok:
                QMessageBox.warning(self, "Canceled", "Sampling rate input canceled.")
                return
        else:
            QMessageBox.warning(self, "Unsupported", "Please select a .wav or .mat file.")
            return
        self.samplerate, self.data = sr, data

        self._update_time_freq_plots()

    def _update_time_freq_plots(self):

        data = self.data
        fs = self.samplerate


        t = np.arange(len(data)) / fs
        self.fig_time.clear()
        ax_time = self.fig_time.add_subplot(111)

        self.fig_time.subplots_adjust(left=0.10, right=0.95, bottom=0.1, top=0.90)

        ax_time.plot(t, data)
        ax_time.set_title("Time Domain")
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude")
        self.canvas_time.draw()


        fftv = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), d=1/fs)
        self.fig_freq.clear()
        ax_freq = self.fig_freq.add_subplot(111)
        self.fig_freq.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.90)
        ax_freq.plot(freqs, fftv)
        ax_freq.set_title("Frequency Domain")
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Amplitude")
        ax_freq.set_xlim(0, fs/2)
        self.canvas_freq.draw()

    def process_tfa(self):
        if self.data is None:
            return


        self._update_time_freq_plots()

        method = self.combo.currentText()
        x = self.data.flatten()
        fs = self.samplerate
        hlen_str = self.hlen_edit.text().strip()
        if not hlen_str:
            QMessageBox.warning(self, "Input Error", "Please enter window length.")
            return
        try:
            hlen = int(hlen_str)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Window length must be an integer.")
            return
        if not (1 <= hlen <= 2000):
            QMessageBox.warning(self, "Input Error", "Window length must be between 1 and 2000.")
            return


        method = self.combo.currentText()
        if method in ("MSST", "TMSST"):
            iters_str = self.iter_edit.text().strip()
            if not iters_str:
                QMessageBox.warning(self, "Input Error", "Please enter number of iterations.")
                return
            try:
                iters = int(iters_str)
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Iterations must be an integer.")
                return
            if not (1 <= iters <= 30):
                QMessageBox.warning(self, "Input Error", "Iterations must be between 1 and 30.")
                return
        else:
            iters = None


        if method == "SET":
            _, Te = SET_Y(x, hlen)
        elif method == "TET":
            Te, _ = TET_Y(x, hlen)
        elif method == "MSST":
            Te = MSST_Y(x, hlen, iters)
        else:  # TMSST
            Te = TMSST_Y(x, hlen, iters)


        N = x.size
        time_axis = np.arange(N) / fs
        freq_axis = np.linspace(0, fs/2, Te.shape[0])

        self.fig_tfa.clear()

        ax = self.fig_tfa.add_subplot(111)
        self.fig_tfa.subplots_adjust(left=0.07, right=0.93, bottom=0.07, top=0.93)
        im = ax.imshow(
            np.abs(Te),
            origin='lower',
            aspect='auto',
            extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
            cmap='jet'
        )
        ax.set_title(f"{method} Heatmap")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        self.canvas_tfa.draw()

    def zoom_region(self):
        if self.data is None:
            return


        try:
            t0, t1 = float(self.start_edit.text()), float(self.end_edit.text())
        except:
            t0 = t1 = None
        try:
            f0, f1 = float(self.lowf_edit.text()), float(self.highf_edit.text())
        except:
            f0 = f1 = None


        if t0 is not None and t1 is not None:
            ax_time = self.fig_time.axes[0]
            ax_time.set_xlim(t0, t1)
            self.canvas_time.draw()

        if f0 is not None and f1 is not None:
            ax_freq = self.fig_freq.axes[0]
            ax_freq.set_xlim(f0, f1)
            self.canvas_freq.draw()


        if self.fig_tfa.axes:
            ax_tfa = self.fig_tfa.axes[0]
            if t0 is not None and t1 is not None:
                ax_tfa.set_xlim(t0, t1)
            if f0 is not None and f1 is not None:
                ax_tfa.set_ylim(f0, f1)
            self.canvas_tfa.draw()
    def save_heatmap(self):

        if not self.fig_tfa.axes:
            QMessageBox.warning(self, "Warning", "No heatmap to save. Please process first.")
            return


        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Heatmap as PNG",
            "",
            "PNG Files (*.png)"
        )
        if not path:
            return


        if not path.lower().endswith(".png"):
            path += ".png"

        try:
            # 保存当前 heatmap Figure
            self.fig_tfa.savefig(
                path,
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.1
            )
            QMessageBox.information(self, "Success", f"Heatmap saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save heatmap:\n{e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)

    app.setFont(QFont("Arial", 12))
    win = TFAViewer()
    win.show()
    sys.exit(app.exec_())
