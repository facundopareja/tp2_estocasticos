import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft

from exercise1 import ar_model, model_psd

N = 2000 + 40
M = 80
WINDOW_FACTOR = 0.5
K = int(WINDOW_FACTOR * M)
L = int(N / M)


def hamming_window(n):
    """"Ventaneado de Hamming."""
    a_0 = 0.54
    return a_0 - (1 - a_0) * np.cos((2 * np.pi * n) * (1 / M))


def ith_segment_periodogram(segment, P):
    """"Periodogram de i-esimo segmento."""
    return np.power(np.abs(fft(segment)), 2) / (M * P)


def window_potency():
    """"Window potency."""
    total_sum = 0
    for t in range(M):
        total_sum += np.power(np.abs(hamming_window(t)), 2)
    return (1 / M) * total_sum


def welch_estimator(signal, P):
    total_sum = 0
    for i in range(1, L+1):
        start = (i - 1) * K
        end = start+M
        x_m = hamming_window(i) * signal[start:end]
        total_sum += ith_segment_periodogram(x_m, P)
    return (1 / L) * total_sum


def exercise_2():
    order = 13
    file = 'eeg_ojos_abiertos_t7.csv'
    signal = np.loadtxt(file, dtype=float)
    signal = np.append(signal, np.zeros(40))
    a, G = ar_model(signal, order)
    P = window_potency()
    x = np.linspace(0, 2 * np.pi, 80)
    plt.plot(x, welch_estimator(signal, P))
    plt.plot(x, model_psd(x, a, G, order))
    plt.show()


exercise_2()
