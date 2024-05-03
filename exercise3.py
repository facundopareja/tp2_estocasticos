import numpy as np
from matplotlib import pyplot as plt
from constants import FILES
from util import ar_model, window_potency, welch_estimator, model_psd


def white_noise_generator(a, G, z, order):
    total_sum = 0
    for i in range(order):
        total_sum += (a[i] * (1 / z))
    return G/(1 - total_sum)


def synthesize_eeg_signal(a, G, order):
    mean = 0
    std = 1
    num_samples = 2000
    synthesized_process = []
    samples = np.random.normal(mean, std, size=num_samples)
    for z in samples:
        synthesized_process.append(white_noise_generator(a, G, z, order))
    return np.array(synthesized_process)


def exercise_3():
    order = 13
    M = 80
    for file in FILES:
        signal = np.loadtxt(file, dtype=float)
        a, G = ar_model(signal, order)
        synthesized_signal = synthesize_eeg_signal(a, G, order)
        P = window_potency(M)
        x = np.linspace(0, np.pi, M)
        # Esto esta mal, me falta corregirlo
        plt.plot(x, welch_estimator(synthesized_signal, P))
        plt.plot(x, model_psd(x, a, G, order))
        plt.show()


exercise_3()
