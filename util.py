import numpy as np
from scipy.fft import fft

N = 2000 + 40
M = 80
WINDOW_FACTOR = 0.5
K = int(WINDOW_FACTOR * M)
L = int(N / M)


def ith_segment_periodogram(segment, P):
    """"Periodogram de i-esimo segmento."""
    return np.power(np.abs(fft(segment)), 2) / (M * P)


def welch_estimator(signal, P):
    total_sum = 0
    for i in range(1, L + 1):
        start = (i - 1) * K
        end = start + M
        x_m = hamming_window(i, M) * signal[start:end]
        total_sum += ith_segment_periodogram(x_m, P)
    return (1 / L) * total_sum


def hamming_window(n, M):
    """"Ventaneado de Hamming."""
    a_0 = 0.54
    return a_0 - (1 - a_0) * np.cos((2 * np.pi * n) * (1 / M))


def window_potency(M):
    """"Window potency."""
    total_sum = 0
    for t in range(M):
        total_sum += np.power(np.abs(hamming_window(t, M)), 2)
    return (1 / M) * total_sum


def model_psd(w, a_coefficients, gain_factor, P):
    """" Funcion de PSD del modelo en base a los parametros estimados: coeficiente a y factor de ganancia."""
    total_sum = 0
    for k in range(1, P + 1):
        total_sum += np.multiply(a_coefficients[k - 1], np.exp(1j * -w * k))
    denominator = np.power(abs(1 - total_sum), 2)
    return np.power(gain_factor, 2) / denominator


def calculate_gain_factor_estimation(x, P, a_coefficients):
    """"Estima el factor de ganancia G a partir de la autocorrelacion
    de la señal y los coeficientes a."""
    total_sum = 0
    for i in range(1, P + 1):
        total_sum += a_coefficients[i - 1] * biased_R_estimator(x, i)
    return np.sqrt(biased_R_estimator(x, 0) - total_sum)


def biased_R_estimator(x, k):
    """"Estima la autocorrelacion de la señal x para un valor k de forma sesgada."""
    N = len(x)
    sum_term = 0
    for n in range(N - k):
        sum_term += np.multiply(x[n], x[n + k])
    return (1 / N) * sum_term


def get_R_matrix_row(i, x, P, limit):
    """"Calcula una fila de la matriz R para los parametros dados."""
    row = np.zeros(P)
    for index, value in enumerate(range(i, limit)):
        row[index] = biased_R_estimator(x, abs(value))
    return row


def calculate_R_matrix(x, P):
    """"Calcula la matriz R donde cada valor es una estimacion de la autocorrelacion de la señal."""
    R_matrix = np.zeros(shape=(P, P))
    for i in range(0, -P, -1):
        R_matrix[abs(i)] = get_R_matrix_row(i, x, P, P - abs(i) * 1)
    return R_matrix


def ar_model(x, P):
    """" Devuelve coeficientes a y ganancia G, recibiendo como parametros el orden de la funcion P y una señal x."""
    R_matrix = calculate_R_matrix(x, P)
    r = get_R_matrix_row(1, x, P, P + 1)
    a_coefficients_estimation = np.matmul(np.linalg.inv(R_matrix), r)
    gain_factor_estimation = calculate_gain_factor_estimation(x, P, a_coefficients_estimation)
    return a_coefficients_estimation, gain_factor_estimation
