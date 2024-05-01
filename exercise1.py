import numpy as np
import matplotlib.pyplot as plt

orders = [2, 13, 30]
files = ['eeg_ojos_abiertos_t7.csv', 'eeg_ojos_cerrados_t7.csv']


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


def calculate_gain_factor_estimation(x, P, a_coefficients):
    """"Estima el factor de ganancia G a partir de la autocorrelacion
    de la señal y los coeficientes a."""
    total_sum = 0
    for i in range(1, P + 1):
        total_sum += a_coefficients[i - 1] * biased_R_estimator(x, i)
    return np.sqrt(biased_R_estimator(x, 0) - total_sum)


def ar_model(x, P):
    """" Devuelve coeficientes a y ganancia G, recibiendo como parametros el orden de la funcion P y una señal x."""
    R_matrix = calculate_R_matrix(x, P)
    r = get_R_matrix_row(1, x, P, P + 1)
    a_coefficients_estimation = np.matmul(np.linalg.inv(R_matrix), r)
    gain_factor_estimation = calculate_gain_factor_estimation(x, P, a_coefficients_estimation)
    return a_coefficients_estimation, gain_factor_estimation


def point_b_parameters():
    """"Muestra parametros (coeficientes a y ganancia G) para archivos y ordenes definidos arriba."""
    for order in orders:
        for file in files:
            loaded_file = np.loadtxt(file, dtype=float)
            a, G = ar_model(loaded_file, order)
            print(f"Los coeficientes para el archivo {file} orden {order} son {a} con factor estimado de ganancia {G}")


def periodgram(w, signal):
    """" Funcion periodograma para un w respecto de un estimador ya definido basado en la señal pasada como
    parametro."""
    N = len(signal)
    total_sum = 0
    for k in range(-N + 1, N):
        total_sum += np.multiply(biased_R_estimator(signal, abs(k)), np.exp(1j * -w * k))
    return total_sum


def model_psd(w, a_coefficients, gain_factor, P):
    """" Funcion de PSD del modelo en base a los parametros estimados: coeficiente a y factor de ganancia."""
    total_sum = 0
    for k in range(1, P + 1):
        total_sum += np.multiply(a_coefficients[k - 1], np.exp(1j * -w * k))
    denominator = np.power(abs(1 - total_sum), 2)
    return np.power(gain_factor, 2) / denominator


def point_b_graphs():
    """" Grafica periodograma superpuesto sobre PSD del modelo en base a los parametros estimados."""
    for file in files:
        signal = np.loadtxt(file, dtype=float)
        x = np.linspace(0, 2 * np.pi, 100)
        for order in orders:
            plt.plot(x, periodgram(x, signal))
            a, G = ar_model(signal, order)
            plt.plot(x, model_psd(x, a, G, order))
            plt.ylabel('Periodograma / PSD')
            plt.title(f"Archivo: {file} - Comparacion entre periodograma y PSD para orden {order}", fontsize=8)
            plt.show()

#point_b_parameters()
point_b_graphs()

