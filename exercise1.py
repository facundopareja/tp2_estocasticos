import numpy as np
import matplotlib.pyplot as plt

from constants import FILES, ORDERS
from util import ar_model, model_psd, biased_R_estimator

def point_b_parameters():
    """"Muestra parametros (coeficientes a y ganancia G) para archivos y ordenes definidos arriba."""
    for order in ORDERS:
        for file in FILES:
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

def point_b_graphs():
    """" Grafica periodograma superpuesto sobre PSD del modelo en base a los parametros estimados."""
    for file in FILES:
        signal = np.loadtxt(file, dtype=float)
        x = np.linspace(0, np.pi, 100)
        for order in ORDERS:
            plt.plot(x, periodgram(x, signal))
            a, G = ar_model(signal, order)
            plt.plot(x, model_psd(x, a, G, order))
            plt.ylabel('Periodograma / PSD')
            plt.title(f"Archivo: {file} - Comparacion entre periodograma y PSD para orden {order}", fontsize=8)
            plt.show()


#point_b_parameters()
point_b_graphs()
