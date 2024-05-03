import numpy as np
from matplotlib import pyplot as plt
from constants import FILES
from util import ar_model, window_potency, welch_estimator


def exercise_2():
    order = 13
    for file in FILES:
        signal = np.loadtxt(file, dtype=float)
        signal = np.append(signal, np.zeros(40))
        a, G = ar_model(signal, order)
        M = 80
        P = window_potency(M)
        x = np.linspace(0, np.pi, 80)
        plt.plot(x, welch_estimator(signal, P))
        #plt.plot(x, model_psd(x, a, G, order))
        plt.show()


exercise_2()
