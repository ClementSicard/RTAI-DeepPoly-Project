import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivate_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def SPU(x):
    if type(x) is np.ndarray:
        return np.array([sigmoid(-x_i) - 1 if x_i <= 0 else x_i ** 2 - 0.5 for x_i in x])
    else:
        return sigmoid(-x) - 1 if x <= 0 else x ** 2 - 0.5


def derivate_SPU(x):
    if type(x) is np.ndarray:
        return np.array([derivate_sigmoid(-x_i) if x_i <= 0 else 2 * x_i for x_i in x])
    else:
        return derivate_sigmoid(-x) if x <= 0 else 2 * x
