import numpy as np


def sigmoid(x, derivative=False):
    return (1 / (1 + np.exp(-x))) if derivative is False else sigmoid(x) * (1 - sigmoid(x))
