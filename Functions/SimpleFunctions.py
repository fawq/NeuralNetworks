import numpy as np


def sigmoid(x, derivative=False):
    return (1 / (1 + np.exp(-x))) if derivative is False else sigmoid(x) * (1 - sigmoid(x))


def relu(x, derivative=False):
    return max(0, x) if derivative is False else 1 if x > 0 else 0
