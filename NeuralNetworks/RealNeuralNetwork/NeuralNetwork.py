import numpy as np


class NeuralNetwork:
    def __init__(self, array_of_lengths):
        self.all_biases = []
        self.all_weights = []

        for i in range(len(array_of_lengths)-1):
            biases = np.random.rand(array_of_lengths[i]).tolist()
            self.all_biases.append(biases)

            weights = []
            for j in range(array_of_lengths[i+1]):
                weight = np.random.rand(array_of_lengths[i]).tolist()
                weights.append(weight)

            self.all_weights.append(weights)

        self.all_biases = np.asarray(self.all_biases)
        self.all_weights = np.asarray(self.all_weights)

    def show_all(self):
        print("All biases:")
        print(self.all_biases)
        print("All weights:")
        print(self.all_weights)
