import numpy as np

from Functions.SimpleFunctions import sigmoid


class NeuralNetwork:
    def __init__(self, array_of_lengths):
        self.all_biases = []
        self.all_weights = []
        self.all_nodes = []

        for i in range(len(array_of_lengths)-1):
            biases = np.random.rand(array_of_lengths[i+1]).tolist()
            self.all_biases.append(biases)

            weights = []
            for j in range(array_of_lengths[i+1]):
                weight = np.random.rand(array_of_lengths[i]).tolist()
                weights.append(weight)

            self.all_weights.append(weights)

        for i in range(len(array_of_lengths)):
            nodes = np.random.rand(array_of_lengths[i]).tolist()
            self.all_nodes.append(nodes)

        self.all_biases = np.asarray(self.all_biases)
        self.all_weights = np.asarray(self.all_weights)
        self.all_nodes = np.asarray(self.all_nodes)

    def show_all(self):
        print("All biases:")
        print(self.all_biases)
        print("All weights:")
        print(self.all_weights)
        print("All random nodes:")
        print(self.all_nodes)

    # Need to check
    def get_value(self, data, out):
        self.all_nodes[0] = data

        for i in range(len(self.all_nodes)-1):
            self.all_nodes[i+1] = sigmoid(np.matrix(self.all_nodes[i])*np.matrix(self.all_weights[i]).T +
                                          np.matrix(self.all_biases[i]))

        return (self.all_nodes[len(self.all_nodes)-1] - out)**2

    # This is fucked up
    def learn(self, data, out):
        result = self.get_value(data, out)
