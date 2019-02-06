from NeuralNetworks.RealNeuralNetwork.NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    start = NeuralNetwork([3, 3, 2, 1])
    start.show_all()
    print(start.get_value([1, 2, 3]))
