from Functions.SimpleFunctionsTF import *
from NeuralNetworksTensorFlow.TensorFlowClass.TFObject import TFObject
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    data_todo = input_data.read_data_sets("../../Data/MNIST_data", one_hot=True)
    data = [[data_todo.train.images, data_todo.train.labels], [data_todo.test.images, data_todo.test.labels]]

    neural_networkTF = TFObject([784, 200, 100, 60, 30, 10], data, relu, cross_entropy)
    neural_networkTF.run()
