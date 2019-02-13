import tensorflow as tf


def cross_entropy(y, y_label, batch):
    return -tf.reduce_mean(y * tf.log(y_label)) * batch


def relu(result):
    return tf.nn.relu(result)


def sigmoid(result):
    return tf.nn.sigmoid(result)


def elu(result):
    return tf.nn.elu(result)


def tanh(result):
    return tf.nn.tanh(result)
