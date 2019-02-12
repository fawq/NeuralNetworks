from NeuralNetworks.RealNeuralNetwork.NeuralNetwork import NeuralNetwork
import tensorflow as tf

if __name__ == "__main__":
    #start = NeuralNetwork([3, 3, 2, 1])
    #start.show_all()

    r1 = tf.constant(1, tf.int16)
    print(r1)

    r2 = tf.constant(1, tf.int16, name="my_scalar")
    print(r2)

    # Decimal
    r1_decimal = tf.constant(1.12345, tf.float32)
    print(r1_decimal)
    # String
    r1_string = tf.constant("Guru99", tf.string)
    print(r1_string)

    r1_vector = tf.constant([1, 3, 5], tf.int16)
    print(r1_vector)
    r2_boolean = tf.constant([True, True, False], tf.bool)
    print(r2_boolean)

    r2_matrix = tf.constant([[1, 2],
                             [3, 4]], tf.int16)
    print(r2_matrix)

    r3_matrix = tf.constant([[[1, 2],
                              [3, 4],
                              [5, 6]]], tf.int16)
    print(r3_matrix)

    x = tf.constant([2.0], dtype=tf.float32)
    print(tf.sqrt(x))

    # Add
    tensor_a = tf.constant([[1, 2]], dtype=tf.int32)
    tensor_b = tf.constant([[3, 4]], dtype=tf.int32)

    tensor_add = tf.add(tensor_a, tensor_b)
    print(tensor_add)

    var = tf.get_variable("var", [1, 2])
    print(var.shape)

    var_init_1 = tf.get_variable("var_init_1", [1, 2], dtype=tf.int32, initializer=tf.zeros_initializer)
    print(var_init_1.shape)

    # Create a 2x2 matrix
    tensor_const = tf.constant([[10, 20], [30, 40]])
    # Initialize the first value of the tensor equals to tensor_const
    var_init_2 = tf.get_variable("var_init_2", dtype=tf.int32, initializer=tensor_const)
    print(var_init_2.shape)

    data_placeholder_a = tf.placeholder(tf.float32, name="data_placeholder_a")
    print(data_placeholder_a)
