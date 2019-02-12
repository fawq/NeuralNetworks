import tensorflow as tf

if __name__ == "__main__":
    x = tf.constant([2])
    y = tf.constant([4])

    multiply = tf.multiply(x, y)

    sess = tf.Session()
    result_1 = sess.run(multiply)
    print(result_1)
    sess.close()

    x = tf.get_variable("x", dtype=tf.int32, initializer=tf.constant([5]))
    z = tf.get_variable("z", dtype=tf.int32, initializer=tf.constant([6]))
    c = tf.constant([5], name="constant")
    square = tf.constant([2], name="square")
    f = tf.multiply(x, z) + tf.pow(x, square) + z + c

    init = tf.global_variables_initializer()  # prepare to initialize all variables
    with tf.Session() as sess:
        init.run()  # Initialize x and y
        function_result = f.eval()
    print(function_result)
