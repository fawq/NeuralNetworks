import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class TFObject:
    def __init__(self, array_of_lengths, all_data, function, loss_function, soft_max=True, batch=100, display_step=100,
                 number_of_iterations=5000, my_stddev=0.1):
        self.all_data = all_data
        self.array_of_lengths = array_of_lengths
        self.function = function
        self.loss_function = loss_function
        self.soft_max = soft_max
        self.batch = batch
        self.display_step = display_step
        self.number_of_iterations = number_of_iterations
        self.my_stddev = my_stddev
        self.batch_X = 0
        self.batch_Y = 0

        tf.set_random_seed(0)

        self.X = tf.placeholder(tf.float32, [None, self.array_of_lengths[0]])
        self.XX = tf.reshape(self.X, [-1, self.array_of_lengths[0]])
        self.Y = tf.placeholder(tf.float32, [None, self.array_of_lengths[len(self.array_of_lengths)-1]])
        self.W = []
        self.b = []
        self.A = []

        for i in range(len(self.array_of_lengths)-1):
            self.W.append(tf.Variable(tf.truncated_normal([self.array_of_lengths[i], self.array_of_lengths[i+1]],
                                                          stddev=self.my_stddev)))
            self.b.append(tf.Variable(tf.zeros([self.array_of_lengths[i+1]])))
            if i == 0:
                self.A.append(self.function(tf.matmul(self.XX, self.W[i]) + self.b[i]))
            elif i < len(self.array_of_lengths)-2:
                self.A.append(self.function(tf.matmul(self.A[i-1], self.W[i]) + self.b[i]))

        self.A.append(tf.matmul(self.A[len(self.A)-1], self.W[len(self.W)-1]) +
                      self.b[len(self.b)-1])

        if self.soft_max:
            self.A[len(self.A)-1] = tf.nn.softmax(self.A[len(self.A)-1])

        self.loss_function_arguments = self.loss_function(self.Y, self.A[len(self.A)-1], self.batch)

        self.correct_prediction = tf.equal(tf.argmax(self.A[len(self.A)-1], 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.train_step = tf.train.GradientDescentOptimizer(0.005).minimize(self.loss_function_arguments)

        self.init = tf.global_variables_initializer()

        self.test_losses = []
        self.test_acc = []

        self.saver = tf.train.Saver()

    def run(self):
        with tf.Session() as sess:
            sess.run(self.init)

            for i in range(self.number_of_iterations):
                self.batch_X, self.batch_Y = self.next_batch(self.batch, self.all_data[0][0], self.all_data[0][1])

                if i % self.display_step == 0:
                    acc_tst, loss_tst = sess.run([self.accuracy, self.loss_function_arguments],
                                                 feed_dict={self.X: self.all_data[1][0],
                                                            self.Y: self.all_data[1][1]})

                    print("#{} Tst acc={} , Tst loss={}".format(i+self.display_step, acc_tst, loss_tst))

                    self.test_losses.append(loss_tst)
                    self.test_acc.append(acc_tst)

                sess.run(self.train_step, feed_dict={self.X: self.batch_X, self.Y: self.batch_Y})

        plt.plot(self.test_losses)
        plt.title('Test losses')
        plt.show()
        plt.title('Test accept')
        plt.plot(self.test_acc)
        plt.show()

    def next_batch(self, num, data, labels):
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
