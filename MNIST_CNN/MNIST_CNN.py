# simple MNIST classifier implementation with CNN.
# below materials are referred for this implementation.
#
# 1. Tensorflow official tutorial
#   https://www.tensorflow.org/get_started/mnist/pros
#
# 2. 'DeepLearningZeroToAll' by prof. Sunghun-Kim.
#   https://github.com/hunkim/DeepLearningZeroToAll/
#

import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.set_random_seed(777)

# declare several functions to simplifying code implementation.
# note that some parameters are always same if use this functions
# (ex. stddev == 0.01 for all random weight)
def weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def max_pool(layer):
    return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv2d(layer, input_weight):
    return tf.nn.conv2d(layer, input_weight, strides=[1, 1, 1, 1], padding='SAME')


# initializing parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# input and output
X = tf.placeholder(tf.float32, [None, 28 * 28])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# first layer
W1 = weight([3, 3, 1, 32])
L1 = conv2d(X_img, W1)  # shape=(?, 28, 28, 32)
L1 = tf.nn.relu(L1)     # shape=(?, 28, 28, 32)
L1 = max_pool(L1)       # shape=(?, 14, 14, 32)

# second layer
W2 = weight([3, 3, 32, 64])
L2 = conv2d(L1, W2)     # shape=(?, 14, 14, 64)
L2 = tf.nn.relu(L2)     # shape=(?, 14, 14, 64)
L2 = max_pool(L2)       # shape=(?, 7, 7, 64)

L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])  # for fully connected layer

# third layer. fully connected
W3 = tf.get_variable('W3', shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# start training
for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += c / total_batch

    print('Epoch :', '%02d' % (epoch + 1), 'cost =', '{:.5f}'.format(avg_cost))


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy :', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


# check training result visually for 5 example

for i in range(5):
    r = random.randint(0, mnist.test.num_examples - 1)
    print('label :', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('prediction :', sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='gray', interpolation='nearest')
    plt.show()

