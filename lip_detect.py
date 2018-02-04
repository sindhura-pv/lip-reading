import tensorflow as tf
import numpy as np
import h5py

image_width = 640
image_height = 480

x = tf.placeholder(tf.float32, [None, image_width, image_height])
y = tf.placeholder(tf.float32, [None, 20])

def get_data():
    filename= 'X_train.hdf'
    f = h5py.File(filename, 'r')
    a_group_key = list(f.keys())[0]
    a = list(f[a_group_key])
    x= np.array(a)
    print(x.shape)

    filename= 'Y_train.hdf'
    f = h5py.File(filename, 'r')
    a_group_key = list(f.keys())[0]
    a = list(f[a_group_key])
    y= np.array(a)
    print(y.shape)

    return x,y


weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 24])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 24, 36])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 36, 48])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 48, 64])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    'wd1': tf.Variable(tf.random_normal([160 * 120 * 64, 500])),
    'wd2': tf.Variable(tf.random_normal([500, 90])),
    'wd3': tf.Variable(tf.random_normal([90, 20])),

}
biases = {
    'bc1': tf.Variable(tf.random_normal([24])),
    'bc2': tf.Variable(tf.random_normal([36])),
    'bc3': tf.Variable(tf.random_normal([48])),
    'bc4': tf.Variable(tf.random_normal([64])),
    'bc5': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([500])),
    'bd2': tf.Variable(tf.random_normal([90])),
    'bd3': tf.Variable(tf.random_normal([20]))
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv_net(x, weights, biases):
    x = tf.reshape(x, shape=[-1, image_width, image_height, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # conv3 = maxpool2d(conv3, k=2)

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)

    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv5 = maxpool2d(conv5, k=2)

    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)

    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)

    return fc3

conv_output = conv_net(x, weights, biases)
cross_entropy = tf.squared_difference(y, conv_output)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)

epochs = 31
iters = 70

X_train, Y_train = get_data()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_num in range(1, epochs):
        print(epoch_num)
        for iter in range(iters):
            sess.run(optimizer, feed_dict={x: X_train[iter:iter + 10], y: Y_train[iter:iter + 10]})
            if iter % 10 == 0:
                print(sess.run(cost, feed_dict={x: X_train[iter:iter + 10], y: Y_train[iter:iter + 10]}))

    saver = tf.train.Saver()
    saver.save(sess, 'my-model')

