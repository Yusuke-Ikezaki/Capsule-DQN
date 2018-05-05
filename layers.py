import numpy as np
import tensorflow as tf

from functions import routing, squash

# convolutional layer
def conv(x, kernel, strides, padding="VALID", activation_fn=None, scope="conv"):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", kernel, dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(x, W, strides=strides, padding=padding)

        if activation_fn is not None:
            conv = activation_fn(conv)

        return conv

# fully connected layer
def fc(x, in_dim, out_dim, activation_fn=None, scope="fc"):
    with tf.variable_scope(scope):
        W = tf.get_variable("W", [in_dim, out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        fc = tf.nn.bias_add(tf.matmul(x, W), b)

        if activation_fn is not None:
            fc = activation_fn(fc)

        return fc

# flatten layer
def flatten(x, scope="flatten"):
    with tf.variable_scope(scope):
        shape = x.get_shape()[1:].as_list()
        dim = np.prod(shape)
        flatten = tf.reshape(x, [-1, dim])

        return flatten, dim

# capsule layer
def capsule(x, num_outputs, vec_len, kernel=None, strides=None, scope="capsule"):
    with tf.variable_scope(scope):
        depth = num_outputs * vec_len
        
        if kernel is not None and strides is not None:
            capsule = tf.contrib.layers.conv2d(x, depth, kernel, strides, padding="VALID", activation_fn=tf.nn.relu)
            capsule = tf.reshape(capsule, [-1, capsule.shape[1].value*capsule.shape[2].value*num_outputs, vec_len, 1])
            capsule = squash(capsule)

            return capsule
        else:
            x = tf.reshape(x, [-1, x.shape[1].value, 1, x.shape[-2].value, 1])

            W = tf.get_variable("W", [1, x.shape[1].value, depth, x.shape[-2].value, 1], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
            
            u = tf.tile(x, [1, 1, depth, 1, 1])
            
            u_hat = tf.reduce_sum(W*u, axis=3, keepdims=True)
            u_hat = tf.reshape(u_hat, [-1, x.shape[1].value, num_outputs, vec_len, 1])
            
            capsule = routing(u_hat)
            capsule = tf.squeeze(capsule, axis=1)

            return capsule
