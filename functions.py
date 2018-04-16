import numpy as np
import tensorflow as tf

from config import cfg

# routing algorithm
def routing(u_hat):
    b_IJ = tf.constant(np.zeros([u_hat.shape[0].value, u_hat.shape[1].value, u_hat.shape[2].value, 1, 1], dtype=np.float32))

    u_hat_stopped = tf.stop_gradient(u_hat, name="stop_gradient")

    for routing_iter in range(cfg.routing_iters):
        with tf.variable_scope("iter_"+str(routing_iter)):
            c_IJ = tf.nn.softmax(b_IJ, axis=2)

            if routing_iter == cfg.routing_iters-1:
                s_J = tf.multiply(c_IJ, u_hat)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                v_J = squash(s_J)
            else:
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True)
                v_J = squash(s_J)
                v_J_tiled = tf.tile(v_J, [1, u_hat.shape[1].value, 1, 1 ,1])
                a_IJ = tf.reduce_sum(u_hat_stopped*v_J_tiled, axis=3, keepdims=True)
                b_IJ += a_IJ

    return v_J

# squash function
def squash(vector):
    vector_squared_norm = tf.reduce_sum(tf.square(vector), axis=-2, keepdims=True)
    scalar_factor = tf.sqrt(vector_squared_norm) / (1 + vector_squared_norm)
    vector_squashed = scalar_factor * vector

    return vector_squashed

# huber loss funciton
def huber_loss(y_true, y_pred, delta=1.0):
    error = tf.abs(y_true - y_pred)
    cond = tf.less(error, delta)
    L2 = 0.5 * tf.square(error)
    L1 = error - 0.5
    loss = tf.where(cond, L2, L1)
    
    return tf.reduce_mean(loss)
