import numpy as np
import tensorflow as tf

from model import QFunction
from functions import huber_loss
from config import cfg

def copy_params(src, dst):
    src_params = [v for v in tf.trainable_variables() if v.name.startswith(src.scope)]
    dst_params = [v for v in tf.trainable_variables() if v.name.startswith(dst.scope)]
    
    op = [d.assign(s) for s, d in zip(src_params, dst_params)]
        
    return op

class DDQN:
    def __init__(self, input_shape, action_n):
        self.Q = QFunction(action_n, scope="Q")
        target_Q = QFunction(action_n, scope="target_Q")
        
        # Forward Q
        self.s = tf.placeholder(shape=input_shape, dtype=tf.float32)
        self.a = tf.placeholder(shape=[cfg.batch_size, 1], dtype=tf.int32)
        probs = self.Q(self.s)
        
        # add offset
        first = tf.expand_dims(tf.range(cfg.batch_size), axis=1)
        indices = tf.concat(values=[first, self.a], axis=1)
        q_val = tf.expand_dims(tf.gather_nd(probs, indices), axis=1)
        
        # TD target
        self.r = tf.placeholder(shape=[cfg.batch_size, 1], dtype=tf.float32)
        self.done = tf.placeholder(shape=[cfg.batch_size, 1], dtype=tf.float32)
        self.next_s = tf.placeholder(shape=input_shape, dtype=tf.float32)
        
        # DDQN
        a_max = tf.expand_dims(tf.argmax(self.Q(self.next_s, reuse=True), axis=1), axis=1)
        a_max = tf.to_int32(a_max)
        target_q_val = tf.expand_dims(tf.gather_nd(target_Q(self.next_s), tf.concat(values=[first, a_max], axis=1)), axis=1)
        y = self.r + cfg.gamma*(1.0 - self.done)*target_q_val
        loss = huber_loss(y, q_val)

        # Update Q
        opt = tf.train.RMSPropOptimizer(0.001, epsilon=1e-8)
        """
        grads_and_vars = opt.compute_gradients(loss)
        grads_and_vars = [[grad, var] for grad, var in grads_and_vars \
                        if grad is not None and (var.name.startswith("Q") or var.name.startswith("shared"))]
        self.train_op = opt.apply_gradients(grads_and_vars)
        """
        self.train_op = opt.minimize(loss)

        # Update target Q
        self.target_train_op = copy_params(self.Q, target_Q)
    
    def update(self, sess, memory):
        # sample from replay buffer
        samples = memory.sample()
        s = np.asarray([sample[0] for sample in samples], dtype=np.float32)
        a = np.asarray([[sample[1]] for sample in samples], dtype=np.int32)
        r = np.asarray([[sample[2]] for sample in samples], dtype=np.float32)
        done = np.asarray([[sample[3]] for sample in samples], dtype=np.float32)
        next_s = np.asarray([sample[4] for sample in samples], dtype=np.float32)
        
        feed = {self.s:s, self.a:a, self.r:r, self.done:done, self.next_s:next_s}
        _ = sess.run(self.train_op, feed_dict=feed)
    
    def update_target(self, sess):
        _ = sess.run(self.target_train_op)
    
    def greedy(self, s):
        probs = self.Q(s, reuse=True)
        return np.argmax(probs)
