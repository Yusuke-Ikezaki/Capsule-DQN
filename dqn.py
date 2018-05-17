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
    def __init__(self, action_n):
        # create Q networks
        Q = QFunction(action_n, scope="Q")
        target_Q = QFunction(action_n, scope="target_Q")
        
        # define placeholders
        self.s = tf.placeholder(shape=[None, cfg.height, cfg.width, cfg.state_length], dtype=tf.float32)
        self.a = tf.placeholder(shape=[cfg.batch_size, 1], dtype=tf.int32)
        self.r = tf.placeholder(shape=[cfg.batch_size, 1], dtype=tf.float32)
        self.done = tf.placeholder(shape=[cfg.batch_size, 1], dtype=tf.float32)
        self.next_s = tf.placeholder(shape=[cfg.batch_size, cfg.height, cfg.width, cfg.state_length], dtype=tf.float32)
        
        # predict Q values
        self.probs = Q(self.s)
        
        # add offset
        first = tf.expand_dims(tf.range(cfg.batch_size), axis=1)
        
        # choose Q value
        q_val = tf.expand_dims(tf.gather_nd(self.probs, tf.concat([first, self.a], axis=1)), axis=1)
        
        # create teacher
        a_max = tf.expand_dims(tf.argmax(Q(self.next_s, reuse=True), axis=1, output_type=tf.int32), axis=1)
        target_q_val = tf.expand_dims(tf.gather_nd(target_Q(self.next_s), tf.concat([first, a_max], axis=1)), axis=1)
        y = self.r + cfg.gamma*(1.0 - self.done)*target_q_val
        
        # calculate loss
        self.loss = huber_loss(y, q_val)

        # update Q
        opt = tf.train.RMSPropOptimizer(0.001, epsilon=1e-8)
        self.train_op = opt.minimize(self.loss)

        # update target Q
        self.target_train_op = copy_params(Q, target_Q)
    
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
    
    def greedy(self, sess, s):
        probs = sess.run(self.probs, feed_dict={self.s: s})
        return np.argmax(probs)
