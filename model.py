import tensorflow as tf

from layers import conv, flatten, fc, capsule
from config import cfg

class QFunction:
    def __init__(self, action_n, scope):
        self.a = action_n
        self.scope = scope
    
    # build model
    def _model(self, X, reuse=False):
        X = tf.cast(X, tf.float32)
        
        with tf.variable_scope(self.scope, reuse=reuse):
            """
            conv1 = conv(X, [9, 9, cfg.state_length, 256], [1, 1, 1, 1], activation_fn=tf.nn.relu)
            primaryCaps = capsule(conv1, num_outputs=32, vec_len=8, kernel=9, strides=2)
            digitCaps = capsule(primaryCaps, num_outputs=self.a, vec_len=16)
            v_length = tf.sqrt(tf.reduce_sum(tf.square(digitCaps), axis=2, keepdims=True))
            output = tf.squeeze(tf.nn.softmax(v_length, axis=1), axis=[2, 3])
            """
            
            conv1 = conv(X, [8, 8, cfg.state_length, 32], [1, 4, 4, 1], activation_fn=tf.nn.relu, scope="conv1")
            conv2 = conv(conv1, [4, 4, 32, 64], [1, 2, 2, 1], activation_fn=tf.nn.relu, scope="conv2")
            conv3 = conv(conv2, [3, 3, 64, 64], [1, 1 ,1, 1], activation_fn=tf.nn.relu, scope="conv3")
            flt, dim = flatten(conv3)
            fc1 = fc(flt, dim, 512, activation_fn=tf.nn.relu, scope="fc1")
            output = fc(fc1, 512, self.a, scope="output")
        
            return output

    def __call__(self, X, reuse=False):
        return self._model(X, reuse=reuse)

