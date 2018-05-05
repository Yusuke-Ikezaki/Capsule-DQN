import os
import random
import numpy as np
import tensorflow as tf

from dqn import DDQN
from memory import Memory
from config import cfg

class Agent:
    def __init__(self, action_n):
        self.action_n = action_n
        self.t = 0
        
        # create DQN
        self.dqn = DDQN(input_shape=[cfg.batch_size, 84, 84, cfg.state_length], action_n=action_n)
        
        # create replay memory
        self.replay_memory = Memory()
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        """
        self.summary_writer = tf.summary.FileWriter(cfg.log_dir, self.sess.graph)
        """
        
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())
    
        # load network
        if cfg.restore and os.path.exists(cfg.stored_path):
            self.saver.restore(self.sess, cfg.stored_path)
            print("Successfully loaded network.")
        else:
            print("Train new network.")
    
    def get_action(self, s, is_training=True):
        # epsilon decay
        epsilon = max(cfg.min_epsilon, np.interp(self.t, [0, cfg.decay], [1.0, cfg.min_epsilon])) if is_training else 0.01
        
        # epsilon greedy
        if self.t < cfg.replay_start_size or np.random.rand() < epsilon:
            a = random.randrange(self.action_n)
        else:
            a = self.dqn.greedy(s[np.newaxis])
        
        return a

    def after_action(self):
        # update model
        if self.t >= cfg.replay_start_size:
            self.dqn.update(self.sess, self.replay_memory)
        
        # update target
        if self.t % cfg.sync_freq == 0:
            self.dqn.update_target(self.sess)
        
        # save network
        if cfg.save and self.t % cfg.save_freq == 0:
            self.saver.save(self.sess, cfg.stored_path)
            print("Successfully saved network.")
    
        self.t += 1
