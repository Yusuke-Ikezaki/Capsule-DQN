import random
import numpy as np

from dqn import DDQN
from config import cfg

class Agent:
    def __init__(self, action_n):
        self.action_n = action_n
        self.dqn = DDQN(input_shape=[cfg.batch_size, 84, 84, cfg.state_length], action_n=action_n)
        self.global_step = 0
    
    def get_action(self, s, is_training=True):
        # epsilon decay
        epsilon = (1.0 if self.global_step < cfg.replay_start_size else \
                   max(cfg.min_epsilon, np.interp(self.global_step, [0, cfg.decay], [1.0, cfg.min_epsilon]))) if is_training else 0.01
        
        # epsilon greedy
        if np.random.rand() < epsilon:
            a = random.randrange(self.action_n)
        else:
            a = self.dqn.greedy(s[np.newaxis])
        
        return a

    def after_action(self, sess, s, a, r, done, next_s):
        # store experience
        self.dqn.set_exp((s, a, r*cfg.reward_scale, done, next_s))
        
        # update model
        if self.global_step >= cfg.replay_start_size:
            self.dqn.update(sess)
        # update target
        if self.global_step % cfg.sync_freq == 0:
            self.dqn.update_target(sess)
    
        self.global_step += 1
