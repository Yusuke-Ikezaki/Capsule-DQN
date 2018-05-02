import random
import numpy as np

from dqn import DDQN
from config import cfg

class Agent:
    def __init__(self, action_n):
        self.action_n = action_n
        self.dqn = DDQN(input_shape=[cfg.batch_size, 84, 84, cfg.state_length], action_n=action_n)
    
    def epsilon_greedy(self, s, epsilon):
        if np.random.rand() < epsilon:
            a = random.randrange(self.action_n)
        else:
            a = self.dqn.greedy(s[np.newaxis])

        return a
