import random

from config import cfg

class Memory:
    def __init__(self):
        self.D = []

    def add(self, exp):
        if len(self.D) < cfg.N:
            self.D.append(exp)
        else:
            s, a, r, done, next_s = self.D[0]
            self.D = self.D[1:]
            self.D.append(exp)
            
            del s, a, r, done, next_s

    def sample(self):
        exps = random.sample(self.D, cfg.batch_size)

        return exps
