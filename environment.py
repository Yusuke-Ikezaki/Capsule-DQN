import cv2
import gym
import numpy as np

from config import cfg

def preprocess(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized_gray = cv2.resize(gray, (84, 84))
    preprocessed = (resized_gray - 127.5) / 127.5
        
    del obs, gray, resized_gray
        
    return preprocessed

class Environment:
    def __init__(self):
        self.env = gym.make(cfg.env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = self.env.spec
        self.states = None

    def reset(self):
        obs = self.env.reset()
        pred = preprocess(obs)
        states = [pred for _ in range(cfg.state_length)]
        self.states = np.stack(states, axis=2)
        
        del obs, pred, states

        return self.states

    def step(self, a):
        cumulated_reward = 0
        
        for _ in range(cfg.action_repeat):
            obs, r, done, info = self.env.step(a)
            cumulated_reward += r
        
            if done:
                break
        
        pred = preprocess(obs)
        last = self.states[:,:,1:]
        self.states = np.concatenate((last, pred[:,:,np.newaxis]), axis=2)
        
        del obs, pred, last

        return self.states, cumulated_reward, done, info

    def render(self):
        self.env.render()
