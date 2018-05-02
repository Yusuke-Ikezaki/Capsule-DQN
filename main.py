import os
import numpy as np
import tensorflow as tf
from hyperdash import Experiment

from environment import Environment
from agent import Agent
from config import cfg

class Recorder:
    def __init__(self):
        self.global_step = 0

sess = tf.Session()

def play(env, actor, recorder, is_training=True):
    # initialize environment
    s = env.reset()
    # total reward
    R = 0
    # time step
    step = 0
    # time step limit
    limit = env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps")
    
    # play game
    while True:
        # epsilon decay
        epsilon = (1.0 if recorder.global_step < cfg.replay_start_size else \
            max(cfg.min_epsilon, np.interp(recorder.global_step, [0, cfg.decay], [1.0, cfg.min_epsilon]))) if is_training else 0.01
            
        # select action
        a = actor.epsilon_greedy(s, epsilon)
        
        # take action
        next_s, r, done, _ = env.step(a)
       
        if is_training:
            # store experience
            actor.dqn.set_exp((s, a, r*cfg.reward_scale, done, next_s))
            # update model
            if recorder.global_step >= cfg.replay_start_size:
                actor.dqn.update(sess)
            # update target
            if recorder.global_step % cfg.sync_freq == 0:
                actor.dqn.update_target(sess)
            
            recorder.global_step += 1
               
        # set state
        s = next_s
        
        R += r
        step += 1
    
        # render current state
        if cfg.render:
            env.render()
            
        if done or step >= limit:
            break
    
    return R, step

def main(_):
    # build environment
    env = Environment()
    # create actor
    actor = Agent(env.action_space.n)
    # time recorder
    recorder = Recorder()
    # hyperdash experiment
    exp = Experiment("Capsule-DQN")
    
    # model saver
    saver = tf.train.Saver()
    if cfg.restore and os.path.exists(cfg.stored_path):
        saver.restore(sess, cfg.stored_path)
    else:
        sess.run(tf.global_variables_initializer())

    for episode in range(cfg.episode):
        # train actor
        _, _ = play(env, actor, recorder)
        
        print("Episode {} completed.".format(episode))
        
        # save model
        if cfg.save and episode % cfg.save_freq == 0:
            saver.save(sess, cfg.stored_path)

        # evaluate actor
        if episode % cfg.eval == 0:
            R, step = play(env, actor, recorder, is_training=False)
            exp.metric("reward", R)
            exp.metric("step", step)
        
            print("global_step:{}".format(recorder.global_step))

    exp.end()

if __name__ == "__main__":
    tf.app.run()



