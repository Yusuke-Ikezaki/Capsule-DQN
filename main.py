import os
import numpy as np
import tensorflow as tf
from hyperdash import Experiment

from environment import Environment
from agent import Agent
from config import cfg

sess = tf.Session()

def play(env, agent, is_training=True):
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
        # select action
        a = agent.get_action(s, is_training)
        
        # take action
        next_s, r, done, _ = env.step(a)
       
        # update agent
        if is_training:
            agent.after_action(sess, s, a, r, done, next_s)
               
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
    # create agent
    agent = Agent(env.action_space.n)
    # hyperdash experiment
    exp = Experiment("Capsule-DQN")
    
    # model saver
    saver = tf.train.Saver()
    if cfg.restore and os.path.exists(cfg.stored_path):
        saver.restore(sess, cfg.stored_path)
    else:
        sess.run(tf.global_variables_initializer())

    for episode in range(cfg.episode):
        # train agent
        _, _ = play(env, agent)
        
        print("Episode {} completed.".format(episode))
        
        # save model
        if cfg.save and episode % cfg.save_freq == 0:
            saver.save(sess, cfg.stored_path)

        # evaluate agent
        if episode % cfg.eval == 0:
            R, step = play(env, agent, is_training=False)
            exp.metric("reward", R)
            exp.metric("step", step)
        
            print("global_step:{}".format(agent.global_step))

    exp.end()

if __name__ == "__main__":
    tf.app.run()



