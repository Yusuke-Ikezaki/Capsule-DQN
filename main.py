import tensorflow as tf
from hyperdash import Experiment

from environment import Environment
from agent import Agent
from config import cfg

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
       
        if is_training:
            # store experience
            agent.replay_memory.add((s, a, r, done, next_s))
            # update agent
            agent.after_action()
               
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

    for episode in range(cfg.episode):
        # train agent
        _, _ = play(env, agent)
        
        print("Episode {} completed.".format(episode))
        print("t: {}".format(agent.t))

        # evaluate agent
        if episode % cfg.eval_freq == 0:
            R, step = play(env, agent, is_training=False)
            exp.metric("reward", R)
            exp.metric("step", step)

    exp.end()

if __name__ == "__main__":
    tf.app.run()
