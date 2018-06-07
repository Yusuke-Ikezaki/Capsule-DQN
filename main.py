import tensorflow as tf
# from hyperdash import Experiment

from environment import Environment, TestEnvironment
from agent import Agent, TestAgent
from config import cfg

def play(env, agent, is_training=True):
    # initialize environment
    s = env.reset()
    # total reward
    R = 0
    # time step
    step = 0
    
    # play game
    while True:
        # select action
        a = agent.get_action(s, is_training)
        
        for _ in range(cfg.action_repeat):
            # take action
            next_s, r, done, _ = env.step(a)
           
            if is_training:
                # update agent
                agent.after_action(s, a, r, done, next_s)
               
            # set state
            s = next_s
            
            R += r
            step += 1
    
            # render current state
            if cfg.render:
                env.render()

            if done:
                break
        
        if done:
            break
    
    return R, step

def main(_):
    # build environment
    env = TestEnvironment()
    # create agent
    agent = TestAgent(env.action_space.n)
    # hyperdash experiment
    # exp = Experiment("Capsule-DQN")

    for episode in range(cfg.episode):
        # train agent
        _, _ = play(env, agent)
        
        # print("Episode {} completed.".format(episode))

        # evaluate agent
        if episode % cfg.eval_freq == 0:
            R, step = play(env, agent, is_training=False)
            # exp.metric("reward", R)
            # exp.metric("step", step)

    # exp.end()

if __name__ == "__main__":
    tf.app.run()
