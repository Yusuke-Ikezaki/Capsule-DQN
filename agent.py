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
        self.dqn = DDQN(action_n)
        
        # create replay memory
        self.replay_memory = Memory()
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(cfg.log_dir, self.sess.graph)
        
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
        epsilon = max(cfg.min_epsilon, np.interp(self.t, [cfg.replay_start_size, cfg.decay], [cfg.max_epsilon, cfg.min_epsilon])) if is_training else 0.01
        
        # epsilon greedy
        if np.random.rand() < epsilon:
            # explore
            a = random.randrange(self.action_n)
        else:
            # exploit
            a = self.dqn.greedy(self.sess, s[np.newaxis])
        
        return a

    def after_action(self, s, a, r, done, next_s):
        # store experience
        self.replay_memory.add((s, a, r, done, next_s))
        
        if self.t >= cfg.replay_start_size:
            # update network
            if self.t % cfg.train_freq == 0:
                self.dqn.update(self.sess, self.replay_memory)
        
            # update target network
            if self.t % cfg.sync_freq == 0:
                self.dqn.update_target(self.sess)
        
            # save network
            if cfg.save and self.t % cfg.save_freq == 0:
                self.saver.save(self.sess, cfg.stored_path)
                print("Successfully saved network.")
            
        if done:
            print("t: {}".format(self.t))
    
        self.t += 1



from layers import conv, flatten, fc

class TestAgent:
    def __init__(self, action_n):
        self.action_n = action_n
        self.t = 0
        
        # parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.step = 0
        self.episode = 0
        
        # create replay memory
        self.replay_memory = Memory()
        
        # create Q network
        self.s, self.q_values = self.build_network("Q")
        
        # create target Q network
        self.st, self.target_q_values = self.build_network("target_Q")
        
        # define network update operation
        self.a, self.y, self.loss, self.train_op = self.build_train_op()
        
        # define target network update operation
        self.target_train_op = self.build_target_train_op()
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(cfg.log_dir, self.sess.graph)
        
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())
        
        # load network
        if cfg.restore:
            self.load_network()

        # initialize target network
        self.sess.run(self.target_train_op)

    def build_network(self, scope):
        with tf.variable_scope(scope):
            X = tf.placeholder(shape=[None, cfg.height, cfg.width, cfg.state_length], dtype=tf.float32)
            
            conv1 = conv(X, [8, 8, cfg.state_length, 32], [1, 4, 4, 1], activation_fn=tf.nn.relu, scope="conv1")
            conv2 = conv(conv1, [4, 4, 32, 64], [1, 2, 2, 1], activation_fn=tf.nn.relu, scope="conv2")
            conv3 = conv(conv2, [3, 3, 64, 64], [1, 1 ,1, 1], activation_fn=tf.nn.relu, scope="conv3")
            flt, dim = flatten(conv3)
            fc1 = fc(flt, dim, 512, activation_fn=tf.nn.relu, scope="fc1")
            output = fc(fc1, 512, self.action_n, scope="output")
            
        return X, output

    def build_train_op(self):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.action_n, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), axis=1)
        
        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
        
        opt = tf.train.AdamOptimizer(cfg.eta)
        train_op = opt.minimize(loss)
        
        return a, y, loss, train_op

    def build_target_train_op(self):
        src_params = [v for v in tf.trainable_variables() if v.name.startswith("Q")]
        dst_params = [v for v in tf.trainable_variables() if v.name.startswith("target_Q")]
        
        target_train_op = [d.assign(s) for s, d in zip(src_params, dst_params)]
            
        return target_train_op
    
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_step = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)
        
        tf.summary.scalar(cfg.env + "/Total Reward/Episode", episode_total_reward)
        tf.summary.scalar(cfg.env + "/Step/Episode", episode_step)
        tf.summary.scalar(cfg.env + "/Average Max Q/Episode", episode_avg_max_q)
        tf.summary.scalar(cfg.env + "/Average Loss/Episode", episode_avg_loss)
    
        summary_vars = [episode_total_reward, episode_step, episode_avg_max_q, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
    
        return summary_placeholders, update_ops, summary_op
        
    def get_action(self, state, is_training=True):
        # epsilon decay
        epsilon = max(cfg.min_epsilon, np.interp(self.t, [cfg.replay_start_size, cfg.decay], [cfg.max_epsilon, cfg.min_epsilon])) if is_training else 0.01
        
        # epsilon greedy
        if np.random.rand() < epsilon:
            # explore
            action = random.randrange(self.action_n)
        else:
            # exploit
            action = np.argmax(np.squeeze(self.sess.run(self.q_values, feed_dict={self.s: state[np.newaxis]})))
        
        return action

    def after_action(self, state, action, reward, done, next_state):
        # store experience
        self.replay_memory.add((state, action, reward, done, next_state))
        
        if self.t >= cfg.replay_start_size:
            # train network
            if self.t % cfg.train_freq == 0:
                self.train_network()
            
            # update target network
            if self.t % cfg.sync_freq == 0:
                self.sess.run(self.target_train_op)
        
            # save network
            if cfg.save and self.t % cfg.save_freq == 0:
                self.saver.save(self.sess, cfg.stored_path)
                print("Successfully saved network.")

        self.total_reward += reward
        self.total_q_max += np.max(self.sess.run(self.q_values, feed_dict={self.s: state[np.newaxis]}))
        self.step += 1
        self.t += 1
                
        if done:
            # write summary
            if self.t >= cfg.replay_start_size:
                stats = [self.total_reward, self.step, self.total_q_max / self.step, self.total_loss / (self.step / cfg.train_freq)]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={self.summary_placeholders[i]: float(stats[i])})
                summary = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary, self.episode + 1)
                    
            # debug
            print("EPISODE: {0:6d}, TIMESTEP: {1:8d}, STEP: {2:5d}, TOTAL_REWARD: {3:3.0f}, AVG_MAX_Q: {4:2.4f}, AVG_LOSS: {5:.5f}".format(self.episode + 1, self.t, self.step, self.total_reward, self.total_q_max / self.step, self.total_loss / (self.step / cfg.train_freq)))
                
            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.step = 0
            self.episode += 1
            
    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        next_state_batch = []
        
        # Sample random minibatch of transition from replay memory
        minibatch = self.replay_memory.sample()
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            done_batch.append(data[3])
            next_state_batch.append(data[4])
        
        # Convert True to 1, False to 0
        done_batch = np.array(done_batch)
        
        target_q_values_batch = self.sess.run(self.target_q_values, feed_dict={self.st: next_state_batch})
        y_batch = reward_batch + (1 - done_batch) * cfg.gamma * np.max(target_q_values_batch, axis=1)
        
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.s: state_batch,
            self.a: action_batch,
            self.y: y_batch
        })
                                
        self.total_loss += loss

    def load_network(self):
        if os.path.exists(cfg.stored_path):
            self.saver.restore(self.sess, cfg.stored_path)
            print("Successfully loaded network.")
        else:
            print("Train new network.")
