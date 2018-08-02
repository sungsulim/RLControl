from __future__ import print_function


import random
import numpy as np
import tensorflow as tf

from utils.running_mean_std import RunningMeanStd
from experiment import write_summary

from agents.base_agent import BaseAgent # for python3
# from agents import BaseAgent  # for python2
from agents.network import entropy_network # for python3
# from network import entropy_network # for python2


class InputConvexNetwork(object):
    def __init__(self, state_dim, state_min, state_max, action_dim, action_min, action_max, params, random_seed):
        # type of normalization: 'none', 'batch', 'layer'
        self.norm_type = params['norm']

        if self.norm_type == 'layer':
            self.input_norm = RunningMeanStd(state_dim)
        else:
            self.input_norm = None

        self.actionMax = action_max
        self.actionMin = action_min

        self.episode_ave_max_q = 0.0
        self.graph = tf.Graph()

        # k in the buddle entropy method
        self.inference_max_steps = params['inference_max_steps']

        #self.inference = 'bundle_entropy'
        self.inference = 'adam'

        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.sess = tf.Session()

            critic_layer_dim = [params['critic_l1_dim'], params['critic_l2_dim']]

            self.critic_network = entropy_network.EntropyNetwork(self.sess, self.input_norm, critic_layer_dim, state_dim, state_min, state_max, action_dim, action_min, action_max, \
                                                                 params['critic_lr'], params['tau'], self.inference, norm_type = self.norm_type)
            self.sess.run(tf.global_variables_initializer())
            self.critic_network.update_target_network()

    def takeAction(self, state):
        # initialize action space
        if self.inference == 'bundle_entropy':
            action_init = np.expand_dims((np.random.uniform(self.actionMin, self.actionMax) - self.actionMin) * 1.0 / (self.actionMax - self.actionMin), 0)
            action_init = np.clip(action_init, 0.0001, 0.9999)
        elif self.inference == 'adam':
            action_init = np.expand_dims(np.random.uniform(self.actionMin, self.actionMax), 0)
        else:
            print('Do not know this inference method!')
            exit()
        action_final = self.critic_network.alg_opt(np.expand_dims(state, 0), action_init, self.inference_max_steps)[0]
        return action_final

    def updateNetwork(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        batch_size = np.shape(state_batch)[0]

        if self.inference == 'bundle_entropy':
            next_action_batch_init_target = np.tile((np.random.uniform(self.actionMin, self.actionMax) - self.actionMin) * 1.0 / (self.actionMax - self.actionMin), (batch_size, 1))
            next_action_batch_init_target = np.clip(next_action_batch_init_target, 0.0001, 0.9999)
        elif self.inference == 'adam':
            next_action_batch_init_target = np.tile(np.random.uniform(self.actionMin, self.actionMax), (batch_size, 1))
        else:
            print('Do not know this inference method!')
            exit()

        next_action_batch_final_target = self.critic_network.alg_opt_target(next_state_batch, next_action_batch_init_target, self.inference_max_steps)

        # compute target
        target_q = self.critic_network.predict_target(next_state_batch, next_action_batch_final_target, True)
        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))
        target_q = np.reshape(target_q, (batch_size, 1))

        y_i = reward_batch + gamma_batch * target_q

        # Update the critic given the targets
        predicted_q_value, _ = self.critic_network.train(state_batch, action_batch, y_i)
        #print('y:', np.sum(y_i))
        #print('predicted value:', np.sum(predicted_q_value))
        #print('action_batch:', action_batch)
        #print('predicted value:', predicted_q_value)
        self.episode_ave_max_q += np.amax(predicted_q_value)

        # Update target networks
        self.critic_network.update_target_network()

    def getSumMaxQ(self):  # Returns sum of max Q values
        return self.episode_ave_max_q

    def reset(self):
        self.episode_ave_max_q = 0.0


class ICNN(BaseAgent):
    def __init__(self, env, config, random_seed):
        super(ICNN, self).__init__(env, config)

        np.random.seed(random_seed)
        random.seed(random_seed)

        self.network = InputConvexNetwork(self.state_dim, self.state_min, self.state_max,
                                          self.action_dim, self.action_min, self.action_max,
                                          config, random_seed=random_seed)

        self.cum_steps = 0  # cumulative steps across episodes

    def start(self, state, is_train):
        return self.take_action(state, is_train)

    def step(self, state, is_train):
        return self.take_action(state, is_train)

    def take_action(self, state, is_train):

        # random action during warmup
        if self.cum_steps < self.warmup_steps:
            action = np.random.uniform(self.action_min, self.action_max)

        else:
            # Train
            if is_train:
                greedy_action = self.network.takeAction(state)
                noise = self.exploration_policy.generate()
                action = np.clip(greedy_action + noise, self.actionMin, self.actionMax)

            # Eval: no noise
            elif is_train == 2:
                action = np.clip(self.network.takeAction(state), self.actionMin, self.actionMax)

            else:
                print("Invalid is_train value")
                exit()

        self.cum_steps += 1
        return action

    def update(self, S, Sp, r, a, episodeEnd):
        if not episodeEnd:
            self.replay_buffer.add(S, a, r, Sp, self.gamma)
            if self.network.norm_type == 'layer':
                self.network.input_norm.update(np.array([S]))

            self.learn()
        else:
            self.replay_buffer.add(S, a, r, Sp, 0.0)
            if self.network.norm_type == 'layer':
                self.network.input_norm.update(np.array([S]))
            self.learn()

    def learn(self):
        if self.replay_buffer.getSize() > max(self.warmup_steps, self.batch_size):
            s, a, r, sp, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.network.updateNetwork(s, a, sp, r, gamma)
        else:
            return






    def getQfunction(self, state):
        return None

    def reset(self):
        self.action_is_greedy = None
        self.network.reset()
        self.exploration_policy.reset()

    # set writer
    def setWriter(self, writer):
        self.network.writer = writer

