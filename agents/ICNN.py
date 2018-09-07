from __future__ import print_function


import random
import numpy as np
import tensorflow as tf
from utils.running_mean_std import RunningMeanStd
from experiment import write_summary

from agents.base_agent import BaseAgent # for python3
# from base_agent import BaseAgent  # for python2
from agents.network import entropy_network # for python3
# from network import entropy_network # for python2
import utils.plot_utils


class InputConvexNetwork(object):
    def __init__(self, state_dim, state_min, state_max, action_dim, action_min, action_max, config, random_seed):

        self.write_log = config.write_log
        self.write_plot = config.write_plot

        # record step n for tf Summary
        self.train_global_steps = 0
        self.eval_global_steps = 0
        self.train_ep_count = 0
        self.eval_ep_count = 0


        # type of normalization: 'none', 'batch', 'layer'
        self.norm_type = config.norm

        if self.norm_type == 'input_norm' or self.norm_type == 'layer' or self.norm_type == 'batch':
            self.input_norm = RunningMeanStd(state_dim)
        else:
            assert(self.norm_type == 'none')
            self.input_norm = None

        self.action_max = action_max
        self.action_min = action_min

        self.episode_ave_max_q = 0.0
        self.graph = tf.Graph()

        self.inference_max_steps = config.inference_max_steps

        self.inference = 'bundle_entropy'
        # self.inference = 'adam'

        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.sess = tf.Session()

            critic_layer_dim = [config.critic_l1_dim, config.critic_l2_dim]

            self.critic_network = entropy_network.EntropyNetwork(self.sess, self.input_norm, critic_layer_dim,
                                                                 state_dim, state_min, state_max,
                                                                 action_dim, action_min, action_max,
                                                                 config.critic_lr, config.tau, self.inference,
                                                                 norm_type=self.norm_type)
            self.sess.run(tf.global_variables_initializer())
            self.critic_network.update_target_network()

    def take_action(self, state, is_train, is_start):
        # initialize action space
        if self.inference == 'bundle_entropy':
            action_init = np.expand_dims((np.random.uniform(self.action_min, self.action_max) - self.action_min) * 1.0 / (self.action_max - self.action_min), 0)
            action_init = np.clip(action_init, 0.0001, 0.9999)

        elif self.inference == 'adam':
            action_init = np.expand_dims(np.random.uniform(self.action_min, self.action_max), 0)
        else:
            print('Do not know this inference method!')
            exit()


        action_final = self.critic_network.alg_opt(np.expand_dims(state, 0), action_init, self.inference_max_steps)[0]

        if is_train:
            self.train_global_steps += 1

            # MOVED OUTSIDE TO PLOT BOTH EXPL AND GREEDY ACTION
            # if self.write_plot:
            #     if is_start:
            #         self.train_ep_count += 1
            #
            #     func1 = self.critic_network.getQFunction(state)
            #     # func2 = self.getPolicyFunction(best_action, covmat)
            #
            #     utils.plot_utils.plotFunction("ICNN", [func1], state, action_final, self.action_min, self.action_max,
            #                                   display_title='ep: ' + str(self.train_ep_count) + ', steps: ' + str(
            #                                       self.train_global_steps),
            #                                   save_title='steps_' + str(self.train_global_steps),
            #                                   save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count, show=False)

        return action_final

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        batch_size = np.shape(state_batch)[0]

        if self.inference == 'bundle_entropy':
            next_action_batch_init_target = np.tile((np.random.uniform(self.action_min, self.action_max) - self.action_min) * 1.0 / (self.action_max - self.action_min), (batch_size, 1))
            next_action_batch_init_target = np.clip(next_action_batch_init_target, 0.0001, 0.9999)
        elif self.inference == 'adam':
            next_action_batch_init_target = np.tile(np.random.uniform(self.action_min, self.action_max), (batch_size, 1))
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

    def get_sum_maxQ(self):  # Returns sum of max Q values
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
        return self.take_action(state, is_train, is_start=True)

    def step(self, state, is_train):
        return self.take_action(state, is_train, is_start=False)

    def take_action(self, state, is_train, is_start):

        # random action during warmup
        if self.cum_steps < self.warmup_steps:
            action = np.random.uniform(self.action_min, self.action_max)

        else:
            action = self.network.take_action(state, is_train, is_start)

            # Train
            if is_train:

                greedy_action = action
                # if using an external exploration policy
                if self.use_external_exploration:
                    action = self.exploration_policy.generate(greedy_action, self.cum_steps)

                # only increment during training, not evaluation
                self.cum_steps += 1

                if self.write_plot:
                    if is_start:
                        self.network.train_ep_count += 1

                    func1 = self.network.critic_network.getQFunction(state)
                    # func2 = self.getPolicyFunction(best_action, covmat)

                    utils.plot_utils.plotFunction("ICNN", [func1], state, greedy_action, action, self.action_min,
                                                  self.action_max,
                                                  display_title='ep: ' + str(self.network.train_ep_count) + ', steps: ' + str(
                                                      self.network.train_global_steps),
                                                  save_title='steps_' + str(self.network.train_global_steps),
                                                  save_dir=self.network.writer.get_logdir(), ep_count=self.network.train_ep_count,
                                                  show=False)

            action = np.clip(action, self.action_min, self.action_max)

        return action

    def update(self, state, next_state, reward, action, is_terminal, is_truncated):

        # Add to experience replay buffer
        if not is_truncated:
            if not is_terminal:
                self.replay_buffer.add(state, action, reward, next_state, self.gamma)
            else:
                self.replay_buffer.add(state, action, reward, next_state, 0.0)

        # update running mean/std
        if self.network.norm_type == 'layer' or self.network.norm_type == 'input_norm':
            self.network.input_norm.update(np.array([state]))

        self.learn()

    def learn(self):

        if self.replay_buffer.get_size() > max(self.warmup_steps, self.batch_size):
            state, action, reward, next_state, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.network.update_network(state, action, next_state, reward, gamma)
        else:
            return

    def get_Qfunction(self, state):
        raise NotImplementedError

    def reset(self):
        self.network.reset()
        if self.use_external_exploration:
            self.exploration_policy.reset()

    # set writer
    def set_writer(self, writer):
        self.network.writer = writer
