from __future__ import print_function


import random
import numpy as np
import tensorflow as tf
from utils.running_mean_std import RunningMeanStd
from experiment import write_summary

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager

from agents.network import entropy_network  # for python3
import utils.plot_utils


class PartialInputConvex_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(PartialInputConvex_Network_Manager, self).__init__(config)

        self.rng = np.random.RandomState(config.random_seed)
        self.inference_max_steps = config.inference_max_steps

        self.inference = config.inference_type

        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()

            self.entropy_network = entropy_network.EntropyNetwork(self.sess, self.input_norm, config)
            # self.entropy_network = entropy_network.EntropyNetwork(self.sess, self.input_norm, critic_layer_dim,
            #                                                       state_dim, state_min, state_max,
            #                                                       action_dim, action_min, action_max,
            #                                                       config.critic_lr, config.tau, self.inference,
            #                                                       norm_type=self.norm_type)
            self.sess.run(tf.global_variables_initializer())
            self.entropy_network.init_target_network()

    def take_action(self, state, is_train, is_start):
        # initialize action space
        if self.inference == 'bundle_entropy':
            action_init = np.expand_dims((self.rng.uniform(self.action_min, self.action_max) - self.action_min) * 1.0 / (self.action_max - self.action_min), 0)
            action_init = np.clip(action_init, 0.0001, 0.9999)

        elif self.inference == 'adam':
            action_init = np.expand_dims(np.random.uniform(self.action_min, self.action_max), 0)
        else:
            print('Do not know this inference method!')
            exit()

        greedy_action = self.entropy_network.alg_opt(np.expand_dims(state, 0), action_init, self.inference_max_steps)[0]

        if is_train:
            if is_start:
                self.train_ep_count += 1
            self.train_global_steps += 1

            if self.use_external_exploration:
                chosen_action = self.exploration_policy.generate(greedy_action, self.train_global_steps)
            else:
                chosen_action = greedy_action

            if self.write_log:
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')
            if self.write_plot:
                func1 = self.entropy_network.getQFunction(state)

                utils.plot_utils.plotFunction("PICNN", [func1], state, greedy_action, chosen_action, self.action_min,
                                              self.action_max,
                                              display_title='ep: ' + str(self.train_ep_count) + ', steps: ' + str(self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                                              show=False)

        else:
            if is_start:
                self.eval_ep_count += 1
            self.eval_global_steps += 1

            chosen_action = greedy_action

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

        return chosen_action

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

        next_action_batch_final_target = self.entropy_network.alg_opt_target(next_state_batch, next_action_batch_init_target, self.inference_max_steps)

        # compute target
        target_q = self.entropy_network.predict_target(next_state_batch, next_action_batch_final_target, True)
        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))
        target_q = np.reshape(target_q, (batch_size, 1))

        y_i = reward_batch + gamma_batch * target_q

        # Update the critic given the targets
        predicted_q_value, _ = self.entropy_network.train(state_batch, action_batch, y_i)
        #print('y:', np.sum(y_i))
        #print('predicted value:', np.sum(predicted_q_value))
        #print('action_batch:', action_batch)
        #print('predicted value:', predicted_q_value)

        # Update target networks
        self.entropy_network.update_target_network()

    def reset(self):
        if self.exploration_policy:
            self.exploration_policy.reset()


class PICNN(BaseAgent):
    def __init__(self, config):
        network_manager = PartialInputConvex_Network_Manager(config)
        super(PICNN, self).__init__(config, network_manager)