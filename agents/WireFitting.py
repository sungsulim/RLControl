import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.running_mean_std import RunningMeanStd

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import wf_network

from experiment import write_summary
import utils.plot_utils


class WireFitting_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(WireFitting_Network_Manager, self).__init__(config)
        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()
            self.network = wf_network.WireFitting_Network(self.sess, self.input_norm, config)

            self.sess.run(tf.global_variables_initializer())
            self.network.init_target_network()

    '''return an action to take for each state'''
    def take_action(self, state, is_train, is_start):

        # TODO: Check how WF picks action
        # TODO: Check if it is clipped
        greedy_action, action_points = self.network.predict_action(state.reshape(-1, self.state_dim))

        # train
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
                func1 = self.network.getQFunction(state)

                utils.plot_utils.plotFunction("WireFitting", [func1], state, greedy_action, chosen_action,
                                              self.action_min, self.action_max,
                                              display_title='ep: ' + str(self.train_ep_count) + ', steps: ' + str(self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(),
                                              ep_count=self.train_ep_count, show=False)

            return chosen_action

        # eval
        else:
            if is_start:
                self.eval_ep_count += 1
            self.eval_global_steps += 1
            chosen_action = greedy_action

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

            return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        # compute target
        target_q = self.network.predict_max_q_target(next_state_batch)

        # batch_size = np.shape(state_batch)[0]
        # reward_batch = np.reshape(reward_batch, (batch_size, 1))
        # gamma_batch = np.reshape(gamma_batch, (batch_size, 1))
        # target_q = np.reshape(target_q, (batch_size, 1))

        y_i = reward_batch + gamma_batch * target_q

        # Update the network given the targets
        self.network.train(state_batch, action_batch, y_i)

        # Update target networks
        self.network.update_target_network()


class WireFitting(BaseAgent):
    def __init__(self, config):
        network_manager = WireFitting_Network_Manager(config)
        super(WireFitting, self).__init__(config, network_manager)
