from __future__ import print_function
import random
import numpy as np
import tensorflow as tf

from agents.base_agent import BaseAgent  # for python3
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import qt_opt_network

from utils.running_mean_std import RunningMeanStd

from experiment import write_summary
import utils.plot_utils


class QT_OPT_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config, random_seed):
        super(QT_OPT_Network_Manager, self).__init__(config)

        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.sess = tf.Session()

            self.qt_opt_network = qt_opt_network.QTOPTNetwork(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

            ######
            self.qt_opt_network.init_target_network()
            ######

    def take_action(self, state, is_train, is_start):
        sample, mean_std = self.qt_opt_network.sample_action(np.expand_dims(state, 0))
        chosen_action = sample[0]
        greedy_action = mean_std[0][0]

        if is_train:

            if is_start:
                self.train_ep_count += 1
            self.train_global_steps += 1

            if self.write_log:
                # only good for 1 dim action
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

            if self.write_plot:
                func1 = self.qt_opt_network.getQFunction(state)

                # utils.plot_utils.plotFunction("QT_OPT", [func1, func2], state, [greedy_action, chosen_action], self.action_min,
                #                               self.action_max,
                #                               display_title='ep: ' + str(
                #                                   self.train_ep_count) + ', steps: ' + str(
                #                                   self.train_global_steps),
                #                               save_title='steps_' + str(self.train_global_steps),
                #                               save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                #                               show=False)
            return chosen_action
        else:
            if is_start:
                self.eval_ep_count += 1
            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

            return greedy_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        # compute target
        target_q = self.qt_opt_network.predict_q_target(next_state_batch, self.qt_opt_network.predict_action(next_state_batch), True)

        batch_size = np.shape(state_batch)[0]
        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))
        target_q = np.reshape(target_q, (batch_size, 1))

        y_i = reward_batch + gamma_batch * target_q

        # Update the critic given the targets
        predicted_q_value, _ = self.qt_opt_network.train(state_batch, action_batch, y_i)

        self.qt_opt_network.update_target_network()

    def reset(self):
        if self.exploration_policy:
            self.exploration_policy.reset()


class QT_OPT(BaseAgent):
    def __init__(self, config, random_seed):
        super(QT_OPT, self).__init__(config)

        np.random.seed(random_seed)
        random.seed(random_seed)

        # Network Manager
        self.network_manager = QT_OPT_Network_Manager(config, random_seed=random_seed)

    def start(self, state, is_train):
        return self.take_action(state, is_train, is_start=True)

    def step(self, state, is_train):
        return self.take_action(state, is_train, is_start=False)

    def take_action(self, state, is_train, is_start):
        if is_train and self.replay_buffer.get_size() < self.warmup_steps:
            action = np.random.uniform(self.action_min, self.action_max)
        else:
            action = self.network_manager.take_action(state, is_train, is_start)
        return action

    def update(self, state, next_state, reward, action, is_terminal, is_truncated):

        if not is_truncated:
            if not is_terminal:
                self.replay_buffer.add(state, action, reward, next_state, self.gamma)
            else:
                self.replay_buffer.add(state, action, reward, next_state, 0.0)

        if self.norm_type is not 'none':
            self.network_manager.input_norm.update(np.array([state]))
        self.learn()
    
    def learn(self):

        if self.replay_buffer.get_size() > max(self.warmup_steps, self.batch_size):
            state, action, reward, next_state, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.network_manager.update_network(state, action, next_state, reward, gamma)
        else:
            return

    def reset(self):
        self.network_manager.reset()



