from __future__ import print_function
import numpy as np
import tensorflow as tf

from agents.base_agent import BaseAgent  # for python3
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import qt_opt_network

from utils.running_mean_std import RunningMeanStd

from experiment import write_summary
import utils.plot_utils


class QT_OPT_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(QT_OPT_Network_Manager, self).__init__(config)

        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()

            self.qt_opt_network = qt_opt_network.QTOPTNetwork(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

            ######
            self.qt_opt_network.init_target_network()
            ######

    def take_action(self, state, is_train, is_start):


        if is_train:

            sample, mean_std = self.qt_opt_network.sample_action(np.expand_dims(state, 0))
            chosen_action = sample[0][0]
            greedy_action = mean_std[0][0] ## This is changed to a gaussian mixture so is not a single greedy action

            if is_start:
                self.train_ep_count += 1
            self.train_global_steps += 1

            if self.write_log:
                # only good for 1 dim action
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

            if self.write_plot:

                raise NotImplementedError
                # TODO: Check plotting function
                func1 = self.qt_opt_network.getQFunction(state)
                func2 = self.qt_opt_network.getPolicyFunction(mean_std[0][0], mean_std[0][1])

                utils.plot_utils.plotFunction("QT_OPT", [func1, func2], state, greedy_action, chosen_action, self.action_min,
                                              self.action_max,
                                              display_title='ep: ' + str(
                                                  self.train_ep_count) + ', steps: ' + str(
                                                  self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                                              show=False)

            return chosen_action
        else:
            greedy_action = self.qt_opt_network.predict_action(np.expand_dims(state, 0))[0]
            if is_start:
                self.eval_ep_count += 1
            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, greedy_action[0], tag='eval/action_taken')

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


class QT_OPT(BaseAgent):
    def __init__(self, config):
        network_manager = QT_OPT_Network_Manager(config)
        super(QT_OPT, self).__init__(config, network_manager)

