from __future__ import print_function
import numpy as np
import tensorflow as tf

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import optimal_q_network
from experiment import write_summary
import utils.plot_utils


class OptimalQ_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(OptimalQ_Network_Manager, self).__init__(config)

        self.rng = np.random.RandomState(config.random_seed)

        # Specific Params

        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()
            self.q_network = optimal_q_network.OptimalQ_Network(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

            self.q_network.init_target_network()

    def take_action(self, state, is_train, is_start):

        _, max_action_batch_target = self.q_network.get_max_action(np.expand_dims(state, 0), use_target=False, is_train=False)

        greedy_action = max_action_batch_target[0]
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
                func1 = self.q_network.getQFunction(state)

                raise NotImplementedError
                # utils.plot_utils.plotFunction("DDPG", [func1], state, greedy_action, chosen_action, self.action_min,
                #                               self.action_max,
                #                               display_title='ep: ' + str(
                #                                   self.train_ep_count) + ', steps: ' + str(
                #                                   self.train_global_steps),
                #                               save_title='steps_' + str(self.train_global_steps),
                #                               save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                #                               show=False)

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

        # max_q_batch_target: batch x 1
        # max_action_batch_target: batch x action_dim
        max_q_batch_target, max_action_batch_target = self.q_network.get_max_action(next_state_batch, use_target=True, is_train=True)

        # GA from highest valued action
        # if it is so fine, we might not need this step

        # compute target
        # target_q = self.q_network.predict_target(next_state_batch, self.actor_network.predict_target(next_state_batch, True), True)

        target_q = max_q_batch_target
        batch_size = np.shape(state_batch)[0]
        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))
        target_q = np.reshape(target_q, (batch_size, 1))

        y_i = reward_batch + gamma_batch * target_q

        # Update the critic given the targets
        self.q_network.train(state_batch, action_batch, y_i)

        # Update target networks
        self.q_network.update_target_network()

class OptimalQ(BaseAgent):
    def __init__(self, config):
        network_manager = OptimalQ_Network_Manager(config)
        super(OptimalQ, self).__init__(config, network_manager)





