from __future__ import print_function
import random
import numpy as np
import tensorflow as tf

from agents.base_agent import BaseAgent  # for python3
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import actor_network
from agents.network import critic_network

from utils.running_mean_std import RunningMeanStd

from experiment import write_summary
import utils.plot_utils


class DDPG_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(DDPG_Network_Manager, self).__init__(config)


        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()

            self.actor_network = actor_network.ActorNetwork(self.sess, self.input_norm, config)
            self.critic_network = critic_network.CriticNetwork(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

            ######
            self.actor_network.init_target_network()
            self.critic_network.init_target_network()
            ######

    def take_action(self, state, is_train, is_start):

        greedy_action = self.actor_network.predict(np.expand_dims(state, 0), False)[0]

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
                func1 = self.critic_network.getQFunction(state)

                utils.plot_utils.plotFunction("DDPG", [func1], state, greedy_action, chosen_action, self.action_min,
                                              self.action_max,
                                              display_title='ep: ' + str(
                                                  self.train_ep_count) + ', steps: ' + str(
                                                  self.train_global_steps),
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

        # compute target
        target_q = self.critic_network.predict_target(next_state_batch, self.actor_network.predict_target(next_state_batch, True), True)

        batch_size = np.shape(state_batch)[0]
        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))
        target_q = np.reshape(target_q, (batch_size, 1))

        y_i = reward_batch + gamma_batch * target_q

        # Update the critic given the targets
        predicted_q_value, _ = self.critic_network.train(state_batch, action_batch, y_i)

        # Update the actor using the sampled gradient
        a_outs = self.actor_network.predict(state_batch, True)
        a_grads = self.critic_network.action_gradients(state_batch, a_outs, True)
        self.actor_network.train(state_batch, a_grads[0])

        # Update target networks
        self.actor_network.update_target_network()
        self.critic_network.update_target_network()

    def reset(self):
        if self.exploration_policy:
            self.exploration_policy.reset()


class DDPG(BaseAgent):
    def __init__(self, config):
        network_manager = DDPG_Network_Manager(config)
        super(DDPG, self).__init__(config, network_manager)


