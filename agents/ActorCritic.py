from __future__ import print_function
import random
import numpy as np
import tensorflow as tf

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import ac_network
from experiment import write_summary
import utils.plot_utils


class ActorCritic_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(ActorCritic_Network_Manager, self).__init__(config)

        self.rng = np.random.RandomState(config.random_seed)

        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()
            self.hydra_network = ac_network.ActorCritic_Network(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

            self.hydra_network.init_target_network()

    def take_action(self, state, is_train, is_start):

        greedy_action = self.hydra_network.predict_action(np.expand_dims(state, 0), False)
        greedy_action = greedy_action[0]

        if is_train:
            if is_start:
                self.train_ep_count += 1

            if self.use_external_exploration:
                chosen_action = self.exploration_policy.generate(greedy_action, self.train_global_steps)
            else:
                # single state so first idx
                sampled_action = self.hydra_network.sample_action(np.expand_dims(state, 0), False)[0]
                chosen_action = sampled_action

            self.train_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

            if self.write_plot:
                alpha, mean, sigma = self.hydra_network.getModalStats()
                func1 = self.hydra_network.getQFunction(state)
                func2 = self.hydra_network.getPolicyFunction(alpha, mean, sigma)

                utils.plot_utils.plotFunction("ActorCritic", [func1, func2], state, [greedy_action, mean], chosen_action,
                                              self.action_min, self.action_max,
                                              display_title='ep: ' + str(self.train_ep_count) + ', steps: ' + str(self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                                              show=False)
        else:
            if is_start:
                self.eval_ep_count += 1

            chosen_action = greedy_action
            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        batch_size = np.shape(state_batch)[0]

        # Critic Update

        # TODO: Perhaps do GA on the policy function
        # Use original Actor
        next_action_batch = self.hydra_network.sample_action(next_state_batch, True)

        # batchsize * n
        target_q = self.hydra_network.predict_q_target(next_state_batch, next_action_batch, True)
        
        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))

        # compute target : y_i = r_{i+1} + \gamma * max Q'(s_{i+1}, a')
        y_i = reward_batch + gamma_batch * target_q

        predicted_q_val, _ = self.hydra_network.train_critic(state_batch, action_batch, y_i)

        # Actor Update

        # for each transition, sample again?
        # shape: (batchsize , n actions, action_dim)
        # action_batch_init = self.hydra_network.sample_action(state_batch, True)
        # action_batch_final = action_batch_init
        action_batch_final = action_batch

        q_val_batch = self.hydra_network.predict_q(state_batch, action_batch_final, True)
        self.hydra_network.train_actor(state_batch, action_batch_final, q_val_batch)

        # Update target networks
        self.hydra_network.update_target_network()


class ActorCritic(BaseAgent):
    def __init__(self, config):

        network_manager = ActorCritic_Network_Manager(config)
        super(ActorCritic, self).__init__(config, network_manager)





