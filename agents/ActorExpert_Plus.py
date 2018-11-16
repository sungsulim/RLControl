from __future__ import print_function
import random
import numpy as np
import tensorflow as tf

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import ae_plus_network
from experiment import write_summary
import utils.plot_utils


class ActorExpert_Plus_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(ActorExpert_Plus_Network_Manager, self).__init__(config)

        self.rng = np.random.RandomState(config.random_seed)

        # Cross Entropy Method Params
        self.rho = config.rho
        self.num_samples = config.num_samples

        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()
            self.hydra_network = ae_plus_network.ActorExpert_Plus_Network(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

            self.hydra_network.init_target_network()

    def take_action(self, state, is_train, is_start):

        # greedy_action = self.hydra_network.predict_action(np.expand_dims(state, 0), False)[0]
        old_greedy_action, greedy_action = self.hydra_network.predict_action(np.expand_dims(state, 0), False)

        old_greedy_action = old_greedy_action[0]
        greedy_action = greedy_action[0]

        if is_train:
            if is_start:
                self.train_ep_count += 1

            if self.use_external_exploration:
                chosen_action = self.exploration_policy.generate(greedy_action, self.train_global_steps)
            else:
                # single state so first idx
                sampled_actions = self.hydra_network.sample_action(np.expand_dims(state, 0), False)[0]

                # Choose one random action among n actions
                idx = self.rng.randint(len(sampled_actions))
                chosen_action = sampled_actions[idx]

            self.train_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

                # Currently only works for 1D action
                # if not self.use_external_exploration:
                #     alpha, mean, sigma = self.hydra_network.getModalStats()
                #     for i in range(len(alpha)):
                #         write_summary(self.writer, self.train_global_steps, alpha[i], tag='train/alpha%d' % i)
                #         write_summary(self.writer, self.train_global_steps, mean[i], tag='train/mean%d' % i)
                #         write_summary(self.writer, self.train_global_steps, sigma[i], tag='train/sigma%d' % i)

            if self.write_plot:
                alpha, mean, sigma = self.hydra_network.getModalStats()
                func1 = self.hydra_network.getQFunction(state)
                func2 = self.hydra_network.getPolicyFunction(alpha, mean, sigma)

                utils.plot_utils.plotFunction("ActorExpert", [func1, func2], state, [greedy_action, old_greedy_action, mean], chosen_action,
                                              self.action_min, self.action_max,
                                              display_title='ep: ' + str(self.train_ep_count) + ', steps: ' + str(
                                                  self.train_global_steps),
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

        # Expert Update

        # TODO: Perhaps do GA on the policy function
        _, next_action_batch_final_target = self.hydra_network.predict_action(next_state_batch, True)

        # batchsize * n
        target_q = self.hydra_network.predict_q_target(next_state_batch, next_action_batch_final_target, True)

        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))

        # compute target : y_i = r_{i+1} + \gamma * max Q'(s_{i+1}, a')
        y_i = reward_batch + gamma_batch * target_q

        predicted_q_val, _ = self.hydra_network.train_expert(state_batch, action_batch, y_i)

        # Actor Update

        # for each transition, n sampled actions
        # shape: (batchsize , n actions, action_dim)
        action_batch_init = self.hydra_network.sample_action(state_batch, True)
        # reshape (batchsize * n , action_dim)
        action_batch_init_reshaped =np.reshape(action_batch_init, (batch_size * self.num_samples, self.action_dim))

        # Currently using Current state batch instead of next state batch
        # (batchsize * n action values)
        # restack states (batchsize * n, 1)

        # stacked_state_batch = None
        #
        # for state in state_batch:
        #     stacked_one_state = np.tile(state, (self.num_samples, 1))
        #
        #     if stacked_state_batch is None:
        #         stacked_state_batch = stacked_one_state
        #     else:
        #         stacked_state_batch = np.concatenate((stacked_state_batch, stacked_one_state), axis=0)

        stacked_state_batch = np.array([np.tile(state, (self.num_samples, 1)) for state in state_batch])
        stacked_state_batch = np.reshape(stacked_state_batch, (batch_size * self.num_samples, self.state_dim))


        # Gradient Ascent
        action_batch_final_reshaped = self.hydra_network.gradient_ascent(stacked_state_batch, action_batch_init_reshaped, True)  # do ascent on original network

        q_val = self.hydra_network.predict_q(stacked_state_batch, action_batch_final_reshaped, True)
        q_val = np.reshape(q_val, (batch_size, self.num_samples))

        action_batch_final = np.reshape(action_batch_final_reshaped, (batch_size, self.num_samples, self.action_dim))

        # Find threshold : top (1-rho) percentile
        selected_idxs = list(map(lambda x: x.argsort()[::-1][:int(self.num_samples * self.rho)], q_val))

        # action_list = []
        # for action, idx in zip(action_batch_final, selected_idxs):
        #     action_list.append(action[idx])
        action_list = [actions[idxs] for actions, idxs in zip(action_batch_final, selected_idxs)]

        # restack states (batchsize * top_idx_num, 1)
        # stacked_state_batch = None
        # for state in state_batch:
        #     stacked_one_state = np.tile(state, (int(self.num_samples * self.rho), 1))
        #
        #     if stacked_state_batch is None:
        #         stacked_state_batch = stacked_one_state
        #     else:
        #         stacked_state_batch = np.concatenate((stacked_state_batch, stacked_one_state), axis=0)

        stacked_state_batch = np.array([np.tile(state, (int(self.num_samples * self.rho), 1)) for state in state_batch])
        stacked_state_batch = np.reshape(stacked_state_batch, (batch_size * int(self.num_samples * self.rho), self.state_dim))

        action_list = np.reshape(action_list, (batch_size * int(self.num_samples * self.rho), self.action_dim))
        self.hydra_network.train_actor(stacked_state_batch, action_list)

        # Update target networks
        self.hydra_network.update_target_network()


class ActorExpert_Plus(BaseAgent):
    def __init__(self, config):

        network_manager = ActorExpert_Plus_Network_Manager(config)
        super(ActorExpert_Plus, self).__init__(config, network_manager)



