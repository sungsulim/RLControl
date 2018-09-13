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
    def __init__(self, config, random_seed):
        super(ActorExpert_Plus_Network_Manager, self).__init__(config)

        # Cross Entropy Method Params
        self.rho = config.rho
        self.num_samples = config.num_samples

        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.sess = tf.Session()
            self.hydra_network = ae_plus_network.ActorExpert_Plus_Network(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

    def take_action(self, state, is_train, is_start):

        greedy_action = self.hydra_network.predict_action(np.expand_dims(state, 0), False)[0]

        if is_train:
            if is_start:
                self.train_ep_count += 1

            if self.use_external_exploration:
                chosen_action = self.exploration_policy.generate(greedy_action, self.train_global_steps)
            else:
                # single state so first idx
                sampled_actions = self.hydra_network.sample_action(np.expand_dims(state, 0), False)[0]

                # Choose one random action among n actions
                idx = np.random.randint(len(sampled_actions))
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

                utils.plot_utils.plotFunction("ActorExpert", [func1, func2], state, greedy_action, chosen_action,
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
        next_action_batch_final_target = self.hydra_network.predict_action_target(next_state_batch, True)

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

        # Currently using Current state batch instead of next state batch
        # (batchsize * n action values)
        # restack states (batchsize * n, 1)
        stacked_state_batch = None

        for state in state_batch:
            stacked_one_state = np.tile(state, (self.num_samples, 1))

            if stacked_state_batch is None:
                stacked_state_batch = stacked_one_state
            else:
                stacked_state_batch = np.concatenate((stacked_state_batch, stacked_one_state), axis=0)

        # reshape (batchsize * n , action_dim)
        action_batch_init = np.reshape(action_batch_init, (batch_size * self.num_samples, self.action_dim))

        # Gradient Ascent
        action_batch_final = self.hydra_network.gradient_ascent(stacked_state_batch, action_batch_init, True)  # do ascent on original network
        # action_batch_final = np.reshape(action_batch_init, (batch_size * self.num_samples, self.action_dim))

        q_val = self.hydra_network.predict_q(stacked_state_batch, action_batch_final, True)
        q_val = np.reshape(q_val, (batch_size, self.num_samples))

        action_batch_final = np.reshape(action_batch_final, (batch_size, self.num_samples, self.action_dim))

        # Find threshold : top (1-rho) percentile
        selected_idxs = list(map(lambda x: x.argsort()[::-1][:int(self.num_samples * self.rho)], q_val))

        action_list = []
        for action, idx in zip(action_batch_final, selected_idxs):
            action_list.append(action[idx])

        # restack states (batchsize * top_idx_num, 1)
        stacked_state_batch = None
        for state in state_batch:
            stacked_one_state = np.tile(state, (int(self.num_samples * self.rho), 1))

            if stacked_state_batch is None:
                stacked_state_batch = stacked_one_state
            else:
                stacked_state_batch = np.concatenate((stacked_state_batch, stacked_one_state), axis=0)

        action_list = np.reshape(action_list, (batch_size * int(self.num_samples * self.rho), self.action_dim))
        self.hydra_network.train_actor(stacked_state_batch, action_list)

        # Update target networks
        self.hydra_network.update_target_network()

    def reset(self):
        if self.exploration_policy:
            self.exploration_policy.reset()


class ActorExpert_Plus(BaseAgent):
    def __init__(self, config, random_seed):
        super(ActorExpert_Plus, self).__init__(config)

        np.random.seed(random_seed)
        random.seed(random_seed)

        # Network Manager
        self.network_manager = ActorExpert_Plus_Network_Manager(config, random_seed=random_seed)

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


