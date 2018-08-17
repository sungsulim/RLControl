from __future__ import print_function
import sys
import time

from agents.base_agent import BaseAgent  # for python3
# from agents import Agent # for python2
import random
import numpy as np
import tensorflow as tf
from agents.network import ae_supervised_network
from utils.running_mean_std import RunningMeanStd
from experiment import write_summary


class AE_Supervised_Network(object):
    def __init__(self, state_dim, state_min, state_max, action_dim, action_min, action_max, use_external_exploration,
                 config, random_seed):

        self.write_log = config.write_log
        self.use_external_exploration = use_external_exploration

        # record step n for tf Summary
        self.train_global_steps = 0
        self.eval_global_steps = 0
        self.eval_ep_count = 0

        self.action_min = action_min
        self.action_max = action_max
        self.action_dim = action_dim
        # type of normalization: 'none', 'batch', 'layer', 'input_norm'
        self.norm_type = config.norm

        if self.norm_type is not 'none':
            self.input_norm = RunningMeanStd(state_dim)
        else:
            self.input_norm = None

        self.episode_ave_max_q = 0.0
        self.graph = tf.Graph()

        # Action selection mode
        self.action_selection = config.action_selection

        self.gd_alpha = config.gd_alpha
        self.gd_max_steps = config.gd_max_steps
        self.gd_stop = config.gd_stop

        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.sess = tf.Session()

            self.hydra_network = ae_supervised_network.AE_Supervised_Network(self.sess, self.input_norm,
                                                                 config.shared_l1_dim, config.actor_l2_dim,
                                                                 config.expert_l2_dim,
                                                                 state_dim, state_min, state_max, action_dim,
                                                                 action_min, action_max,
                                                                 config.actor_lr, config.expert_lr, config.tau,
                                                                 config.num_modal, config.action_selection,
                                                                 norm_type=self.norm_type)

            self.sess.run(tf.global_variables_initializer())

    def take_action(self, state, is_train, is_start):

        if is_train:
            # just return the best action
            if self.use_external_exploration:
                chosen_action = self.hydra_network.predict_action(np.expand_dims(state, 0), False)[0]
                chosen_action = np.clip(chosen_action, self.action_min, self.action_max)
            else:
                action_init = self.hydra_network.sample_action(np.expand_dims(state, 0), False)[0][0]  # single batch, and single sample

                # Gradient Ascent
                # action_final = self.expert_network.gradient_ascent(np.expand_dims(state, 0), action_init, self.gd_alpha, self.gd_max_steps, self.gd_stop, False)[0] # do ascent on original network
                action_final = action_init

                chosen_action = action_final
                chosen_action = np.clip(chosen_action, self.action_min, self.action_max)

            ######## LOGGING #########
            if self.write_log:
                self.train_global_steps += 1
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

                if not self.use_external_exploration:
                    alpha, mean, sigma = self.hydra_network.getModalStats()
                    for i in range(len(alpha)):
                        write_summary(self.writer, self.train_global_steps, alpha[i], tag='train/alpha%d' % i)
                        write_summary(self.writer, self.train_global_steps, mean[i], tag='train/mean%d' % i)
                        write_summary(self.writer, self.train_global_steps, sigma[i], tag='train/sigma%d' % i)

                    # plot Q function, and find mode
                    # if self.train_global_steps % 1 == 0:
                    # # if is_start:
                    #     # find modes
                    #     func1 = self.hydra_network.getQFunction(state)
                    #     func2 = self.hydra_network.getPolicyFunction(alpha, mean, sigma)
                    #
                    #     self.hydra_network.plotFunc(func1, func2, state, mean, self.action_min, self.action_max,
                    #                                 display_title='steps: '+str(self.train_global_steps), save_title='steps_'+str(self.train_global_steps), save_dir=self.writer.get_logdir(), show=False)

        else:
            # Use mean directly
            chosen_action = self.hydra_network.predict_action(np.expand_dims(state, 0), False)[0]
            chosen_action = np.clip(chosen_action, self.action_min, self.action_max)

            if self.write_log:
                self.eval_global_steps += 1
                if is_start:
                    self.eval_ep_count += 1

                if self.eval_global_steps % 1 == 0:
                    alpha, mean, sigma = self.hydra_network.getModalStats()
                    func1 = self.hydra_network.getQFunction(state)
                    func2 = self.hydra_network.getPolicyFunction(alpha, mean, sigma)

                    self.hydra_network.plotFunction(func1, func2, state, mean, self.action_min, self.action_max,
                                                    display_title='ep: ' + str(self.eval_ep_count) + ', steps: ' + str(
                                                        self.eval_global_steps),
                                                    save_title='steps_' + str(self.eval_global_steps),
                                                    save_dir=self.writer.get_logdir(), ep_count=self.eval_ep_count,
                                                    show=False)

                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

            #####################
            # # Take best action among n actions
            # action_init = self.actor_network.sample(np.expand_dims(state, 0), False)[0]
            # stacked_states = np.tile(state, (len(action_init), 1))
            # # print('state shape', np.shape(state))
            # # print('stacked state shape', np.shape(stacked_states))

            # # Gradient Ascent
            # # action_final = self.expert_network.gradient_ascent(stacked_states, action_init, self.gd_alpha, self.gd_max_steps, self.gd_stop, False)[0] # do ascent on original network
            # # print('num final actions', len(action_final))
            # action_final = action_init

            # q_val = self.expert_network.predict(stacked_states, action_final, False)

            # # action with highest Q-value
            # chosen_action = action_final[np.argmax(q_val)]
            # ######################################

        # assert(chosen_action >= self.action_min and chosen_action <= self.action_max)

        # print('take Action', chosen_action)

        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        ###### Expert Update #####

        batch_size = np.shape(state_batch)[0]

        # compute target

        next_action_batch_init_target = self.hydra_network.predict_action_target(next_state_batch, True)
        next_action_batch_final_target = next_action_batch_init_target

        # batchsize * n
        target_q = self.hydra_network.predict_q_target(next_state_batch, next_action_batch_final_target, True)

        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))

        y_i = reward_batch + gamma_batch * target_q

        predicted_q_val, _ = self.hydra_network.train_expert(state_batch, action_batch, y_i)
        self.episode_ave_max_q += np.amax(predicted_q_val)

        ###### Actor Update #####
        # for each transition, 1 action
        # shape: (batchsize , 1 action, action_dim)
        action_batch_init = self.hydra_network.sample_action(state_batch, True)

        # reshape (batchsize * 1 , action_dim)
        action_batch_init = np.reshape(action_batch_init, (batch_size * 1, self.action_dim))

        # Gradient Ascent
        action_batch_final = self.hydra_network.gradient_ascent(state_batch, action_batch_init, self.gd_alpha, self.gd_max_steps, self.gd_stop, True)  # do ascent on original network

        # print('action_batch_final shape (should be batch x action_dim)', np.shape(action_batch_final))
        # input()

        self.hydra_network.train_actor(state_batch, action_batch_final)

        # Update target networks
        self.hydra_network.update_target_network()

    def get_sum_maxQ(self):  # Returns sum of max Q values
        return self.episode_ave_max_q

    def reset(self):
        self.episode_ave_max_q = 0.0


class AE_Supervised(BaseAgent):
    def __init__(self, env, config, random_seed):
        super(AE_Supervised, self).__init__(env, config)

        np.random.seed(random_seed)
        random.seed(random_seed)

        # Network
        self.network = AE_Supervised_Network(self.state_dim, self.state_min, self.state_max,
                                       self.action_dim, self.action_min, self.action_max,
                                       self.use_external_exploration, config, random_seed=random_seed)

        self.cum_steps = 0  # cumulative steps across episodes

    def start(self, state, is_train):
        return self.take_action(state, is_train, is_start=True)

    def step(self, state, is_train):
        return self.take_action(state, is_train, is_start=False)

    def take_action(self, state, is_train, is_start):

        if self.cum_steps < self.warmup_steps:
            action = np.random.uniform(self.action_min, self.action_max)

        else:
            action = self.network.take_action(state, is_train, is_start)

            # Train
            if is_train:
                if self.use_external_exploration:
                    action = self.exploration_policy.generate(action, self.cum_steps)

                # only increment during training, not evaluation
                self.cum_steps += 1

            action = np.clip(action, self.action_min, self.action_max)
        return action

    def update(self, state, next_state, reward, action, is_terminal, is_truncated):

        if not is_truncated:
            if not is_terminal:
                self.replay_buffer.add(state, action, reward, next_state, self.gamma)
            else:
                self.replay_buffer.add(state, action, reward, next_state, 0.0)

        if self.network.norm_type is not 'none':
            self.network.input_norm.update(np.array([state]))
        self.learn()

    def learn(self):

        if self.replay_buffer.get_size() > max(self.warmup_steps, self.batch_size):
            state, action, reward, next_state, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.network.update_network(state, action, next_state, reward, gamma)
        else:
            return

    # not implemented
    def get_Qfunction(self, state):
        raise NotImplementedError

    def reset(self):
        self.network.reset()
        if self.exploration_policy:
            self.exploration_policy.reset()

