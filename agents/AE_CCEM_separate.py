from __future__ import print_function
import sys
import time

from agents.base_agent import BaseAgent # for python3
#from agents import Agent # for python2
import random
import numpy as np
import tensorflow as tf

from agents.network.ccem_actor_network import CCEM_ActorNetwork
from agents.network.expert_network import ExpertNetwork

from utils.running_mean_std import RunningMeanStd
from experiment import write_summary

class AE_CCEM_Separate_Network(object):
    def __init__(self, state_dim, state_min, state_max, action_dim, action_min, action_max, use_external_exploration, config, random_seed):

        self.write_log = config.write_log
        self.use_external_exploration = use_external_exploration

        #record step n for tf Summary
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

        # Cross Entropy Method Params
        self.rho = config.rho
        self.num_samples = config.num_samples

        # Action selection mode
        self.action_selection = config.action_selection

        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.sess = tf.Session()

            actor_layer_dim = [config.actor_l1_dim, config.actor_l2_dim]
            expert_layer_dim = [config.expert_l1_dim, config.expert_l2_dim]

            self.actor_network = CCEM_ActorNetwork(self.sess, self.input_norm, actor_layer_dim, state_dim, state_min,
                                                   state_max, action_dim, action_min, action_max, config.actor_lr,
                                                   config.tau, config.rho, config.num_samples, config.num_modal, config.action_selection,
                                                   norm_type=self.norm_type)

            self.expert_network = ExpertNetwork(self.sess, self.input_norm, expert_layer_dim, state_dim, state_min,
                                                state_max, action_dim, action_min, action_max, config.expert_lr,
                                                config.tau, norm_type=self.norm_type)

            self.sess.run(tf.global_variables_initializer())

    def take_action(self, state, is_train, is_start):

        # print('action candidates', action_init)
        if is_train:

            # just return the best action
            if self.use_external_exploration:
                chosen_action = self.actor_network.predict_action(np.expand_dims(state, 0), False)[0]
                chosen_action = np.clip(chosen_action, self.action_min, self.action_max)
            else:
                action_init = self.actor_network.sample_action(np.expand_dims(state, 0), False)[0] # single state so first idx

                # Choose one random action among n actions
                idx = np.random.randint(len(action_init))
                action_init = action_init[idx]

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
                    alpha, mean, sigma = self.actor_network.getModalStats()
                    for i in range(len(alpha)):
                        write_summary(self.writer, self.train_global_steps, alpha[i], tag='train/alpha%d' % i)
                        write_summary(self.writer, self.train_global_steps, mean[i], tag='train/mean%d' % i)
                        write_summary(self.writer, self.train_global_steps, sigma[i], tag='train/sigma%d' % i)

        else:
            # Use mean directly
            chosen_action = self.actor_network.predict_action(np.expand_dims(state, 0), False)[0]
            chosen_action = np.clip(chosen_action, self.action_min, self.action_max)

            if self.write_log:
                self.eval_global_steps += 1
                if is_start:
                    self.eval_ep_count += 1

                if self.eval_global_steps % 1 == 0:
                    alpha, mean, sigma = self.actor_network.getModalStats()
                    func1 = self.expert_network.getQFunction(state)
                    func2 = self.actor_network.getPolicyFunction(alpha, mean, sigma)

                    self.expert_network.plotFunction(func1, func2, state, mean, self.action_min, self.action_max,
                                                    display_title='ep: ' + str(self.eval_ep_count) + ', steps: ' + str(self.eval_global_steps),
                                                    save_title='steps_' + str(self.eval_global_steps),
                                                    save_dir=self.writer.get_logdir(), ep_count=self.eval_ep_count, show=False)

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
        
        #assert(chosen_action >= self.action_min and chosen_action <= self.action_max)
        
        # print('take Action', chosen_action)


        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        ###### Expert Update #####
                
        batch_size = np.shape(state_batch)[0]

        # compute target

        next_action_batch_init_target = self.actor_network.predict_action_target(next_state_batch, True)
        next_action_batch_final_target = next_action_batch_init_target

        # batchsize * n
        target_q = self.expert_network.predict_q_target(next_state_batch, next_action_batch_final_target, True)

        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))

        y_i = reward_batch + gamma_batch * target_q

        predicted_q_val, _ = self.expert_network.train_expert(state_batch, action_batch, y_i)
        self.episode_ave_max_q += np.amax(predicted_q_val)


        ###### Actor Update #####
        # for each transition, n actions
        #shape: (batchsize , n actions, action_dim)
        action_batch_init = self.actor_network.sample_action(state_batch, True)

        # reshape (batchsize * n , action_dim)
        action_batch_init = np.reshape(action_batch_init, (batch_size * self.num_samples, self.action_dim))

        # Gradient Ascent
        #action_batch_final = self.expert_network.gradient_ascent(state_batch, action_batch_init, self.gd_alpha, self.gd_max_steps, self.gd_stop, True) # do ascent on original network
        action_batch_final = action_batch_init

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

        # print('action batch', action_batch_final)

        q_val = self.expert_network.predict_q(stacked_state_batch, action_batch_final, True)
        q_val = np.reshape(q_val, (batch_size, self.num_samples))

        # print('q val', q_val)
        # print()
        # print(np.shape(action_batch_final))
        action_batch_final = np.reshape(action_batch_final, (batch_size, self.num_samples, self.action_dim))
        # print(np.shape(action_batch_final))

        # Find threshold (top (1-rho) percentile)
        selected_idxs = list(map(lambda x: x.argsort()[::-1][:int(self.num_samples*self.rho)], q_val))

        # print('seleted idxs:', np.shape(selected_idxs))

        action_list = []
        for action, idx in zip(action_batch_final, selected_idxs):
            action_list.append(action[idx])


        # restack states (batchsize * top_idx_num, 1)
        stacked_state_batch = None
        for state in state_batch:
            stacked_one_state = np.tile(state, (int(self.num_samples*self.rho), 1))

            if stacked_state_batch is None:
                stacked_state_batch = stacked_one_state
            else:
                stacked_state_batch = np.concatenate((stacked_state_batch, stacked_one_state), axis=0)

        action_list = np.reshape(action_list, (batch_size * int(self.num_samples*self.rho), self.action_dim))


        self.actor_network.train_actor(stacked_state_batch, action_list)
        # print('l: '+str(loss))
        # print('@@@ Gradient @@@')
        # self.hydra_network.printGradient(stacked_state_batch, action_list)
        # input()
        # Update target networks
        self.actor_network.update_target_network()
        self.expert_network.update_target_network()

    def get_sum_maxQ(self): # Returns sum of max Q values
        return self.episode_ave_max_q

    def reset(self):
        self.episode_ave_max_q = 0.0


class AE_CCEM_separate(BaseAgent):
    def __init__(self, env, config, random_seed):
        super(AE_CCEM_separate, self).__init__(env, config)

        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Network
        self.network = AE_CCEM_Separate_Network(self.state_dim, self.state_min, self.state_max,
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

                # if using an external exploration policy
                if self.use_external_exploration:
                    # print('action before', action)
                    action = self.exploration_policy.generate(action, self.cum_steps)
                    # print('action after', action)
                    # input()
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

