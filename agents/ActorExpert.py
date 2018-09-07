from __future__ import print_function

from agents.base_agent import BaseAgent
import random
import numpy as np
import tensorflow as tf
import utils.exploration_policy
from agents.network import ae_network
from utils.running_mean_std import RunningMeanStd
from experiment import write_summary
import utils.plot_utils


class ActorExpert_Network(object):
    def __init__(self, use_external_exploration, config, random_seed):

        self.state_dim = config.state_dim
        self.state_min = config.state_min
        self.state_max = config.state_max

        self.action_dim = config.action_dim
        self.action_min = config.action_min
        self.action_max = config.action_max

        self.write_log = config.write_log
        self.write_plot = config.write_plot
        self.writer = config.writer

        self.use_external_exploration = use_external_exploration

        # record step for tf Summary
        self.train_global_steps = 0
        self.eval_global_steps = 0
        self.train_ep_count = 0
        self.eval_ep_count = 0

        # type of normalization: 'none', 'batch', 'layer', 'input_norm'
        self.norm_type = config.norm

        if config.norm is not 'none':
            self.input_norm = RunningMeanStd(self.state_dim)
        else:
            self.input_norm = None

        # self.episode_ave_max_q = 0.0
        self.graph = tf.Graph()

        # Cross Entropy Method Params
        self.rho = config.rho
        self.num_samples = config.num_samples

        # Action selection mode
        self.action_selection = config.action_selection

        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.sess = tf.Session() 

            self.hydra_network = ae_network.ActorExpert_Network(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

    def take_action(self, state, is_train, is_start):

        if is_train:

            # just return the best action
            if self.use_external_exploration:
                chosen_action = self.hydra_network.predict_action(np.expand_dims(state, 0), False)[0]
                chosen_action = np.clip(chosen_action, self.action_min, self.self.action_max)
            else:
                # single state so first idx
                sampled_actions = self.hydra_network.sample_action(np.expand_dims(state, 0), False)[0]

                # Choose one random action among n actions
                idx = np.random.randint(len(sampled_actions))
                chosen_action = sampled_actions[idx]
                chosen_action = np.clip(chosen_action, self.action_min, self.action_max)

            self.train_global_steps += 1

            if self.write_log:

                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

                if not self.use_external_exploration:
                    alpha, mean, sigma = self.hydra_network.getModalStats()
                    for i in range(len(alpha)):
                        write_summary(self.writer, self.train_global_steps, alpha[i], tag='train/alpha%d' % i)
                        write_summary(self.writer, self.train_global_steps, mean[i], tag='train/mean%d' % i)
                        write_summary(self.writer, self.train_global_steps, sigma[i], tag='train/sigma%d' % i)

            if self.write_plot:
                if is_start:
                    self.train_ep_count += 1

                alpha, mean, sigma = self.hydra_network.getModalStats()
                func1 = self.hydra_network.getQFunction(state)
                func2 = self.hydra_network.getPolicyFunction(alpha, mean, sigma)

                utils.plot_utils.plotFunction("ActorExpert", [func1, func2], state, mean, chosen_action,
                                              self.action_min, self.action_max,
                                              display_title='ep: ' + str(self.train_ep_count) + ', steps: ' + str(self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                                              show=False)

        else:
            # Use mean directly
            chosen_action = self.hydra_network.predict_action(np.expand_dims(state, 0), False)[0]
            chosen_action = np.clip(chosen_action, self.action_min, self.action_max)

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

        # reshape (batchsize * n , action_dim)
        action_batch_final = np.reshape(action_batch_init, (batch_size * self.num_samples, self.action_dim))

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

        q_val = self.hydra_network.predict_q(stacked_state_batch, action_batch_final, True)
        q_val = np.reshape(q_val, (batch_size, self.num_samples))

        action_batch_final = np.reshape(action_batch_final, (batch_size, self.num_samples, self.action_dim))

        # Find threshold : top (1-rho) percentile
        selected_idxs = list(map(lambda x: x.argsort()[::-1][:int(self.num_samples*self.rho)], q_val))

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
        self.hydra_network.train_actor(stacked_state_batch, action_list)

        # Update target networks
        self.hydra_network.update_target_network()

    def reset(self):
        pass


class ActorExpert(BaseAgent):
    def __init__(self, config, random_seed):
        super(ActorExpert, self).__init__(config)

        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Network
        self.network = ActorExpert_Network(self.use_external_exploration, config,
                                           random_seed=random_seed)
        
        self.cum_steps = 0  # cumulative steps across episodes: For warmup steps

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

