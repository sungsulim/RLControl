import numpy as np
import random
import tensorflow as tf

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import naf_network

from utils.running_mean_std import RunningMeanStd
from experiment import write_summary
import utils.plot_utils


class NAF_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config, random_seed):
        super(NAF_Network_Manager, self).__init__(config)
        
        with self.graph.as_default():
            tf.set_random_seed(random_seed)
            self.sess = tf.Session()
            self.network = naf_network.NAF_Network(self.sess, self.input_norm, config)

            self.sess.run(tf.global_variables_initializer())
            self.network.init_target_network()

    '''return an action to take for each state'''
    def take_action(self, state, is_train, is_start):

        greedy_action = self.network.predict_action(state.reshape(-1, self.state_dim))

        # train
        if is_train:
            if is_start:
                self.train_ep_count += 1

            if self.use_external_exploration:
                chosen_action = self.exploration_policy.generate(greedy_action, self.train_global_steps)
                covmat = None

            else:
                chosen_action, covmat = self.network.sample_action(np.expand_dims(state, 0), greedy_action)
                # chosen_action = chosen_action[0]

            self.train_global_steps += 1
            if self.write_log:
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

            # currently doesn't handle external exploration
            if self.write_plot:
                assert (covmat is not None)
                func1 = self.network.getQFunction(state)
                func2 = self.network.getPolicyFunction(greedy_action, covmat)

                utils.plot_utils.plotFunction("NAF", [func1, func2], state, greedy_action, chosen_action, self.action_min, self.action_max,
                                              display_title='ep: ' + str(self.train_ep_count) + ', steps: ' + str(self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count, show=False)

            return chosen_action

        # eval
        else:
            if is_start:
                self.eval_ep_count += 1
            chosen_action = greedy_action.reshape(-1)
            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

            return chosen_action
    
    def update_network(self, state, action, next_state, reward, gamma):
        target_q = reward + gamma * np.squeeze(self.network.predict_max_q_target(next_state))
        state_batch = state
        action_batch = action

        self.network.train(state_batch, action_batch, target_q)
        # self.sess.run(self.optimize, feed_dict={self.state_input: state.reshape(-1, self.state_dim), self.action_input: action.reshape(-1, self.action_dim), self.target_q_input: target_q.reshape(-1), self.phase: True})
        self.network.update_target_network()

    def reset(self):
        if self.exploration_policy:
            self.exploration_policy.reset()



class NAF(BaseAgent):
    def __init__(self, config, random_seed):
        super(NAF, self).__init__(config)

        np.random.seed(random_seed)
        random.seed(random_seed)

        # Network Manager
        self.network_manager = NAF_Network_Manager(config, random_seed=random_seed)

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




