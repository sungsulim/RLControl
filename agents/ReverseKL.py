from __future__ import print_function

import numpy as np

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager

from agents.network import reversekl_network
from experiment import write_summary
import utils.plot_utils


class ReverseKL_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(ReverseKL_Network_Manager, self).__init__(config)

        # self.logger = EpochLogger()
        self.rng = np.random.RandomState(config.random_seed)

        self.sample_for_eval = False
        if config.sample_for_eval == "True":
            self.sample_for_eval = True

        self.use_true_q = False
        if config.use_true_q == "True":
            self.use_true_q = True

        # define network
        self.network = reversekl_network.ReverseKLNetwork(None, self.input_norm, config)

    def take_action(self, state, is_train, is_start):

        # Train
        if is_train:
            if is_start:
                self.train_ep_count += 1
            self.train_global_steps += 1

            if self.use_external_exploration:
                raise NotImplementedError

            else:
                # Get action from network
                chosen_action = self.network.sample_action(np.expand_dims(state, 0))[0]

            if self.write_log:
                raise NotImplementedError

            if self.write_plot:
                if self.use_true_q:
                    q_func = self.network.getTrueQFunction(state)
                else:
                    q_func = self.network.getQFunction(state)
                pi_func = self.network.getPolicyFunction(state)
                greedy_action = self.network.predict_action(np.expand_dims(state, 0))[0]

                utils.plot_utils.plotFunction("SoftActorCritic", [q_func, pi_func], state,
                                              greedy_action, chosen_action,
                                              self.action_min, self.action_max,
                                              display_title='Reverse KL, steps: ' + str(self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                                              show=False)

        # Eval
        else:
            if self.sample_for_eval:
                # sample action
                chosen_action = self.network.sample_action(np.expand_dims(state, 0))[0]

            else:
                # greedy action (mean)
                chosen_action = self.network.predict_action(np.expand_dims(state, 0))[0]

            if is_start:
                self.eval_ep_count += 1
            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        # Policy Update, Qf and Vf Update
        _ = self.network.update_network(state_batch, action_batch, next_state_batch, reward_batch, gamma_batch)

        # Update target networks
        self.network.update_target_network()


class ReverseKL(BaseAgent):
    def __init__(self, config):
        network_manager = ReverseKL_Network_Manager(config)
        super(ReverseKL, self).__init__(config, network_manager)

