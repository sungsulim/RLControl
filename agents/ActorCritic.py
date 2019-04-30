from __future__ import print_function
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
        self.batch_size = config.batch_size

        # Custom parameters
        self.num_samples = config.num_samples
        self.rho = config.rho

        self.critic_update = config.critic_update  # expected, sampled, mean(AE)
        self.actor_update = config.actor_update  # cem(with uniform sampling), ll

        self.sample_for_eval = False
        if config.sample_for_eval == "True":
            self.sample_for_eval = True

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
                # single sample so first idx
                chosen_action = self.hydra_network.sample_action(np.expand_dims(state, 0), False, is_single_sample=True)[0]

            self.train_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

                alpha, mean, sigma = self.hydra_network.getModalStats()

                write_summary(self.writer, self.train_global_steps, alpha[0], tag='train/alpha0')
                write_summary(self.writer, self.train_global_steps, alpha[1], tag='train/alpha1')
                write_summary(self.writer, self.train_global_steps, mean[0], tag='train/mean0')
                write_summary(self.writer, self.train_global_steps, mean[1], tag='train/mean1')
                write_summary(self.writer, self.train_global_steps, sigma[0], tag='train/sigma0')
                write_summary(self.writer, self.train_global_steps, sigma[1], tag='train/sigma1')

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

            if self.sample_for_eval:
                # single state so first idx
                # single sample so first idx
                chosen_action = self.hydra_network.sample_action(np.expand_dims(state, 0), False, is_single_sample=True)[0]

            else:
                chosen_action = greedy_action

            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        # Critic Update

        # Modified Actor-Critic

        if self.critic_update == "sampled":
            next_action_batch = self.hydra_network.sample_action(next_state_batch, True, is_single_sample=True)
            next_action_batch_reshaped = np.reshape(next_action_batch, (self.batch_size * 1, self.action_dim))

            # batchsize * n
            target_q = self.hydra_network.predict_q_target(next_state_batch, next_action_batch_reshaped, True)

        elif self.critic_update == "expected":
            next_action_batch = self.hydra_network.sample_action(next_state_batch, True, is_single_sample=False)
            next_action_batch_reshaped = np.reshape(next_action_batch, (self.batch_size * self.num_samples, self.action_dim))

            stacked_next_state_batch = np.repeat(next_state_batch, self.num_samples, axis=0)

            # batchsize * n
            target_q = self.hydra_network.predict_q_target(stacked_next_state_batch, next_action_batch_reshaped, True)
            target_q = np.reshape(target_q, (self.batch_size, self.num_samples))
            target_q = np.mean(target_q, axis=1, keepdims=True)  # average across samples

        elif self.critic_update == "mean":
            # Use original Actor
            next_action_batch_final_target = self.hydra_network.predict_action(next_state_batch, True)

            # batchsize * n
            target_q = self.hydra_network.predict_q_target(next_state_batch, next_action_batch_final_target, True)

        else:
            raise ValueError("Invalid self.critic_update config")


        reward_batch = np.reshape(reward_batch, (self.batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (self.batch_size, 1))

        # compute target : y_i = r_{i+1} + \gamma * max Q'(s_{i+1}, a')
        y_i = reward_batch + gamma_batch * target_q

        predicted_q_val, _ = self.hydra_network.train_critic(state_batch, action_batch, y_i)
        stacked_state_batch = np.repeat(state_batch, self.num_samples, axis=0)

        # Actor Update
        # LogLikelihood update
        if self.actor_update == "ll":
            # for each transition, sample again?
            # shape: (batchsize , n actions, action_dim)

            # batch_size x num_samples x action_dim
            action_batch_new = self.hydra_network.sample_action(state_batch, True, is_single_sample=False)
            action_batch_new_picked = np.array([a[0] for a in action_batch_new])

            # reshape (batchsize * n , action_dim)
            action_batch_new_reshaped = np.reshape(action_batch_new,
                                                   (self.batch_size * self.num_samples, self.action_dim))

            q_val_batch_reshaped = self.hydra_network.predict_q(stacked_state_batch, action_batch_new_reshaped,
                                                                True)
            q_val_batch = np.reshape(q_val_batch_reshaped, (self.batch_size, self.num_samples))
            q_val_picked = np.array([[b[0]] for b in q_val_batch])
            q_val_mean = np.mean(q_val_batch, axis=1, keepdims=True)

            self.hydra_network.train_actor_ll(state_batch, action_batch_new_picked, q_val_picked - q_val_mean)

        # CEM update
        elif self.actor_update == "cem":
            action_batch_init = self.hydra_network.sample_action(state_batch, True, is_single_sample=False)

            # reshape (batchsize * n , action_dim)
            action_batch_final = action_batch_init
            action_batch_final_reshaped = np.reshape(action_batch_final, (self.batch_size * self.num_samples, self.action_dim))

            q_val = self.hydra_network.predict_q(stacked_state_batch, action_batch_final_reshaped, True)
            q_val = np.reshape(q_val, (self.batch_size, self.num_samples))

            # Find threshold : top (1-rho) percentile
            selected_idxs = list(map(lambda x: x.argsort()[::-1][:int(self.num_samples * self.rho)], q_val))

            action_list = [actions[idxs] for actions, idxs in zip(action_batch_final, selected_idxs)]

            stacked_state_batch = np.repeat(state_batch, int(self.num_samples * self.rho), axis=0)

            action_list = np.reshape(action_list, (self.batch_size * int(self.num_samples * self.rho), self.action_dim))
            self.hydra_network.train_actor_cem(stacked_state_batch, action_list)

        else:
            raise ValueError("Invalid  self.actor_update config")

        # Update target networks
        self.hydra_network.update_target_network()


class ActorCritic(BaseAgent):
    def __init__(self, config):

        network_manager = ActorCritic_Network_Manager(config)
        super(ActorCritic, self).__init__(config, network_manager)





