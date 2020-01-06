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

        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        self.batch_size = config.batch_size

        # Custom parameters
        self.num_samples = config.num_samples
        self.rho = config.rho

        self.critic_update = config.critic_update  # expected, sampled, mean(AE)
        self.actor_update = config.actor_update  # cem, ll, reparam

        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()
            self.hydra_network = ac_network.ActorCritic_Network(self.sess, self.input_norm, config)
            self.sess.run(tf.global_variables_initializer())

            self.hydra_network.init_target_network()

        self.sample_for_eval = False
        if config.sample_for_eval == "True":
            self.sample_for_eval = True

        self.use_true_q = False
        if config.use_true_q == "True":
            self.use_true_q = True

        self.add_entropy = False
        self.entropy_scale = 0.0
        if config.add_entropy == "True":
            self.add_entropy = True
            self.entropy_scale = config.entropy_scale

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
                _, chosen_action = self.hydra_network.sample_action(np.expand_dims(state, 0), False, is_single_sample=True)
                chosen_action = chosen_action[0][0]

            self.train_global_steps += 1

            if self.write_log:
                raise NotImplementedError
                # write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')
                #
                # alpha, mean, sigma = self.hydra_network.getModalStats()
                #
                # write_summary(self.writer, self.train_global_steps, alpha[0], tag='train/alpha0')
                # write_summary(self.writer, self.train_global_steps, alpha[1], tag='train/alpha1')
                # write_summary(self.writer, self.train_global_steps, mean[0], tag='train/mean0')
                # write_summary(self.writer, self.train_global_steps, mean[1], tag='train/mean1')
                # write_summary(self.writer, self.train_global_steps, sigma[0], tag='train/sigma0')
                # write_summary(self.writer, self.train_global_steps, sigma[1], tag='train/sigma1')

            if self.write_plot:
                alpha, mean, sigma = self.hydra_network.getModalStats()
                if self.use_true_q:
                    func1 = self.hydra_network.getTrueQFunction(state)
                else:
                    func1 = self.hydra_network.getQFunction(state)
                func2 = self.hydra_network.getPolicyFunction(alpha, mean, sigma)

                utils.plot_utils.plotFunction("ActorCritic_unimodal", [func1, func2], state, [greedy_action, mean], chosen_action,
                                              self.action_min, self.action_max,
                                              display_title='Actor-Critic, steps: ' + str(self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                                              show=False)
        else:
            if is_start:
                self.eval_ep_count += 1

            if self.sample_for_eval:
                # single state so first idx
                # single sample so first idx
                _, chosen_action = self.hydra_network.sample_action(np.expand_dims(state, 0), False, is_single_sample=True)
                chosen_action = chosen_action[0][0]

            else:
                chosen_action = greedy_action

            self.eval_global_steps += 1

            if self.write_log:
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

        # print('chosen_action: {}'.format(chosen_action))
        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        # Critic Update

        # Modified Actor-Critic

        if not self.use_true_q:
            if self.critic_update == "sampled":
                _, next_action_batch = self.hydra_network.sample_action(next_state_batch, True, is_single_sample=True)
                next_action_batch_reshaped = np.reshape(next_action_batch, (self.batch_size * 1, self.action_dim))

                # batchsize * n
                target_q = self.hydra_network.predict_q_target(next_state_batch, next_action_batch_reshaped, True)

            elif self.critic_update == "expected":
                _, next_action_batch = self.hydra_network.sample_action(next_state_batch, True, is_single_sample=False)
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

            elif self.critic_update == 'random_q':

                # batchsize x num_samples x action_dim
                next_action_batch = self.hydra_network.sample_uniform_action(self.batch_size)
                next_action_batch_reshaped = np.reshape(next_action_batch, (self.batch_size * self.num_samples, self.action_dim))

                stacked_next_state_batch = np.repeat(next_state_batch, self.num_samples, axis=0)

                target_q = self.hydra_network.predict_q_target(stacked_next_state_batch, next_action_batch_reshaped, True)
                target_q = np.reshape(target_q, (self.batch_size, self.num_samples))
                target_q = np.max(target_q, axis=1, keepdims=True)  # find max across samples

            else:
                raise ValueError("Invalid self.critic_update config")

            reward_batch = np.reshape(reward_batch, (self.batch_size, 1))
            gamma_batch = np.reshape(gamma_batch, (self.batch_size, 1))

            # compute target : y_i = r_{i+1} + \gamma * max Q'(s_{i+1}, a')
            y_i = reward_batch + gamma_batch * target_q

            predicted_q_val, _ = self.hydra_network.train_critic(state_batch, action_batch, y_i)

        # Actor Update

        # sample actions
        raw_sampled_action_batch, sampled_action_batch = self.hydra_network.sample_action(state_batch, True, is_single_sample=False)

        sampled_action_batch_reshaped = np.reshape(sampled_action_batch,
                                                   (self.batch_size * self.num_samples, self.action_dim))
        raw_sampled_action_batch_reshaped = np.reshape(raw_sampled_action_batch,
                                                   (self.batch_size * self.num_samples, self.action_dim))

        # get Q val
        stacked_state_batch = np.repeat(state_batch, self.num_samples, axis=0)
        if self.use_true_q:
            q_val_batch_reshaped = self.hydra_network.predict_true_q(stacked_state_batch, sampled_action_batch_reshaped)
        else:
            q_val_batch_reshaped = self.hydra_network.predict_q(stacked_state_batch, sampled_action_batch_reshaped, True)
        q_val_batch = np.reshape(q_val_batch_reshaped, (self.batch_size, self.num_samples))

        # LogLikelihood update
        if self.actor_update == "ll":
            # taken from raw_sampled_action_batch
            selected_raw_sampled_action_batch = np.array([a[0] for a in raw_sampled_action_batch])
            selected_q_val_batch = np.array([b[0] for b in q_val_batch])
            selected_q_val_batch = np.expand_dims(selected_q_val_batch, -1)


            # get state val (baseline)
            q_val_mean = np.mean(q_val_batch, axis=1, keepdims=True)

            if self.add_entropy:
                entropy_batch = self.hydra_network.get_loglikelihood(state_batch, selected_raw_sampled_action_batch)
                entropy_batch = np.expand_dims(entropy_batch, -1)
            else:
                entropy_batch = np.zeros((self.batch_size, 1))

            self.hydra_network.train_actor_ll(state_batch, selected_raw_sampled_action_batch, selected_q_val_batch - q_val_mean, self.entropy_scale * entropy_batch)

        if self.actor_update == "ll_update_all":
            # taken from raw_sampled_action_batch
            # selected_raw_sampled_action_batch = np.array([a[0] for a in raw_sampled_action_batch])

            # selected_q_val_batch = np.array([b[0] for b in q_val_batch])
            # selected_q_val_batch = np.expand_dims(selected_q_val_batch, -1)

            # get state val (baseline)
            q_val_mean = np.mean(q_val_batch, axis=1, keepdims=True)

            stacked_q_val_mean = np.repeat(q_val_mean, self.num_samples, axis=0)

            # (batch_size * num_samples, state/action_dim)
            # stacked_state_batch, raw_sampled_action_batch_reshaped, q_val_batch_reshaped,


            if self.add_entropy:
                entropy_batch = self.hydra_network.get_loglikelihood(stacked_state_batch, raw_sampled_action_batch_reshaped)
                entropy_batch = np.expand_dims(entropy_batch, -1)
            else:
                entropy_batch = np.zeros((self.batch_size * self.num_samples, 1))

            self.hydra_network.train_actor_ll(stacked_state_batch, raw_sampled_action_batch_reshaped, q_val_batch_reshaped - stacked_q_val_mean, self.entropy_scale * entropy_batch)


        # CEM update
        elif self.actor_update == "cem":

            # TODO: Update to use raw_sampled_action
            if self.add_entropy:
                # (batch_size * num_samples, 1)
                entropy_batch_reshaped = self.hydra_network.get_loglikelihood(stacked_state_batch, raw_sampled_action_batch_reshaped)
                entropy_batch_reshaped = np.expand_dims(entropy_batch_reshaped, -1)
                entropy_batch = np.reshape(entropy_batch_reshaped, (self.batch_size, self.num_samples))
            else:
                entropy_batch = np.zeros((self.batch_size, self.num_samples))

            # assert(np.shape(q_val_batch) == np.shape(entropy_batch))

            # Find threshold : top (1-rho) percentile
            selected_idxs = list(map(lambda x: x.argsort()[::-1][:int(self.num_samples * self.rho)], q_val_batch - self.entropy_scale * entropy_batch))

            selected_raw_sampled_action_batch = [actions[idxs] for actions, idxs in zip(raw_sampled_action_batch, selected_idxs)]
            selected_raw_sampled_action_batch = np.reshape(selected_raw_sampled_action_batch, (self.batch_size * int(self.num_samples * self.rho), self.action_dim))

            rho_stacked_state_batch = np.repeat(state_batch, int(self.num_samples * self.rho), axis=0)
            self.hydra_network.train_actor_cem(rho_stacked_state_batch, selected_raw_sampled_action_batch)

        elif self.actor_update == "reparam":

            raise NotImplementedError

            # taken from raw_sampled_action_batch
            selected_raw_sampled_action_batch = np.array([a[0] for a in raw_sampled_action_batch])
            selected_q_val_batch = np.array([b[0] for b in q_val_batch])
            selected_q_val_batch = np.expand_dims(selected_q_val_batch, -1)

            # get state val (baseline)
            q_val_mean = np.mean(q_val_batch, axis=1, keepdims=True)

            if self.add_entropy:
                entropy_batch = self.hydra_network.get_loglikelihood(state_batch, selected_raw_sampled_action_batch)
                entropy_batch = np.expand_dims(entropy_batch, -1)
            else:
                entropy_batch = np.zeros((self.batch_size, 1))

            self.hydra_network.train_actor_ll(state_batch, selected_raw_sampled_action_batch, selected_q_val_batch - q_val_mean, self.entropy_scale * entropy_batch)

        else:
            raise ValueError("Invalid  self.actor_update config")

        # Update target networks
        self.hydra_network.update_target_network()


class ActorCritic(BaseAgent):
    def __init__(self, config):

        network_manager = ActorCritic_Network_Manager(config)
        super(ActorCritic, self).__init__(config, network_manager)





