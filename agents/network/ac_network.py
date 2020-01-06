import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import environments.environments
import scipy.stats
from tensorflow.contrib.distributions import MultivariateNormalDiag  # as MultivariateNormalDiag
# from tensorflow.contrib.distributions import Normal
# import tensorflow_probability as tfp

EPS = 1e-6

class ActorCritic_Network(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(ActorCritic_Network, self).__init__(sess, config, [config.actor_lr, config.critic_lr])

        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        self.batch_size = config.batch_size

        self.shared_layer_dim = config.shared_l1_dim
        self.actor_layer_dim = config.actor_l2_dim
        self.critic_layer_dim = config.critic_l2_dim

        self.input_norm = input_norm

        # ac specific params
        self.actor_update = config.actor_update

        self.num_modal = config.num_modal
        self.num_samples = config.num_samples
        self.actor_output_dim = self.num_modal * (1 + 2 * self.action_dim)

        self.entropy_scale = config.entropy_scale
        # self.sigma_scale = 1.0  # config.sigma_scale
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        # Not being used
        self.equal_modal_selection = False
        if config.equal_modal_selection == "True":
            self.equal_modal_selection = True

        # define placeholders
        self.state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.q_action_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.pi_raw_action_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.phase_ph = tf.placeholder(tf.bool)
        self.n_samples_ph = tf.placeholder(tf.int32)

        self.target_state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.target_pi_raw_action_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.target_q_action_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.target_phase_ph = tf.placeholder(tf.bool)
        self.target_n_samples_ph = tf.placeholder(tf.int32)  # this is just dummy placeholder

        self.q_target_ph = tf.placeholder(tf.float32, shape=(None, 1))
        self.q_val_ph = tf.placeholder(tf.float32, shape=(None, 1))
        self.entropy_ph = tf.placeholder(tf.float32, shape=(None, 1))

        # original network
        self.pi_alpha, self.pi_mu, self.pi_std, self.pi_raw_samples, self.pi_samples, self.pi_samples_logprob, self.pi_actions_logprob, self.q_samples_prediction, self.q_actions_prediction = self.build_network(self.state_ph, self.pi_raw_action_ph, self.q_action_ph, self.phase_ph, self.n_samples_ph, scope_name='actorcritic')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actorcritic')

        # Target network
        _, _, _, _, _, _, _, _, self.target_q_actions_prediction = self.build_network(self.target_state_ph, self.target_pi_raw_action_ph, self.target_q_action_ph, self.target_phase_ph, self.target_n_samples_ph, scope_name='target_actorcritic')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actorcritic')

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = [
            tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx]))
            for idx in range(len(self.target_net_params))]

        # Op for init. target network with identical parameter as the original network
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx]) for idx in
                                       range(len(self.target_net_params))]

        # TODO: Currently doesn't support batchnorm
        if self.norm_type == 'batch':
            raise NotImplementedError

        else:
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()

        # Optimization Op
        with tf.control_dependencies(self.batchnorm_ops):
            # TODO: Update loss

            # critic Update
            self.critic_loss = tf.reduce_mean(tf.squared_difference(self.q_target_ph, self.q_actions_prediction))
            self.critic_optimize = tf.train.AdamOptimizer(self.learning_rate[1]).minimize(self.critic_loss)

            # Actor update
            # Loglikelihood
            self.actor_loss_ll = self.get_actor_loss_ll(self.pi_actions_logprob, self.q_val_ph, self.entropy_ph)
            self.actor_optimize_ll = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.actor_loss_ll)

            # CEM
            self.actor_loss_cem = self.get_actor_loss_cem(self.pi_actions_logprob)
            self.actor_optimize_cem = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.actor_loss_cem)

            # Reparam
            self.actor_loss_reparam = self.get_actor_loss_reparam(self.pi_samples_logprob, self.q_samples_prediction)
            self.actor_optimize_reparam = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.actor_loss_reparam)

        # # # Get the gradient of the policy w.r.t. the action
        # self.temp_alpha, self.temp_mean, self.temp_sigma, self.temp_action, self.policy_func = self.get_policyfunc()
        # self.policy_action_grads = tf.gradients(self.policy_func, self.temp_action)

    # TODO: combine this with network()
    def build_network(self, state_ph, pi_raw_action_ph, q_action_ph, phase_ph, n_samples_ph, scope_name):
        with tf.variable_scope(scope_name):
            # normalize inputs
            if self.norm_type != 'none':
                # I don't think clip by value is necessary
                # tf.clip_by_value(self.input_norm.normalize(state_ph), self.state_min, self.state_max)
                state_ph = self.input_norm.normalize(state_ph)

            pi_alpha, pi_mu, pi_std, pi_raw_samples, pi_samples, pi_samples_logprob, pi_actions_logprob, q_samples_prediction, q_actions_prediction = self.network(state_ph, pi_raw_action_ph, q_action_ph, phase_ph, n_samples_ph)

        return pi_alpha, pi_mu, pi_std, pi_raw_samples, pi_samples, pi_samples_logprob, pi_actions_logprob, q_samples_prediction, q_actions_prediction

    def network(self, inputs, pi_raw_action, q_action, phase, num_samples):
        # TODO: Remove alpha (not using multimodal)
        # shared net
        shared_net = tf.contrib.layers.fully_connected(inputs, self.shared_layer_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        shared_net = self.apply_norm(shared_net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # action branch
        pi_net = tf.contrib.layers.fully_connected(shared_net, self.actor_layer_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=None,
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        pi_net = self.apply_norm(pi_net, activation_fn=tf.nn.relu, phase=phase, layer_num=2)

        # no activation
        pi_mu = tf.contrib.layers.fully_connected(pi_net, self.num_modal * self.action_dim,
                                                                   activation_fn=None,
                                                                   weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True),
                                                                   # weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                                   weights_regularizer=None,
                                                                   # tf.contrib.layers.l2_regularizer(0.001),
                                                                   biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True))
        # biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        pi_logstd = tf.contrib.layers.fully_connected(pi_net, self.num_modal * self.action_dim,
                                                                    activation_fn=tf.tanh,
                                                                    weights_initializer=tf.random_uniform_initializer(0,1),
                                                                    weights_regularizer=None,
                                                                    # tf.contrib.layers.l2_regularizer(0.001),
                                                                    biases_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3))

        pi_alpha = tf.contrib.layers.fully_connected(pi_net, self.num_modal, activation_fn=tf.tanh,
                                                                    weights_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3),
                                                                    weights_regularizer=None,
                                                                    # tf.contrib.layers.l2_regularizer(0.001),
                                                                    biases_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3))

        # reshape output
        assert (self.num_modal == 1)

        # pi_mu = tf.reshape(pi_mu, [-1, self.num_modal, self.action_dim])
        # pi_logstd = tf.reshape(pi_logstd, [-1, self.num_modal, self.action_dim])
        # pi_alpha = tf.reshape(pi_alpha, [-1, self.num_modal, 1])

        pi_mu = tf.reshape(pi_mu, [-1, self.action_dim])
        pi_logstd = tf.reshape(pi_logstd, [-1, self.action_dim])
        pi_alpha = tf.reshape(pi_alpha, [-1, 1])

        # exponentiate logstd
        # pi_std = tf.exp(tf.scalar_mul(self.sigma_scale, pi_logstd))
        pi_std = tf.exp(self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (pi_logstd + 1))

        # construct MultivariateNormalDiag dist.
        mvn = MultivariateNormalDiag(
            loc=pi_mu,
            scale_diag=pi_std
        )

        if self.actor_update == "reparam":
            # pi = mu + tf.random_normal(tf.shape(mu)) * std
            # logp_pi = self.gaussian_likelihood(pi, mu, log_std)

            # pi_mu: (batch_size, action_dim)

            # (batch_size x num_samples, action_dim)
            # If updating multiple samples
            stacked_pi_mu = tf.expand_dims(pi_mu, 1)
            stacked_pi_mu = tf.tile(stacked_pi_mu, [1, num_samples, 1])
            stacked_pi_mu = tf.reshape(stacked_pi_mu, (-1, self.action_dim))

            stacked_pi_std = tf.expand_dims(pi_std, 1)
            stacked_pi_std = tf.tile(stacked_pi_std, [1, num_samples, 1])
            stacked_pi_std = tf.reshape(stacked_pi_std, (-1, self.action_dim))

            noise = tf.random_normal(tf.shape(stacked_pi_mu))

            # (batch_size * num_samples, action_dim)
            pi_raw_samples = stacked_pi_mu + noise * stacked_pi_std
            pi_raw_samples_logprob = self.gaussian_loglikelihood(pi_raw_samples, stacked_pi_mu, stacked_pi_std)

            pi_raw_samples = tf.reshape(pi_raw_samples, (-1, num_samples, self.action_dim))
            pi_raw_samples_logprob = tf.reshape(pi_raw_samples_logprob, (-1, num_samples, self.action_dim))

        else:
            pi_raw_samples_og = mvn.sample(num_samples)

            # dim: (batch_size, num_samples, action_dim)
            pi_raw_samples = tf.transpose(pi_raw_samples_og, [1, 0, 2])

            # get raw logprob
            pi_raw_samples_logprob_og = mvn.log_prob(pi_raw_samples_og)
            pi_raw_samples_logprob = tf.transpose(pi_raw_samples_logprob_og, [1, 0, 2])

        # apply tanh
        pi_mu = tf.tanh(pi_mu)
        pi_samples = tf.tanh(pi_raw_samples)

        pi_samples_logprob = pi_raw_samples_logprob - tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - pi_samples ** 2, l=0, u=1) + 1e-6), axis=1)

        pi_mu = tf.multiply(pi_mu, self.action_max)
        pi_samples = tf.multiply(pi_samples, self.action_max)

        # compute logprob for input action
        pi_raw_actions_logprob = mvn.log_prob(pi_raw_action)
        pi_action = tf.tanh(pi_raw_action)
        pi_actions_logprob = pi_raw_actions_logprob - tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - pi_action ** 2, l=0, u=1) + 1e-6), axis=1)

        # TODO: Remove alpha
        # compute softmax prob. of alpha
        max_alpha = tf.reduce_max(pi_alpha, axis=1, keepdims=True)
        pi_alpha = tf.subtract(pi_alpha, max_alpha)
        pi_alpha = tf.exp(pi_alpha)

        normalize_alpha = tf.reciprocal(tf.reduce_sum(pi_alpha, axis=1, keepdims=True))
        pi_alpha = tf.multiply(normalize_alpha, pi_alpha)

        # Q branch
        with tf.variable_scope('qf'):
            q_actions_prediction = self.q_network(shared_net, q_action, phase)
        with tf.variable_scope('qf', reuse=True):
            # if len(tf.shape(pi_samples)) == 3:
            pi_samples_reshaped = tf.reshape(pi_samples, (self.batch_size * num_samples, self.action_dim))
            # else:
            #     assert(len(tf.shape(pi_samples)) == 2)
            #     pi_samples_reshaped = pi_samples
            q_samples_prediction = self.q_network(shared_net, pi_samples_reshaped, phase)

        # print(pi_raw_action, pi_action)
        # print(pi_raw_actions_logprob, pi_raw_actions_logprob)
        # print(pi_action, pi_actions_logprob)

        return pi_alpha, pi_mu, pi_std, pi_raw_samples, pi_samples, pi_samples_logprob, pi_actions_logprob, q_samples_prediction, q_actions_prediction

    def q_network(self, qnet_layer, qnet_action, phase):
        q_net = tf.contrib.layers.fully_connected(tf.concat([qnet_layer, qnet_action], 1), self.critic_layer_dim,
                                                  activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True),
                                                  # tf.truncated_normal_initializer(), \
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True))

        q_net = self.apply_norm(q_net, activation_fn=tf.nn.relu, phase=phase, layer_num=3)
        qnet_prediction = tf.contrib.layers.fully_connected(q_net, 1, activation_fn=None,
                                                                 weights_initializer=tf.random_uniform_initializer(
                                                                     -3e-3, 3e-3),
                                                                 weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                                     0.01),
                                                                 biases_initializer=tf.random_uniform_initializer(-3e-3,
                                                                                                                  3e-3))
        return qnet_prediction

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)

        return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)

    # def tf_normal(self, y, mu, sigma):
    #
    #     # y: batch x action_dim
    #     # mu: batch x num_modal x action_dim
    #     # sigma: batch x num_modal x action_dim
    #
    #     # stacked y: batch x num_modal x action_dim
    #     stacked_y = tf.expand_dims(y, 1)
    #     stacked_y = tf.tile(stacked_y, [1, self.num_modal, 1])
    #
    #     return tf.reduce_prod(
    #         tf.sqrt(1.0 / (2 * np.pi * tf.square(sigma))) * tf.exp(-tf.square(stacked_y - mu) / (2 * tf.square(sigma))), axis=2)

    # def get_policyfunc(self):
    #
    #     alpha = tf.placeholder(tf.float32, shape=(None, self.num_modal, 1), name='temp_alpha')
    #     mean = tf.placeholder(tf.float32, shape=(None, self.num_modal, self.action_dim), name='temp_mean')
    #     sigma = tf.placeholder(tf.float32, shape=(None, self.num_modal, self.action_dim), name='temp_sigma')
    #     action = tf.placeholder(tf.float32, shape=(None, self.action_dim), name='temp_action')
    #
    #     result = self.tf_normal(action, mean, sigma)
    #     result = tf.multiply(result, tf.squeeze(alpha, axis=2))
    #     result = tf.reduce_sum(result, 1, keepdims=True)
    #
    #     return alpha, mean, sigma, action, result

    def get_actor_loss_ll(self, pi_actions_logprob, q_val, entropy):

        neg_ll = -pi_actions_logprob
        loss = tf.multiply(neg_ll, q_val - entropy)

        return tf.reduce_mean(loss)

    def get_actor_loss_cem(self, pi_actions_logprob):

        neg_ll = -pi_actions_logprob
        loss = neg_ll

        return tf.reduce_mean(loss)

    def get_actor_loss_reparam(self, pi_samples_logprob, q_pi_val):

        # pi_loss = tf.reduce_mean(self.q_pi - self.entropy_scale * self.logp_pi)
        return tf.reduce_mean(q_pi_val - self.entropy_scale * pi_samples_logprob)

    def gaussian_loglikelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))

        return tf.reduce_sum(pre_sum, axis=1)

    def get_loglikelihood(self, state_batch, raw_action_batch):

        action_ll_batch = self.sess.run(self.pi_actions_logprob, feed_dict={
            self.state_ph: state_batch,
            self.pi_raw_action_ph: raw_action_batch,
            self.phase_ph: True
        })

        # for pair in zip(raw_action_batch, action_ll_batch):
        #     print('raw_action: {}, ll: {}'.format(pair[0], pair[1]))
        # input()
        return action_ll_batch

    # def policy_action_gradients(self, alpha, mean, sigma, action):
    #
    #     return self.sess.run(self.policy_action_grads, feed_dict={
    #         self.temp_alpha: alpha,
    #         self.temp_mean: mean,
    #         self.temp_sigma: sigma,
    #         self.temp_action: action
    #     })

    def train_critic(self, state_batch, action_batch, q_target_batch):
        return self.sess.run([self.q_actions_prediction, self.critic_optimize], feed_dict={
            self.state_ph: state_batch,
            self.q_action_ph: action_batch,
            self.q_target_ph: q_target_batch,
            self.phase_ph: True
        })

    def train_actor_ll(self, state_batch, action_batch, q_val, entropy):

        return self.sess.run(self.actor_optimize_ll, feed_dict={
            self.state_ph: state_batch,
            self.pi_raw_action_ph: action_batch,
            self.phase_ph: True,
            self.q_val_ph: q_val,
            self.entropy_ph: entropy
            # self.n_samples_ph: 1,
            # self.state_val: q_val_mean
        })

    def train_actor_cem(self, state_batch, raw_action_batch):
        # args [inputs, actions, phase]
        return self.sess.run(self.actor_optimize_cem, feed_dict={
            self.state_ph: state_batch,
            self.pi_raw_action_ph: raw_action_batch,
            self.phase_ph: True
        })

    def train_actor_reparam(self, state_batch):
        # args [inputs, actions, phase]
        return self.sess.run(self.actor_optimize_reparam, feed_dict={
            self.state_ph: state_batch,
            self.phase_ph: True,
            self.n_samples_ph: 1
        })

    def predict_q(self, state_batch, action_batch, phase):

        return self.sess.run(self.q_actions_prediction, feed_dict={
            self.state_ph: state_batch,
            self.q_action_ph: action_batch,
            self.phase_ph: phase
        })

    def predict_q_target(self, state_batch, action_batch, phase):

        return self.sess.run(self.target_q_actions_prediction, feed_dict={
            self.target_state_ph: state_batch,
            self.target_q_action_ph: action_batch,
            self.target_phase_ph: phase
        })

    # bandit setting
    def predict_true_q(self, inputs, action):

        q_val_batch = [getattr(environments.environments, self.config.env_name).reward_func(a[0]) for a in action]
        return np.expand_dims(q_val_batch, -1)

    # return sampled actions
    def sample_action(self, state_batch, phase, is_single_sample):
        if is_single_sample:
            n_samples = 1
        else:
            n_samples = self.num_samples

        # print("num_samples: {}".format(n_samples))
        raw_sampled_actions, sampled_actions = self.sess.run(
            [self.pi_raw_samples, self.pi_samples], feed_dict={
                self.state_ph: state_batch,
                self.phase_ph: phase,
                self.n_samples_ph: n_samples
            })

        # print()
        # print('dist batch shape: {}, dist_event shape: {}, n_samples: {}'.format(dist_info[0], dist_info[1], n_samples))
        # print('og sampled actions shape: {}'.format(np.shape(sampled_actions_og)))
        # print('sampled actions shape: {}'.format(np.shape(sampled_actions)))
        return raw_sampled_actions, sampled_actions

    # return uniformly sampled actions (batchsize x num_samples x action_dim)
    def sample_uniform_action(self, batch_size):
        return self.rng.uniform(self.action_min, self.action_max, size=(batch_size, self.num_samples, self.action_dim))

    def predict_action(self, state_batch, phase):

        # alpha: batchsize x num_modal x 1
        # mean: batchsize x num_modal x action_dim
        pi_alpha, pi_mu, pi_std = self.sess.run(
            [self.pi_alpha, self.pi_mu, self.pi_std], feed_dict={
                self.state_ph: state_batch,
                self.phase_ph: phase
            })

        assert(np.shape(pi_mu) == (np.shape(state_batch)[0], self.action_dim))

        self.setModalStats(pi_alpha[0], pi_mu[0], pi_std[0])

        # assuming unimodal
        # alpha = np.squeeze(alpha, axis=2)
        # if self.equal_modal_selection:
        #     max_idx = self.rng.randint(0, self.num_modal, size=len(mean))
        # else:
        #     max_idx = np.argmax(alpha, axis=1)
        #
        # best_mean = [m[idx] for idx, m in zip(max_idx, mean)]

        return pi_mu

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_actions_prediction, feed_dict={self.state_ph: np.expand_dims(state, 0),
                                                                                  self.q_action_ph: np.expand_dims([action], 0),
                                                                                  self.phase_ph: False})

    def getTrueQFunction(self, state):
        return lambda action: self.predict_true_q(np.expand_dims(state, 0), np.expand_dims([action], 0))

    def getPolicyFunction(self, alpha, mean, sigma):

        # alpha = np.squeeze(alpha, axis=1)
        # mean = np.squeeze(mean, axis=1)
        # sigma = np.squeeze(sigma, axis=1)

        if self.equal_modal_selection:
            return lambda action: np.sum((np.ones(self.num_modal) * (1.0 / self.num_modal)) * np.multiply(
                np.sqrt(1.0 / (2 * np.pi * np.square(sigma))),
                np.exp(-np.square(action - mean) / (2.0 * np.square(sigma)))))
        else:
            return lambda action: np.sum(alpha * np.multiply(
                np.sqrt(1.0 / (2 * np.pi * np.square(sigma))),
                np.exp(-np.square(action - mean) / (2.0 * np.square(sigma)))))

    def setModalStats(self, alpha, mean, sigma):
        self.saved_alpha = alpha
        self.saved_mean = mean
        self.saved_sigma = sigma

    def getModalStats(self):
        return self.saved_alpha, self.saved_mean, self.saved_sigma
