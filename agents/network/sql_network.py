import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
from utils.sql_kernel import adaptive_isotropic_gaussian_kernel

EPS = 1e-6


class SoftQlearningNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(SoftQlearningNetwork, self).__init__(sess, config, [config.actor_lr, config.expert_lr])

        self.rng = np.random.RandomState(config.random_seed)

        self.actor_l1_dim = config.actor_l1_dim
        self.actor_l2_dim = config.actor_l2_dim

        self.expert_l1_dim = config.expert_l1_dim
        self.expert_l2_dim = config.expert_l2_dim

        self.input_norm = input_norm

        # specific params
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        # TODO: Switch to entropy scale
        # self.entropy_scale = config.entropy_scale
        self.reward_scale = config.reward_scale
        self.value_n_particles = config.value_n_particles
        self.kernel_n_particles = config.kernel_n_particles
        self.kernel_update_ratio = config.kernel_update_ratio
        self.kernel_fn = adaptive_isotropic_gaussian_kernel

        self.n_updated_actions = int(self.kernel_n_particles * self.kernel_update_ratio)
        self.n_fixed_actions = self.kernel_n_particles - self.n_updated_actions

        self.x_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.a_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.x2_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.a2_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))  # Not gonna be used
        self.r_ph = tf.placeholder(tf.float32, shape=[None])
        self.g_ph = tf.placeholder(tf.float32, shape=[None])

        self.phase_ph = tf.placeholder(tf.bool)

        with tf.variable_scope('main'):
            self.pi, self.pi_svgd, self.q, self.q_svgd, self.fixed_a_svgd, self.updated_a_svgd, _ = self.build_networks(self.x_ph, self.a_ph, self.phase_ph)

        with tf.variable_scope('target'):
            _, _, _, _, _, _, self.q_targ = self.build_networks(self.x2_ph, self.a2_ph, self.phase_ph)

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = tf.group([tf.assign(v_targ, (1 - self.tau) * v_targ + self.tau * v_main)
                                                  for v_main, v_targ in zip(self.get_vars('main'), self.get_vars('target'))])

        self.init_target_net_params = tf.group([tf.assign(v_targ, v_main)
                                                for v_main, v_targ in zip(self.get_vars('main'), self.get_vars('target'))])

        # TODO: Currently doesn't support batchnorm
        if self.norm_type == 'batch':
            raise NotImplementedError

        else:
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()


        # Optimization Op
        with tf.control_dependencies(self.batchnorm_ops):

            ##### Update Expert

            # Equation 10:
            next_value = tf.reduce_logsumexp(self.q_targ, axis=1)
            assert_shape(next_value, [None])

            # Importance weights add just a constant to the value.
            next_value -= tf.log(tf.cast(self.value_n_particles, tf.float32))
            next_value += self.action_dim * np.log(2)

            # \hat Q in Equation 11:
            ys = tf.stop_gradient(self.reward_scale * self.r_ph + self.g_ph * next_value)

            # print(np.shape(self.reward_scale * self.r_ph))
            # print(np.shape(self.g_ph * next_value))
            #
            # print(np.shape(ys))
            assert_shape(ys, [None])

            # Equation 11:
            bellman_residual = 0.5 * tf.reduce_mean((ys - self.q) ** 2)

            td_optimizer = tf.train.AdamOptimizer(self.learning_rate[1])
            train_td_op = td_optimizer.minimize(loss=bellman_residual, var_list=self.get_vars('main/qf'))

            ##### Update Actor

            # Target log-density. Q_soft in Equation 13:
            squash_correction = tf.reduce_sum(
                tf.log(1 - self.fixed_a_svgd ** 2 + EPS), axis=-1)
            log_p = self.q_svgd + squash_correction

            grad_log_p = tf.gradients(log_p, self.fixed_a_svgd)[0]
            grad_log_p = tf.expand_dims(grad_log_p, axis=2)
            grad_log_p = tf.stop_gradient(grad_log_p)
            assert_shape(grad_log_p, [None, self.n_fixed_actions, 1, self.action_dim])

            kernel_dict = self.kernel_fn(xs=self.fixed_a_svgd, ys=self.updated_a_svgd)

            # Kernel function in Equation 13:
            kappa = tf.expand_dims(kernel_dict["output"], dim=3)
            assert_shape(kappa, [None, self.n_fixed_actions, self.n_updated_actions, 1])

            # Stein Variational Gradient in Equation 13:
            action_gradients = tf.reduce_mean(kappa * grad_log_p + kernel_dict["gradient"], reduction_indices=1)
            assert_shape(action_gradients, [None, self.n_updated_actions, self.action_dim])

            # TODO: Not sure if this is correct. Else your get_vars()
            policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main/pi')
            # policy_params = self.get_vars('main/pi')

            # Propagate the gradient through the policy network (Equation 14).
            gradients = tf.gradients(self.updated_a_svgd, policy_params, grad_ys=action_gradients)

            surrogate_loss = tf.reduce_sum([tf.reduce_sum(w * tf.stop_gradient(g)) for w, g in zip(policy_params, gradients)])

            pi_optimizer = tf.train.AdamOptimizer(self.learning_rate[0])
            train_svgd_op = pi_optimizer.minimize(loss=-surrogate_loss, var_list=policy_params)

            self.train_ops = [train_td_op, train_svgd_op]

    def get_vars(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]

    def build_networks(self, state_ph, action_ph, phase_ph):

        batch_size = tf.shape(state_ph)[0]

        # policy
        with tf.variable_scope('pi'):
            latent_shape = (batch_size, self.action_dim)
            latents = tf.random_normal(latent_shape)

            pi = self.policy_network(state_ph, phase_ph, latents)
            # mu, pi, logp_pi = self.apply_squashing_func(mu, pi, logp_pi)

            # make sure actions are in correct range
            pi *= self.action_max[0]

        with tf.variable_scope('pi', reuse=True):
            # stack inputs
            stacked_state_ph = tf.expand_dims(state_ph, 1)
            stacked_state_ph = tf.tile(stacked_state_ph, [1, self.kernel_n_particles, 1])

            latent_shape = (batch_size, self.kernel_n_particles, self.action_dim)
            latents = tf.random_normal(latent_shape)

            pi_svgd = self.policy_network(stacked_state_ph, phase_ph, latents)
            pi_svgd *= self.action_max[0]

        # main q
        with tf.variable_scope('qf'):
            qf_a = self.qf_network(state_ph, action_ph, phase_ph)
            assert_shape(qf_a, [None])

        # q for svgd
        with tf.variable_scope('qf', reuse=True):

            actions = pi_svgd
            assert_shape(actions, [None, self.kernel_n_particles, self.action_dim])

            # SVGD requires computing two empirical expectations over actions
            # (see Appendix C1.1.). To that end, we first sample a single set of
            # actions, and later split them into two sets: `fixed_actions` are used
            # to evaluate the expectation indexed by `j` and `updated_actions`
            # the expectation indexed by `i`.

            fixed_actions, updated_actions = tf.split(actions, [self.n_fixed_actions, self.n_updated_actions], axis=1)
            fixed_actions = tf.stop_gradient(fixed_actions)

            assert_shape(fixed_actions, [None, self.n_fixed_actions, self.action_dim])
            assert_shape(updated_actions, [None, self.n_updated_actions, self.action_dim])

            stacked_state_ph2 = tf.expand_dims(state_ph, 1)
            stacked_state_ph2 = tf.tile(stacked_state_ph2, [1, self.n_fixed_actions, 1])

            qf_svgd = self.qf_network(stacked_state_ph2, fixed_actions, phase_ph)

        # target network
        with tf.variable_scope('qf', reuse=True):

            # target network
            a_targ = tf.random_uniform((batch_size, self.value_n_particles, self.action_dim), -1, 1)

            stacked_state_ph3 = tf.expand_dims(state_ph, 1)
            stacked_state_ph3 = tf.tile(stacked_state_ph3, [1, self.value_n_particles, 1])

            qf_targ = self.qf_network(stacked_state_ph3, a_targ, phase_ph)

            # print(np.shape(stacked_state_ph3), np.shape(a_targ))
            # print(np.shape(qf_targ))
            # input()
            assert_shape(qf_targ, [None, self.value_n_particles])

        return pi, pi_svgd, qf_a, qf_svgd, fixed_actions, updated_actions, qf_targ

    def policy_network(self, state_ph, phase_ph, latents):
        if self.norm_type != 'none':
            inputs = tf.clip_by_value(self.input_norm.normalize(state_ph), self.state_min[0], self.state_max[0])


        # shared net
        action_net = tf.contrib.layers.fully_connected(tf.concat([inputs, latents], -1), self.actor_l1_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        action_net = self.apply_norm(action_net, activation_fn=tf.nn.relu, phase=phase_ph, layer_num=1)

        # action branch
        action_net = tf.contrib.layers.fully_connected(action_net, self.actor_l2_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       # tf.truncated_normal_initializer(),
                                                       weights_regularizer=None,
                                                       # tf.contrib.layers.l2_regularizer(0.001),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        action_net = self.apply_norm(action_net, activation_fn=tf.nn.relu, phase=phase_ph, layer_num=2)

        # No activation
        mu = tf.contrib.layers.fully_connected(action_net, 1 * self.action_dim,
                                                                   activation_fn=None,
                                                                   weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True),
                                                                   # weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                                   weights_regularizer=None,
                                                                   # tf.contrib.layers.l2_regularizer(0.001),
                                                                   biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True))

        # tanh activation
        # log_std = tf.contrib.layers.fully_connected(action_net, 1 * self.action_dim,
        #                                                             activation_fn=tf.tanh,
        #                                                             weights_initializer=tf.random_uniform_initializer(-3e-3,3e-3),
        #                                                             weights_regularizer=None,
        #                                                             # tf.contrib.layers.l2_regularizer(0.001),
        #                                                             biases_initializer=tf.random_uniform_initializer(
        #                                                                 -3e-3, 3e-3))
        #
        # log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        #
        # std = tf.exp(log_std)
        # pi = mu + tf.random_normal(tf.shape(mu)) * std
        # logp_pi = self.gaussian_likelihood(pi, mu, log_std)

        return tf.tanh(mu)

    # def apply_squashing_func(self, mu, pi, logp_pi):
    #     mu = tf.tanh(mu)
    #     pi = tf.tanh(pi)
    #     # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    #     logp_pi -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - pi ** 2, l=0, u=1) + 1e-6), axis=1)
    #     return mu, pi, logp_pi

    # def clip_but_pass_gradient(self, x, l=-1., u=1.):
    #     clip_up = tf.cast(x > u, tf.float32)
    #     clip_low = tf.cast(x < l, tf.float32)
    #
    #     return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)

    def qf_network(self, state_ph, action_ph, phase_ph):
        if self.norm_type != 'none':
            inputs = tf.clip_by_value(self.input_norm.normalize(state_ph), self.state_min[0], self.state_max[0])

        q_net = tf.contrib.layers.fully_connected(tf.concat([inputs, action_ph], -1), self.expert_l1_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        q_net = self.apply_norm(q_net, activation_fn=tf.nn.relu, phase=phase_ph, layer_num=1)

        # Q branch
        q_net = tf.contrib.layers.fully_connected(q_net, self.expert_l2_dim,
                                                  activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True),
                                                  # tf.truncated_normal_initializer(), \
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True))

        q_net = self.apply_norm(q_net, activation_fn=tf.nn.relu, phase=phase_ph, layer_num=3)
        q_val = tf.contrib.layers.fully_connected(q_net, 1, activation_fn=None,
                                                         weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                         weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                         biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        # TODO: Not sure about this. Return first element in last dimension
        q_val = q_val[..., 0]

        return q_val

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        batch_size = np.shape(state_batch)[0]

        # reward_batch = np.reshape(reward_batch, (batch_size, 1))
        # gamma_batch = np.reshape(gamma_batch, (batch_size, 1))

        # TODO: include self.phase_ph?
        return self.sess.run(self.train_ops, feed_dict={
            self.x_ph: state_batch,
            self.a_ph: action_batch,
            self.x2_ph: next_state_batch,
            self.r_ph: reward_batch,
            self.g_ph: gamma_batch
        })

    def take_action(self, state):

        mu = self.sess.run(self.pi, feed_dict={
                self.x_ph: state,
                self.phase_ph: True
            })
        return mu

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getQFunction(self, state):
        # return lambda action: self.sess.run(self.q, feed_dict={
        #     self.x_ph: np.expand_dims(state, 0),
        #     self.a_ph: np.expand_dims([action], 0),
        #     self.phase_ph: False
        # })

        raise NotImplementedError

    def getPolicyFunction(self, state):

        # mean, std = self.sess.run([self.mu, self.std], feed_dict={
        #     self.x_ph: np.expand_dims(state, 0),
        #     self.phase_ph: False
        # })
        # return lambda action: 1/(std * np.sqrt(2 * np.pi)) * np.exp(- (action - mean)**2 / (2 * std**2))

        raise NotImplementedError


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)])