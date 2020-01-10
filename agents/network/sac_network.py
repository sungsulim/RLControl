import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import environments.environments

EPS = 1e-6


class SoftActorCriticNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(SoftActorCriticNetwork, self).__init__(sess, config, [config.pi_lr, config.qf_vf_lr])

        self.config = config

        self.use_true_q = False
        if config.use_true_q == "True":
            self.use_true_q = True

        self.rng = np.random.RandomState(config.random_seed)

        self.actor_l1_dim = config.actor_l1_dim
        self.actor_l2_dim = config.actor_l2_dim
        self.critic_l1_dim = config.critic_l1_dim
        self.critic_l2_dim = config.critic_l2_dim

        self.input_norm = input_norm

        # specific params
        # self.num_modal = config.num_modal
        # self.LOG_SIG_CAP_MIN = -20
        # self.LOG_SIG_CAP_MAX = 2
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

        self.entropy_scale = config.entropy_scale
        # self.reward_scale = config.reward_scale

        # self.reparameterize = False
        # if config.reparameterize == "True":
        #     self.reparameterize = True

        # TODO: Currently only supports single Gaussian Policy
        # self.pi_output_dim = self.num_modal * (1 + 2 * self.action_dim)
        self.pi_output_dim = 1 * (2 * self.action_dim)

        # TODO:  Define tensorflow ops
        self.x_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.a_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.x2_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.r_ph = tf.placeholder(tf.float32, shape=(None, 1))
        self.g_ph = tf.placeholder(tf.float32, shape=(None, 1))

        # for self.use_true_q
        self.true_q_pi_ph = tf.placeholder(tf.float32, shape=(None, 1))

        self.phase_ph = tf.placeholder(tf.bool)

        with tf.variable_scope('main'):
            self.mu, self.pi, self.logp_pi, self.std, self.q, self.q_pi, self.v = self.build_networks(self.x_ph, self.a_ph, self.phase_ph)

        with tf.variable_scope('target'):
            _, _, _, _, _, _, self.v_targ = self.build_networks(self.x2_ph, self.a_ph, self.phase_ph)

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
        if self.use_true_q:
            with tf.control_dependencies(self.batchnorm_ops):

                # TODO: override self.v_targ, self.q_pi, self.q, self.v
                # Soft actor-critic losses
                pi_loss = tf.reduce_mean(self.entropy_scale * self.logp_pi - self.true_q_pi_ph)

                # Policy train op
                # (has to be separate from value train op, because q1_pi appears in pi_loss)
                pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate[0])
                train_pi_op = pi_optimizer.minimize(pi_loss, var_list=self.get_vars('main/pi'))

                self.train_ops = [pi_loss, self.logp_pi, train_pi_op]
        else:

            with tf.control_dependencies(self.batchnorm_ops):

                # Targets for Q and V regression
                q_backup = tf.stop_gradient(self.r_ph + self.g_ph * self.v_targ)
                # q_backup = tf.stop_gradient(self.reward_scale * self.r_ph + self.g_ph * self.v_targ)

                v_backup = tf.stop_gradient(self.q_pi - self.entropy_scale * self.logp_pi)
                # v_backup = tf.stop_gradient(self.q_pi - self.logp_pi)

                # Soft actor-critic losses
                pi_loss = tf.reduce_mean(self.entropy_scale * self.logp_pi - self.q_pi)
                # pi_loss = tf.reduce_mean(self.logp_pi - self.q_pi)

                q_loss = 0.5 * tf.reduce_mean((q_backup - self.q) ** 2)

                v_loss = 0.5 * tf.reduce_mean((v_backup - self.v) ** 2)
                value_loss = q_loss + v_loss

                # Policy train op
                # (has to be separate from value train op, because q1_pi appears in pi_loss)
                pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate[0])
                train_pi_op = pi_optimizer.minimize(pi_loss, var_list=self.get_vars('main/pi'))

                # Value train op
                # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
                value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate[1])
                value_params = self.get_vars('main/qf') + self.get_vars('main/vf')

                with tf.control_dependencies([train_pi_op]):
                    train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

                self.train_ops = [pi_loss, q_loss, v_loss, self.q, self.v, self.logp_pi,
                            train_pi_op, train_value_op]

    def get_vars(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]

    def build_networks(self, state_ph, action_ph, phase_ph):

        # policy
        with tf.variable_scope('pi'):
            mu, pi, logp_pi, std = self.policy_network(state_ph, phase_ph)
            mu, pi, logp_pi = self.apply_squashing_func(mu, pi, logp_pi)

        # make sure actions are in correct range
        mu *= self.action_max[0]
        pi *= self.action_max[0]

        with tf.variable_scope('qf'):
            qf_a = self.qf_network(state_ph, action_ph, phase_ph)

        with tf.variable_scope('qf', reuse=True):
            qf_pi = self.qf_network(state_ph, pi, phase_ph)

        with tf.variable_scope('vf'):
            vf = self.vf_network(state_ph, phase_ph)

        return mu, pi, logp_pi, std, qf_a, qf_pi, vf

    def qf_network(self, state_ph, action_ph, phase_ph):
        if self.norm_type != 'none':
            inputs = self.input_norm.normalize(state_ph) # tf.clip_by_value(self.input_norm.normalize(state_ph), self.state_min[0], self.state_max[0])

        q_net = tf.contrib.layers.fully_connected(inputs, self.critic_l1_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        q_net = self.apply_norm(q_net, activation_fn=tf.nn.relu, phase=phase_ph, layer_num=1)

        # Q branch
        q_net = tf.contrib.layers.fully_connected(tf.concat([q_net, action_ph], 1), self.critic_l2_dim,
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

        return q_val

    def vf_network(self, state_ph, phase_ph):
        if self.norm_type != 'none':
            inputs = tf.clip_by_value(self.input_norm.normalize(state_ph), self.state_min[0], self.state_max[0])

        v_net = tf.contrib.layers.fully_connected(inputs, self.critic_l1_dim, activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True),
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True))
        v_net = self.apply_norm(v_net, activation_fn=tf.nn.relu, phase=phase_ph, layer_num=1)

        v_net = tf.contrib.layers.fully_connected(v_net, self.critic_l2_dim,
                                                  activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True),
                                                  # tf.truncated_normal_initializer(), \
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True))
        v_net = self.apply_norm(v_net, activation_fn=tf.nn.relu, phase=phase_ph, layer_num=3)

        v_val = tf.contrib.layers.fully_connected(v_net, 1, activation_fn=None,
                                                  weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return v_val

    def policy_network(self, state_ph, phase_ph):

        if self.norm_type != 'none':
            inputs = tf.clip_by_value(self.input_norm.normalize(state_ph), self.state_min[0], self.state_max[0])

        # shared net
        action_net = tf.contrib.layers.fully_connected(inputs, self.actor_l1_dim, activation_fn=None,
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
        log_std = tf.contrib.layers.fully_connected(action_net, 1 * self.action_dim,
                                                                    activation_fn=tf.tanh,
                                                                    # weights_initializer=tf.random_uniform_initializer(-3e-3,3e-3),
                                                                    weights_initializer=tf.random_uniform_initializer(0, 1),
                                                                    weights_regularizer=None,
                                                                    # tf.contrib.layers.l2_regularizer(0.001),
                                                                    biases_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3))

        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        # log_std = tf.scalar_mul(1.0, log_std)
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)

        return mu, pi, logp_pi, std

    def gaussian_likelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))

        return tf.reduce_sum(pre_sum, axis=1)

    def apply_squashing_func(self, mu, pi, logp_pi):
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - pi ** 2, l=0, u=1) + 1e-6), axis=1)
        return mu, pi, logp_pi

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)

        return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        batch_size = np.shape(state_batch)[0]

        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))

        return self.sess.run(self.train_ops, feed_dict={
            self.x_ph: state_batch,
            self.a_ph: action_batch,
            self.x2_ph: next_state_batch,
            self.r_ph: reward_batch,
            self.g_ph: gamma_batch
        })

    def update_network_true_q(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        # batch_size = np.shape(state_batch)[0]

        # reward_batch = np.reshape(reward_batch, (batch_size, 1))
        # gamma_batch = np.reshape(gamma_batch, (batch_size, 1))

        true_q_pi_batch = np.expand_dims(self.predict_true_q(state_batch, action_batch), 1)

        return self.sess.run(self.train_ops, feed_dict={
            self.x_ph: state_batch,
            self.a_ph: action_batch,
            # self.x2_ph: next_state_batch,
            # self.r_ph: reward_batch,
            # self.g_ph: gamma_batch
            self.true_q_pi_ph: true_q_pi_batch
        })

    def predict_action(self, state):

        mu = self.sess.run(self.mu, feed_dict={
                self.x_ph: state,
                self.phase_ph: True
            })
        return mu

    # Should return n actions
    def sample_action(self, state):

        pi = self.sess.run(self.pi, feed_dict={
                self.x_ph: state,
                self.phase_ph: True
            })
        return pi

    def predict_true_q(self, *args):
        # args  (inputs, action, phase)
        inputs = args[0]
        action = args[1]

        return [getattr(environments.environments, self.config.env_name).reward_func(a[0]) for a in action]


    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q, feed_dict={
            self.x_ph: np.expand_dims(state, 0),
            self.a_ph: np.expand_dims([action], 0),
            self.phase_ph: False
        })

    def getPolicyFunction(self, state):

        mean, std = self.sess.run([self.mu, self.std], feed_dict={
            self.x_ph: np.expand_dims(state, 0),
            self.phase_ph: False
        })
        return lambda action: 1/(std * np.sqrt(2 * np.pi)) * np.exp(- (action - mean)**2 / (2 * std**2))

    def getTrueQFunction(self, state):
        return lambda action: self.predict_true_q(np.expand_dims(state, 0), np.expand_dims([action], 0))