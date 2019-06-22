import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import environments.environments

EPS = 1e-6


class SoftActorCriticNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(SoftActorCriticNetwork, self).__init__(sess, config, [config.pi_lr, config.qf_vf_lr])

        self.rng = np.random.RandomState(config.random_seed)

        self.actor_l1_dim = config.actor_l1_dim
        self.actor_l2_dim = config.actor_l2_dim
        self.critic_l1_dim = config.critic_l1_dim
        self.critic_l2_dim = config.critic_l2_dim

        self.input_norm = input_norm

        # specific params
        # self.num_modal = config.num_modal
        self.LOG_SIG_CAP_MIN = -20
        self.LOG_SIG_CAP_MAX = 2

        self.entropy_scale = config.entropy_scale

        self.reparameterize = False
        if config.reparameterize == "True":
            self.reparameterize = True

        # TODO: Currently only supports single Gaussian Policy
        # self.pi_output_dim = self.num_modal * (1 + 2 * self.action_dim)
        self.pi_output_dim = 1 * (2 * self.action_dim)

        # Policy network
        # pi_sample_action, pi_log_pi, dist
        self.pi_inputs, self.pi_phase, self.pi_sample_action, self.pi_log_pi, self.pi_mean, self.pi_log_sigma, self.pi_dist = self.build_pi_network(
            scope_name='pi')
        self.pi_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pi')

        # Q network
        self.q_inputs, self.q_phase, self.q_action, self.q_val = self.build_q_network(scope_name='qf')
        self.q_pi = self.q_pi_network(scope_name='qf')

        self.q_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qf')

        # V network
        self.v_inputs, self.v_phase, self.v_val = self.build_v_network(scope_name='vf')
        self.v_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vf')

        self.target_v_inputs, self.target_v_phase, self.target_v_val = self.build_v_network(scope_name='target_vf')
        self.target_v_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_vf')



        # Op for periodically updating target network with online network weights
        self.update_target_net_params = [
            tf.assign_add(self.target_v_net_params[idx], self.tau * (self.v_net_params[idx] - self.target_v_net_params[idx]))
            for idx in range(len(self.target_v_net_params))]

        # Op for init. target network with identical parameter as the original network
        self.init_target_net_params = [tf.assign(self.target_v_net_params[idx], self.v_net_params[idx]) for idx in
                                       range(len(self.target_v_net_params))]

        # TODO: Currently doesn't support batchnorm
        if self.norm_type == 'batch':
            raise NotImplementedError

        else:
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()

        self.q_backup = tf.placeholder(tf.float32, [None, 1])
        self.v_backup = tf.placeholder(tf.float32, [None, 1])


        # Optimization Op
        with tf.control_dependencies(self.batchnorm_ops):

            # Soft actor-critic losses
            # Qf, Vf Update
            self.qf_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q_val) ** 2)
            self.vf_loss = 0.5 * tf.reduce_mean((self.v_backup - self.v_val) ** 2)

            self.value_loss = self.qf_loss + self.vf_loss
            self.qf_vf_optimize = tf.train.AdamOptimizer(self.learning_rate[1]).minimize(self.value_loss)

            # Pi update
            self.pi_loss = tf.reduce_mean(self.entropy_scale * self.pi_log_pi - self.q_pi)
            self.pi_optimize = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.pi_loss)

    def build_q_network(self, scope_name):

        # self.q_inputs, self.q_phase, self.q_action, self.q_val
        with tf.variable_scope(scope_name):
            q_inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="q_network_input_state")
            q_phase = tf.placeholder(tf.bool, name="q_network_input_phase")
            q_action = tf.placeholder(tf.float32, shape=(None, self.action_dim), name="q_network_input_action")

            # normalize inputs
            if self.norm_type != 'none':
                q_inputs = tf.clip_by_value(self.input_norm.normalize(q_inputs), self.state_min, self.state_max)

            q_val = self.q_network(q_inputs, q_action, q_phase)

        return q_inputs, q_phase, q_action, q_val

    def q_pi_network(self, scope_name):
        with tf.variable_scope(scope_name, reuse=True):
            q_pi = self.q_network(self.q_inputs, self.pi_sample_action, self.q_phase)
        return q_pi

    def q_network(self, inputs, action, phase):
        # shared net
        q_net = tf.contrib.layers.fully_connected(inputs, self.critic_l1_dim, activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True),
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True))

        q_net = self.apply_norm(q_net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # Q branch
        q_net = tf.contrib.layers.fully_connected(tf.concat([q_net, action], 1), self.critic_l2_dim,
                                                  activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True),
                                                  # tf.truncated_normal_initializer(), \
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True))

        q_net = self.apply_norm(q_net, activation_fn=tf.nn.relu, phase=phase, layer_num=3)
        q_val = tf.contrib.layers.fully_connected(q_net, 1, activation_fn=None,
                                                  weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        return q_val

    def build_v_network(self, scope_name):
        # self.v_inputs, self.v_phase, self.v_val
        with tf.variable_scope(scope_name):
            v_inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="v_network_input_state")
            v_phase = tf.placeholder(tf.bool, name="v_network_input_phase")

            # normalize inputs
            if self.norm_type != 'none':
                v_inputs = tf.clip_by_value(self.input_norm.normalize(v_inputs), self.state_min, self.state_max)

            v_val = self.v_network(v_inputs, v_phase)

        return v_inputs, v_phase, v_val

    def v_network(self, inputs, phase):
        # shared net
        v_net = tf.contrib.layers.fully_connected(inputs, self.critic_l1_dim, activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True),
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True))

        v_net = self.apply_norm(v_net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # Q branch
        v_net = tf.contrib.layers.fully_connected(v_net, self.critic_l2_dim, activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True),
                                                  # tf.truncated_normal_initializer(), \
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True))

        v_net = self.apply_norm(v_net, activation_fn=tf.nn.relu, phase=phase, layer_num=3)
        v_val = tf.contrib.layers.fully_connected(v_net, 1, activation_fn=None,
                                                  weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        return v_val

    def build_pi_network(self, scope_name):
        with tf.variable_scope(scope_name):
            pi_inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="pi_network_input_state")
            pi_phase = tf.placeholder(tf.bool, name="pi_network_input_phase")

            # normalize inputs
            if self.norm_type != 'none':
                pi_inputs = tf.clip_by_value(self.input_norm.normalize(pi_inputs), self.state_min, self.state_max)

            pi_sample_action, pi_log_pi, pi_mean, pi_log_sigma, dist = self.pi_network(pi_inputs, pi_phase)

        return pi_inputs, pi_phase, pi_sample_action, pi_log_pi, pi_mean, pi_log_sigma, dist

    def pi_network(self, inputs, phase):
        # shared net
        pi_net = tf.contrib.layers.fully_connected(inputs, self.actor_l1_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        pi_net = self.apply_norm(pi_net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # action branch
        pi_net = tf.contrib.layers.fully_connected(pi_net, self.actor_l2_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       # tf.truncated_normal_initializer(),
                                                       weights_regularizer=None,
                                                       # tf.contrib.layers.l2_regularizer(0.001),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        pi_net = self.apply_norm(pi_net, activation_fn=tf.nn.relu, phase=phase, layer_num=2)

        # No activation (squash later)
        pi_mean = tf.contrib.layers.fully_connected(pi_net, 1 * self.action_dim,
                                                                   activation_fn=None,
                                                                   weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True),
                                                                   # weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                                   weights_regularizer=None,
                                                                   # tf.contrib.layers.l2_regularizer(0.001),
                                                                   biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True))
                                                                    # biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        # No activation on sigma: later capped to [-20, 1]
        action_prediction_log_sigma = tf.contrib.layers.fully_connected(pi_net, 1 * self.action_dim,
                                                                    activation_fn=None,
                                                                    weights_initializer=tf.random_uniform_initializer(0,3e-3),
                                                                    weights_regularizer=None,
                                                                    # tf.contrib.layers.l2_regularizer(0.001),
                                                                    biases_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3))

        # reshape output
        pi_mean = tf.reshape(pi_mean, [-1, self.action_dim])
        action_prediction_log_sigma = tf.reshape(action_prediction_log_sigma, [-1, self.action_dim])

        # scale mean to env. action domain
        # pi_mean = tf.multiply(pi_mean, self.action_max)

        # exp. sigma
        action_prediction_log_sigma = tf.clip_by_value(action_prediction_log_sigma, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=pi_mean, scale_diag=tf.exp(action_prediction_log_sigma))

        ##### TODO: Not sure about this part

        pi_sample_action = dist.sample()
        if not self.reparameterize:
            pi_sample_action = tf.stop_gradient(pi_sample_action)

        pi_log_pi = dist.log_prob(pi_sample_action)
        pi_log_sigma = action_prediction_log_sigma

        # squash correction
        pi_mean = tf.tanh(pi_mean)
        pi_sample_action = tf.tanh(pi_sample_action)

        pi_log_pi -= tf.reduce_sum(tf.log(1 - pi_sample_action ** 2 + EPS), axis=1) # sac
        # pi_log_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1) # spinning up

        pi_sample_action = tf.multiply(pi_sample_action, self.action_max)
        pi_mean = tf.multiply(pi_mean, self.action_max)

        return pi_sample_action, pi_log_pi, pi_mean, pi_log_sigma, dist

    def train_pi(self, state_batch):

        # pi_mean_batch, pi_sample_action_batch, pi_log_pi_batch = self.sess.run([self.pi_mean, self.pi_sample_action, self.pi_log_pi], feed_dict={
        #     self.pi_inputs: state_batch,
        #     self.pi_phase: True
        # })
        #
        # qf_pi = self.sess.run(self.q_val, feed_dict={
        #     self.q_inputs: state_batch,
        #     self.q_action: pi_sample_action_batch,
        #     self.q_phase: True
        # })
        #
        # # logp_pi, q1_pi
        # # args [inputs, actions, phase]

        return self.sess.run(self.pi_optimize, feed_dict={
            self.pi_inputs: state_batch,
            self.pi_phase: True,
            self.q_inputs: state_batch,
            self.q_phase: True
        })

    def train_qf_vf(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        batch_size = np.shape(state_batch)[0]

        reward_batch = np.reshape(reward_batch, (batch_size, 1))
        gamma_batch = np.reshape(gamma_batch, (batch_size, 1))

        v_target = self.sess.run(self.target_v_val, feed_dict={
            self.target_v_inputs: next_state_batch,
            self.target_v_phase: True
        })

        # TODO: Stop tf gradients?
        q_backup = reward_batch + gamma_batch * v_target

        pi_log_pi, q_pi = self.sess.run([self.pi_log_pi, self.q_pi], feed_dict={
            self.pi_inputs: state_batch,
            self.pi_phase: True,
            self.q_inputs: state_batch,
            self.q_phase: True
        })

        # TODO: Stop tf gradients?
        v_backup = q_pi - np.reshape(self.entropy_scale * pi_log_pi, (batch_size, 1))

        # step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, train_pi_op, train_value_op]
        return self.sess.run([self.q_val, self.qf_vf_optimize], feed_dict={
            self.q_inputs: state_batch,
            self.v_inputs: state_batch,
            self.q_action: action_batch,
            self.q_backup: q_backup,
            self.v_backup: v_backup,
            self.q_phase: True,
            self.v_phase: True
        })

    def predict_q(self, state, action, phase):

        return self.sess.run(self.q_val, feed_dict={
            self.q_inputs: state,
            self.q_action: action,
            self.q_phase: phase
        })

    def predict_v_target(self, state):

        return self.sess.run(self.target_v_val, feed_dict={
            self.target_v_inputs: state,
            self.target_v_phase: True
        })

    def predict_action(self, state):

        pi_mean = self.sess.run(self.pi_mean, feed_dict={
                self.pi_inputs: state,
                self.pi_phase: True
            })

        return pi_mean

    # Should return n actions
    def sample_action(self, state):

        pi_sample_action = self.sess.run(self.pi_sample_action, feed_dict={
                self.pi_inputs: state,
                self.pi_phase: True
            })

        return pi_sample_action

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_val, feed_dict={self.q_inputs: np.expand_dims(state, 0),
                                                                          self.q_action: np.expand_dims([action], 0),
                                                                          self.q_phase: False})

    def getPolicyFunction(self, state):

        dist = self.sess.run(self.pi_dist, feed_dict={
            self.pi_inputs: state,
            self.pi_phase: False
        })

        # return lambda action: np.sum(alpha * np.multiply(np.sqrt(1.0 / (2 * np.pi * np.square(sigma))), np.exp(-np.square(action - mean) / (2.0 * np.square(sigma)))))
        return lambda action: dist.prob(action)

