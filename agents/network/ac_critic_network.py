import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import environments.environments


class AC_Critic_Network(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(AC_Critic_Network, self).__init__(sess, config, config.critic_lr)

        self.rng = np.random.RandomState(config.random_seed)

        self.critic_layer1_dim = config.l1_dim
        self.critic_layer2_dim = config.l2_dim

        self.input_norm = input_norm

        # ac specific params
        self.actor_update = config.actor_update

        # original network
        self.inputs, self.phase, self.action, self.q_prediction = self.build_network(scope_name='ac_critic')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ac_critic')

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_q_prediction = self.build_network(scope_name='target_ac_critic')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_ac_critic')

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

        self.q_target = tf.placeholder(tf.float32, [None, 1])

        # Optimization Op
        with tf.control_dependencies(self.batchnorm_ops):

            # critic Update
            self.critic_loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_prediction))
            self.critic_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim), name="network_input_state")
            phase = tf.placeholder(tf.bool, name="network_input_phase")
            action = tf.placeholder(tf.float32, shape=(None, self.action_dim), name="network_input_action")

            # normalize inputs
            if self.norm_type != 'none':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            q_prediction = self.network(inputs, action, phase)

        return inputs, phase, action, q_prediction

    def network(self, inputs, action, phase):
        # shared net
        q_net = tf.contrib.layers.fully_connected(inputs, self.critic_layer1_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        q_net = self.apply_norm(q_net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # Q branch
        q_net = tf.contrib.layers.fully_connected(tf.concat([q_net, action], 1), self.critic_layer2_dim,
                                                  activation_fn=None,
                                                  weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True),
                                                  # tf.truncated_normal_initializer(), \
                                                  weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                  biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      factor=1.0, mode="FAN_IN", uniform=True))

        q_net = self.apply_norm(q_net, activation_fn=tf.nn.relu, phase=phase, layer_num=3)
        q_prediction = tf.contrib.layers.fully_connected(q_net, 1, activation_fn=None,
                                                         weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                         weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                         biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return q_prediction

    def train_critic(self, *args):
        # args (inputs, action, q_target)
        return self.sess.run([self.q_prediction, self.critic_optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.q_target: args[2],
            self.phase: True
        })

    def predict_q(self, *args):
        # args  (inputs, action, phase)
        inputs = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.q_prediction, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: phase
        })

    def predict_q_target(self, *args):
        # args  (inputs, action, phase)
        inputs = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.target_q_prediction, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_phase: phase
        })

    def predict_true_q(self, *args):
        # args  (inputs, action, phase)
        inputs = args[0]
        action = args[1]
        phase = args[2]
        env_name = args[3]

        return [getattr(environments.environments, env_name).reward_func(a[0]) for a in action]

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_prediction, feed_dict={self.inputs: np.expand_dims(state, 0),
                                                                          self.action: np.expand_dims([action], 0),
                                                                          self.phase: False})