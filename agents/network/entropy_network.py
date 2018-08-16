import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import itertools
import matplotlib.pyplot as plt


class EntropyNetwork(BaseNetwork):

    def __init__(self, sess, input_norm, layer_dim, state_dim, state_min, state_max, action_dim, action_min, action_max, learning_rate, tau, inference, norm_type):
        super(EntropyNetwork, self).__init__(sess, state_dim, action_dim, learning_rate, tau)

        self.l1 = layer_dim[0]
        self.l2 = layer_dim[1]
        self.action_dim = action_dim

        self.state_min = state_min
        self.state_max = state_max

        self.action_min = action_min
        self.action_max = action_max

        self.input_norm = input_norm
        self.norm_type = norm_type

        self.inference = inference

        self.actionRange = action_max - action_min

        # Critic network
        # notice here outputs is -1.0 * qvalue, not qvalue directly! f_outputs is outputs - entropy and is only for bunddle entropy optimization
        self.inputs, self.phase, self.action, self.outputs, self.f_outputs = self.build_network(scope_name = 'critic')
        self.qvalue = -1.0 * self.outputs
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic') # tf.trainable_variables()[num_actor_vars:]
        self.wz_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic/zub')
        # project Wz to nonnegative after training
        self.project = [p.assign(tf.maximum(p, 0)) for p in self.wz_params]

        # Target network
        # notice here target_outputs is -1.0 * qvalue, not qvalue directly!
        self.target_inputs, self.target_phase, self.target_action, self.target_outputs, self.f_target_outputs = self.build_network(scope_name = 'target_critic')
        self.target_qvalue = -1.0 * self.target_outputs
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic') # tf.trainable_variables()[len(self.net_params) + num_actor_vars:]

        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])


        # Op for periodically updating target network with online network weights
        self.update_target_net_params  = [tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx])) for idx in range(len(self.target_net_params))]


        if self.norm_type == 'batch':
            # Batchnorm Ops and Vars
            self.batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/batchnorm')
            self.target_batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_critic/batchnorm')

            self.batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='critic/batchnorm')
            self.target_batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='target_critic/batchnorm')

            self.update_target_batchnorm_params = [tf.assign(self.target_batchnorm_vars[idx], \
                                                self.batchnorm_vars[idx]) for idx in range(len(self.target_batchnorm_vars)) \
                                                if self.target_batchnorm_vars[idx].name.endswith('moving_mean:0') or self.target_batchnorm_vars[idx].name.endswith('moving_variance:0')]


        else:
            assert (self.norm_type == 'none' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()

        # Define loss and optimization Op
        with tf.control_dependencies(self.batchnorm_ops):
            self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.qvalue))
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the critic w.r.t. the action
        self.action_grads = tf.gradients(self.outputs, self.action)
        self.action_grads_target = tf.gradients(self.target_outputs, self.target_action)

        # Get the gradient of the network part (without entropy regularization) w.r.t. the action
        self.f_action_grads = tf.gradients(self.f_outputs, self.action)
        self.f_action_grads_target = tf.gradients(self.f_target_outputs, self.target_action)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):

            inputs = tf.placeholder(tf.float32, shape=(None,self.state_dim))
            phase = tf.placeholder(tf.bool)

            action = tf.placeholder(tf.float32, [None, self.action_dim])

            if self.norm_type == 'input_norm' or self.norm_type == 'layer':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            if self.norm_type == 'layer':
                f_outputs = self.layer_norm_network(inputs, action, phase)

            elif self.norm_type == 'batch':
                assert (self.input_norm is None)
                f_outputs = self.batch_norm_network(inputs, action, phase)

            else:
                assert (self.norm_type == 'none' or self.norm_type == 'input_norm')
                f_outputs = self.no_norm_network(inputs, action, phase)

            tmpA = tf.clip_by_value(self.rmapAction(action), 0.0001, 0.9999)
            outputs = f_outputs + tf.reduce_sum(tmpA * tf.log(tmpA) + (1 - tmpA) * tf.log(1 - tmpA), reduction_indices = 1, keep_dims=True)

        return inputs, phase, action, outputs, f_outputs

    def layer_norm_network(self, inputs, action, phase):
        # TODO: make codes of the PICNN more compact. It can now only learn two layers, make it learnable in the case of multiple layers.

        u1 = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        u1 = tf.contrib.layers.layer_norm(u1, center=True, scale=True, activation_fn=tf.nn.relu)

        u2 = tf.contrib.layers.fully_connected(u1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))


        wub0 = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        yub0 = tf.contrib.layers.fully_connected(inputs, self.action_dim, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        yub0 = tf.contrib.layers.fully_connected(action * yub0, self.l1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None)

        z1 = tf.contrib.layers.layer_norm(yub0 + wub0, center=True, scale=True, activation_fn=tf.nn.relu)

        wub1 = tf.contrib.layers.fully_connected(u1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        zub1 = tf.contrib.layers.fully_connected(u1, self.l1, activation_fn=tf.nn.relu, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        zub1 = tf.contrib.layers.fully_connected(z1 * zub1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.random_uniform_initializer(minval = 0, maxval = tf.sqrt(3.0 / self.l1), dtype=tf.float32),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None, scope = "zub_0")

        yub1 = tf.contrib.layers.fully_connected(u1, self.action_dim, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        yub1 = tf.contrib.layers.fully_connected(action * yub1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None)

        z2 = tf.contrib.layers.layer_norm(zub1 + yub1 + wub1, center=True, scale=True, activation_fn=tf.nn.relu)


        wub2 = tf.contrib.layers.fully_connected(u2, 1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        zub2 = tf.contrib.layers.fully_connected(u2, self.l2, activation_fn=tf.nn.relu, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        zub2 = tf.contrib.layers.fully_connected(z2 * zub2, 1, activation_fn=None, \
                                                weights_initializer=tf.random_uniform_initializer(minval = 0, maxval = tf.sqrt(3.0 / self.l2), dtype=tf.float32),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None, scope = "zub_1")

        yub2 = tf.contrib.layers.fully_connected(u2, self.action_dim, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        yub2 = tf.contrib.layers.fully_connected(action * yub2, 1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None)

        f_outputs = zub2 + yub2 + wub2

        return f_outputs

    def batch_norm_network(self, inputs, action, phase):
        # state input -> bn
        net = tf.contrib.layers.batch_norm(inputs, fused=True, center=True, scale=True, activation_fn=None,
                                           is_training=phase, scope='batchnorm_0')

        u1 = tf.contrib.layers.fully_connected(net, self.l1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        # bn -> relu
        u1 = tf.contrib.layers.batch_norm(u1,  fused=True, center=True, scale=True, activation_fn=tf.nn.relu, \
                                           is_training=phase, scope='batchnorm_1')


        u2 = tf.contrib.layers.fully_connected(u1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))


        wub0 = tf.contrib.layers.fully_connected(net, self.l1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        yub0 = tf.contrib.layers.fully_connected(net, self.action_dim, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        yub0 = tf.contrib.layers.fully_connected(action * yub0, self.l1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = False)

        z1 = tf.contrib.layers.batch_norm(yub0 + wub0, fused=True, center=True, scale=True, activation_fn=tf.nn.relu, \
                                           is_training=phase, scope='batchnorm_2')

        wub1 = tf.contrib.layers.fully_connected(u1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        zub1 = tf.contrib.layers.fully_connected(u1, self.l1, activation_fn=tf.nn.relu, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        zub1 = tf.contrib.layers.fully_connected(z1 * zub1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.random_uniform_initializer(minval = 0, maxval = tf.sqrt(3.0 / self.l1), dtype=tf.float32),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None, scope = "zub_0")

        yub1 = tf.contrib.layers.fully_connected(u1, self.action_dim, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        yub1 = tf.contrib.layers.fully_connected(action * yub1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None)

        # bn -> relu
        #z2 = tf.contrib.layers.batch_norm(zub1 + yub1 + wub1, fused=True, center=True, scale=True, activation_fn=tf.nn.relu, is_training=phase, scope='batchnorm_3')

        z2 = tf.nn.relu(zub1 + yub1 + wub1)


        wub2 = tf.contrib.layers.fully_connected(u2, 1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        zub2 = tf.contrib.layers.fully_connected(u2, self.l2, activation_fn=tf.nn.relu, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        zub2 = tf.contrib.layers.fully_connected(z2 * zub2, 1, activation_fn=None, \
                                                weights_initializer=tf.random_uniform_initializer(minval = 0, maxval = tf.sqrt(3.0 / self.l2), dtype=tf.float32),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None, scope = "zub_1")

        yub2 = tf.contrib.layers.fully_connected(u2, self.action_dim, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        yub2 = tf.contrib.layers.fully_connected(action * yub2, 1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None)

        f_outputs = zub2 + yub2 + wub2

        return f_outputs

    def no_norm_network(self, inputs, action, phase):

        u1 = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=tf.nn.relu, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))


        u2 = tf.contrib.layers.fully_connected(u1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))


        wub0 = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        yub0 = tf.contrib.layers.fully_connected(inputs, self.action_dim, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        yub0 = tf.contrib.layers.fully_connected(action * yub0, self.l1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None)

        z1 = tf.nn.relu(yub0 + wub0)

        wub1 = tf.contrib.layers.fully_connected(u1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        zub1 = tf.contrib.layers.fully_connected(u1, self.l1, activation_fn=tf.nn.relu, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        zub1 = tf.contrib.layers.fully_connected(z1 * zub1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.random_uniform_initializer(minval = 0, maxval = tf.sqrt(3.0 / self.l1), dtype=tf.float32),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None, scope = "zub_0")

        yub1 = tf.contrib.layers.fully_connected(u1, self.action_dim, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        yub1 = tf.contrib.layers.fully_connected(action * yub1, self.l2, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None)

        z2 = tf.nn.relu(zub1 + yub1 + wub1)


        wub2 = tf.contrib.layers.fully_connected(u2, 1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        zub2 = tf.contrib.layers.fully_connected(u2, self.l2, activation_fn=tf.nn.relu, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        zub2 = tf.contrib.layers.fully_connected(z2 * zub2, 1, activation_fn=None, \
                                                weights_initializer=tf.random_uniform_initializer(minval = 0, maxval = tf.sqrt(3.0 / self.l2), dtype=tf.float32),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None, scope = "zub_1")

        yub2 = tf.contrib.layers.fully_connected(u2, self.action_dim, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))
        yub2 = tf.contrib.layers.fully_connected(action * yub2, 1, activation_fn=None, \
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(), \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                                 biases_initializer = None)

        f_outputs = zub2 + yub2 + wub2

        return f_outputs


    def train(self, *args):
        # args (inputs, action, predicted_q_value, phase)
        res = self.sess.run([self.qvalue, self.optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.predicted_q_value: args[2],
            self.phase: True
        })
        self.sess.run(self.project)
        return res


    def predict(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.qvalue, feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.phase: args[2]
        })

    def predict_target(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.target_qvalue, feed_dict={
            self.target_inputs: args[0],
            self.target_action: args[1],
            self.target_phase: args[2]
        })

    # note this is for buddle entropy optimization and it returns f_outputs rather than -f_ouputs as in predict qvalues.
    def f_predict(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.f_outputs, feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.phase: args[2]
        })

    def f_predict_target(self, *args):
        # args  (inputs, action, phase)
        return self.sess.run(self.f_target_outputs, feed_dict={
            self.target_inputs: args[0],
            self.target_action: args[1],
            self.target_phase: args[2]
        })

    def alg_opt(self, state, action_init, inference_max_step):
        #print('param:', self.sess.run(self.net_params))
        if self.inference == 'bundle_entropy':
            return self.bundle_entropy(state, action_init, inference_max_step)
        elif self.inference == 'adam':
            return self.adam(state, action_init, inference_max_step)

    def bundle_entropy(self, state, action_init, inference_max_step):

        num = action_init.shape[0]

        #print('state:', state)
        #print('action', action_init)

        action = action_init

        G = [[] for i in range(num)]
        h = [[] for i in range(num)]
        actions = [[] for i in range(num)]
        lam = [None] * num

        finished = set([])
        for t in range(inference_max_step):
            fi = self.f_predict(state, self.mapAction(action), False)[:, 0]
            gi = self.actionRange * self.f_action_gradients(state, self.mapAction(action), False)[0]
            Gi = gi
            hi = fi - np.sum(gi * action, axis=1)
            for u in range(num):
                if u in finished:
                    continue

                G[u].append(Gi[u])
                h[u].append(hi[u])
                actions[u].append(np.copy(action[u]))

                prev_action = action[u].copy()
                #print('G:', G[u][0])
                if len(G[u]) > 1:
                    lam[u] = self.proj_newton_logistic(np.array(G[u]), np.array(h[u]), None)
                    action[u] = 1 / (1 + np.exp(np.array(G[u]).T.dot(lam[u])))
                    action[u] = np.clip(action[u], 0.03, 0.97)
                else:
                    lam[u] = np.array([1])
                    #print('G:',G[u][0])
                    action[u] = 1 / (1 + np.exp(G[u][0]))
                    #print('a:', action[u])
                    action[u] = np.clip(action[u], 0.03, 0.97)

                if max(abs((prev_action - action[u]))) < 1e-6:
                    finished.add(u)

                G[u] = [y for i, y in enumerate(G[u]) if lam[u][i] > 0]
                h[u] = [y for i, y in enumerate(h[u]) if lam[u][i] > 0]
                actions[u] = [y for i, y in enumerate(actions[u]) if lam[u][i] > 0]
                lam[u] = lam[u][lam[u] > 0]

            if len(finished) == num:
                #print(t)
                return self.mapAction(action)

        #print(t)
        return self.mapAction(action)

    def adam(self, obs, act, inference_max_step):
        b1 = 0.9
        b2 = 0.999
        lam = 0.5
        eps = 1e-8
        alpha = 0.01
        m = np.zeros_like(act)
        v = np.zeros_like(act)
        b1t, b2t = 1., 1.
        act_best, a_diff, f_best = [None] * 3
        #print("sample")
        for i in range(inference_max_step):
            f = -1 * self.predict(obs, act, False)
            g = self.action_gradients(obs, act, False)[0]
            if i == 0:
                act_best = act.copy()
                f_best = f.copy()
            else:
                prev_act_best = act_best.copy()
                I = (f < f_best)
                f_best[I] = f[I]
                # make it runnable when action is more than one dimension
                I = np.tile(I, self.action_dim)
                act_best[I] = act[I]
                a_diff_i = np.mean(np.linalg.norm(act_best - prev_act_best, axis = 1))
                a_diff = a_diff_i if a_diff is None \
                    else lam * a_diff + (1. - lam) * a_diff_i
                #print(a_diff_i, a_diff, np.sum(f))
                if a_diff < 1e-3 and i > 5:
                    #print(a_diff_i, a_diff, np.sum(f_best))
                    #print('  + Adam took {} iterations'.format(i))
                    return act_best

            m = b1 * m + (1. - b1) * g
            v = b2 * v + (1. - b2) * (g * g)
            b1t *= b1
            b2t *= b2
            mhat = m / (1. - b1t)

            act -= alpha * mhat / (np.sqrt(v) + eps)
            act = np.clip(act, self.action_min, self.action_max)

        #print(a_diff_i, a_diff, np.sum(f_best))
        #print('  + Warning: Adam did not converge.')
        return act_best

    def alg_opt_target(self, state, action_init, inference_max_step):
        if self.inference == 'bundle_entropy':
            return self.bundle_entropy_target(state, action_init, inference_max_step)
        elif self.inference == 'adam':
            return self.adam_target(state, action_init, inference_max_step)

    def bundle_entropy_target(self, state, action_init, inference_max_step):

        num = action_init.shape[0]

        action = action_init
        #print(action)

        G = [[] for i in range(num)]
        h = [[] for i in range(num)]
        actions = [[] for i in range(num)]
        lam = [None] * num

        finished = set([])
        for t in range(inference_max_step):
            fi = self.f_predict_target(state, self.mapAction(action), False)[:, 0]
            gi = self.actionRange * self.f_action_gradients_target(state, self.mapAction(action), False)[0]
            Gi = gi
            hi = fi - np.sum(gi * action, axis=1)
            for u in range(num):
                if u in finished:
                    continue

                G[u].append(Gi[u])
                h[u].append(hi[u])
                actions[u].append(np.copy(action[u]))

                prev_action = action[u].copy()
                if len(G[u]) > 1:
                    lam[u] = self.proj_newton_logistic(np.array(G[u]), np.array(h[u]), None)
                    action[u] = 1 / (1 + np.exp(np.array(G[u]).T.dot(lam[u])))
                    action[u] = np.clip(action[u], 0.03, 0.97)

                else:
                    lam[u] = np.array([1])
                    action[u] = 1 / (1 + np.exp(G[u][0]))
                    action[u] = np.clip(action[u], 0.03, 0.97)

                if max(abs((prev_action - action[u]))) < 1e-6:
                    finished.add(u)

                G[u] = [y for i, y in enumerate(G[u]) if lam[u][i] > 0]
                h[u] = [y for i, y in enumerate(h[u]) if lam[u][i] > 0]
                actions[u] = [y for i, y in enumerate(actions[u]) if lam[u][i] > 0]
                lam[u] = lam[u][lam[u] > 0]

            if len(finished) == num:
                #print(t)
                return self.mapAction(action)

        #print(t)
        return self.mapAction(action)

    def adam_target(self, obs, act, inference_max_step):
        b1 = 0.9
        b2 = 0.999
        lam = 0.5
        eps = 1e-8
        alpha = 0.01
        m = np.zeros_like(act)
        v = np.zeros_like(act)
        #print("target")
        b1t, b2t = 1., 1.
        act_best, a_diff, f_best = [None] * 3
        for i in range(inference_max_step):
            f = -1 * self.predict_target(obs, act, False)
            g = self.action_gradients_target(obs, act, False)[0]
            if i == 0:
                act_best = act.copy()
                f_best = f.copy()
            else:
                prev_act_best = act_best.copy()
                I = (f < f_best)
                f_best[I] = f[I]
                # make it runnable when action is more than one dimension
                I = np.tile(I, self.action_dim)
                act_best[I] = act[I]
                a_diff_i = np.mean(np.linalg.norm(act_best - prev_act_best, axis = 1))
                a_diff = a_diff_i if a_diff is None \
                    else lam * a_diff + (1. - lam) * a_diff_i
                #print(a_diff_i, a_diff, np.sum(f))
                if a_diff < 1e-3 and i > 5:
                    #print(a_diff_i, a_diff, np.sum(f_best))
                    #print('  + Adam took {} iterations'.format(i))
                    return act_best

            m = b1 * m + (1. - b1) * g
            v = b2 * v + (1. - b2) * (g * g)
            b1t *= b1
            b2t *= b2
            mhat = m / (1. - b1t)

            act -= alpha * mhat / (np.sqrt(v) + eps)
            act = np.clip(act, self.action_min, self.action_max)

        #print(a_diff_i, a_diff, np.sum(f_best))
        #print('  + Warning: Adam did not converge.')
        return act_best

    def action_gradients(self, inputs, action, is_training):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: is_training
        })

    def action_gradients_target(self, inputs, action, is_training):
        return self.sess.run(self.action_grads_target, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_phase: is_training
        })

    def f_action_gradients(self, inputs, action, is_training):
        return self.sess.run(self.f_action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: is_training
        })

    def f_action_gradients_target(self, inputs, action, is_training):
        return self.sess.run(self.f_action_grads_target, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_phase: is_training
        })

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def logexp1p(self, x):
        """ Numerically stable log(1+exp(x))"""
        y = np.zeros_like(x)
        I = x > 1
        y[I] = np.log1p(np.exp(-x[I])) + x[I]
        y[~I] = np.log1p(np.exp(x[~I]))
        return y

    def proj_newton_logistic(self, A, b, lam0=None, line_search=True):
        """ minimize_{lam>=0, sum(lam)=1} -(A*1 + b)^T*lam + sum(log(1+exp(A^T*lam)))"""
        n = A.shape[0]
        c = np.sum(A,axis=1) + b
        e = np.ones(n)

        eps = 1e-12
        ALPHA = 1e-5
        BETA = 0.5

        if lam0 is None:
            lam = np.ones(n)/n
        else:
            lam = lam0.copy()

        for i in range(20):
            # compute gradient and Hessian of objective
            ATlam = A.T.dot(lam)
            z = 1/(1+np.exp(-ATlam))
            f = -c.dot(lam) + np.sum(self.logexp1p(ATlam))
            g = -c + A.dot(z)
            H = (A*(z*(1-z))).dot(A.T)

            # change of variables
            i = np.argmax(lam)
            y = lam.copy()
            y[i] = 1
            e[i] = 0

            g0 = g - e*g[i]
            H0 = H - np.outer(e,H[:,i]) - np.outer(H[:,i],e) + H[i,i]*np.outer(e,e)

            # compute bound set and Hessian of free set
            I = (y <= eps) & (g0 > 0)
            I[i] = True
            if np.linalg.norm(g0[~I]) < 1e-10:
                return lam
            d = np.zeros(n)
            H0_ = H0[~I,:][:,~I]
            try:
                d[~I] = np.linalg.solve(H0_, -g0[~I])
            except:
                # print('\n=== A\n\n', A)
                # print('\n=== H\n\n', H)
                # print('\n=== H0\n\n', H0)
                # print('\n=== H0_\n\n', H0_)
                # print('\n=== z\n\n', z)
                # print('\n=== iter: {}\n\n'.format(i))
                break
            # line search
            t = min(1. / np.max(abs(d)), 1.)
            for _ in range(10):
                y_n = np.maximum(y + t*d,0)
                y_n[i] = 1
                lam_n = y_n.copy()
                lam_n[i] = 1.-e.dot(y_n)
                if lam_n[i] >= 0:
                    if line_search:
                        fn = -c.dot(lam_n) + np.sum(self.logexp1p(A.T.dot(lam_n)))
                        if fn < f + t*ALPHA*d.dot(g0):
                            break
                    else:
                        break
                if max(t * abs(d)) < 1e-10:
                    return lam_n
                t *= BETA

            e[i] = 1.
            lam = lam_n.copy()
        return lam

    def print_variables(self, variable_list):
        variable_names = [v.name for v in variable_list]
        values = self.sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)


    # Buggy
    def getQFunction(self, state):
        raise NotImplementedError

    # Buggy
    def plotFunc(self,func, x_min, x_max, resolution=1e5, display_title='', save_title='', linewidth=2.0, grid=True, show=True, equal_aspect=False):
        raise NotImplementedError

    # for bundle_entropy, mapping [0, 1] to real range of actions
    def mapAction(self, act):
        return self.actionRange * act + self.action_min

    # for bundle_entropy, mapping real range of actions to [0, 1]
    def rmapAction(self, act):
        return  (act - self.action_min) / self.actionRange
