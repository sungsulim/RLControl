import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np


class OptimalQ_Network(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(OptimalQ_Network, self).__init__(sess, config, config.learning_rate)

        self.rng = np.random.RandomState(config.random_seed)

        # precompute action pairs
        self.discretization = config.discretization
        self.discretized_action_pairs = self.compute_discretized_action_pairs()

        # precompute stacked action pairs
        self.compute_stacked_action_pairs()

        self.l1_dim = config.l1_dim
        self.l2_dim = config.l2_dim

        self.input_norm = input_norm

        # original network
        self.inputs, self.phase, self.action, self.q_val = self.build_network(scope_name="optimal_q")
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='optimal_q')

        # target network
        self.target_inputs, self.target_phase, self.target_action, self.target_q_val = self.build_network(scope_name="target_optimal_q")
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_optimal_q')

        # Batchnorm Ops and Vars
        if self.norm_type == 'batch':

            self.batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimal_q/batchnorm')
            self.target_batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_optimal_q/batchnorm')

            self.batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='optimal_q/batchnorm')
            self.target_batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='target_optimal_q/batchnorm')

            self.update_target_batchnorm_params = [tf.assign(self.target_batchnorm_vars[idx],
                                                             self.batchnorm_vars[idx]) for idx in
                                                   range(len(self.target_batchnorm_vars))
                                                   if self.target_batchnorm_vars[idx].name.endswith('moving_mean:0')
                                                   or self.target_batchnorm_vars[idx].name.endswith('moving_variance:0')]

        else:
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()

        # define loss and update operation
        with tf.control_dependencies(self.batchnorm_ops):

            self.target_q_input = tf.placeholder(tf.float32, [None, 1])
            # TODO: check dimension between self.target_q_input, self.q_val
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q_input, self.q_val))
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Op for init. target network with original network weights
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx])
                                       for idx in range(len(self.target_net_params))]

        # Op for periodically updating target network with original network weights
        self.update_target_net_params = [tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx]))
                                         for idx in range(len(self.target_net_params))]

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):

            inputs = tf.placeholder(tf.float32, shape=(None,self.state_dim))
            phase = tf.placeholder(tf.bool)
            action = tf.placeholder(tf.float32, [None, self.action_dim])

            # normalize state inputs if using "input_norm" or "layer" or "batch"
            if self.norm_type is not 'none':
                # inputs = self.input_norm.normalize(inputs)
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            q_prediction = self.network(inputs, action, phase)

        return inputs, phase, action, q_prediction

    def network(self, inputs, action, phase):
        # 1st fc
        net = tf.contrib.layers.fully_connected(inputs, self.l1_dim, activation_fn=None,
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        net = self.apply_norm(net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # 2nd fc
        net = tf.contrib.layers.fully_connected(tf.concat([net, action], 1), self.l2_dim, activation_fn=None,
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        net = self.apply_norm(net, activation_fn=tf.nn.relu, phase=phase, layer_num=2)

        q_prediction = tf.contrib.layers.fully_connected(net, 1, activation_fn=None,
                                                         weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                         weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                         biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return q_prediction

    def train(self, state_batch, action_batch, target_batch):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: state_batch,
            self.action: action_batch,
            self.target_q_input: target_batch,
            self.phase: True
        })

        return

    def get_max_action(self, state_batch, use_target, is_train):

        batch_size = np.shape(state_batch)[0]
        num_action_pairs = np.shape(self.discretized_action_pairs)[0]

        stacked_state_batch = np.array([np.tile(state, (num_action_pairs, 1)) for state in state_batch])

        # (batch size * num_action_pairs) x state_dim
        stacked_state_batch = np.reshape(stacked_state_batch, (batch_size * num_action_pairs, self.state_dim))

        # for efficiency precomputed
        # (batch size * num_action_pairs) x action_dim
        # stacked_action_batch = np.tile(self.discretized_action_pairs, (batch_size, 1))
        if batch_size == 32:
            stacked_action_batch = self.stacked_action_batch32
        elif batch_size == 1:
            stacked_action_batch = self.stacked_action_batch1
        else:
            raise ValueError("Invalid batch_size")

        if use_target:
            q_val_batch = self.sess.run(self.target_q_val, feed_dict={
                self.target_inputs: stacked_state_batch,
                self.target_action: stacked_action_batch,
                self.target_phase: is_train
        })
        else:
            q_val_batch = self.sess.run(self.q_val, feed_dict={
                self.inputs: stacked_state_batch,
                self.action: stacked_action_batch,
                self.phase: is_train
        })


        # find maxQ, argmaxQ
        q_val_batch_reshaped = np.reshape(q_val_batch, (batch_size, num_action_pairs))
        maxQ_batch = np.max(q_val_batch_reshaped, axis=1)
        argmaxQ_idx_batch = np.argmax(q_val_batch_reshaped, axis=1)
        argmaxQ_batch = self.discretized_action_pairs[argmaxQ_idx_batch]

        return maxQ_batch, argmaxQ_batch

    def compute_discretized_action_pairs(self):
        small_eps = 1e-10

        # assuming all action dimensions have the same range as the first one
        discretized_actions = np.arange(self.action_min[0], self.action_max[0] + small_eps, self.discretization)

        stacked_actions = np.tile(discretized_actions, (self.action_dim, 1))

        # (action_dim x num. actions in dim_1, num. actions in dim_2, ..., num. actions in dim_n)
        discretized_action_pairs = np.meshgrid(*stacked_actions)

        discretized_action_pairs = [a.flatten() for a in discretized_action_pairs]

        # (num actions in one dim ^n x action_dim)
        discretized_action_pairs = np.array(list(zip(*discretized_action_pairs)))

        return discretized_action_pairs

    def compute_stacked_action_pairs(self):

        self.stacked_action_batch32 = np.tile(self.discretized_action_pairs, (32, 1))
        self.stacked_action_batch1 = np.tile(self.discretized_action_pairs, (1, 1))


    # def predict_action(self, *args):
    #
    #     inputs = args[0]
    #     best_action = self.sess.run(self.best_action, feed_dict={self.inputs: inputs,
    #                                                              self.phase: False})
    #     return best_action

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_val, feed_dict={self.inputs: np.expand_dims(state, 0),
                                                                   self.action: np.expand_dims(action, 0),
                                                                   self.phase: False})
