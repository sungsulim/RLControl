import tensorflow as tf
import tensorflow.contrib.slim as slim

from agents.network.base_network import BaseNetwork
import numpy as np


class WireFitting_Network(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(WireFitting_Network, self).__init__(sess, config, config.learning_rate)

        self.rng = np.random.RandomState(config.random_seed)

        self.l1_dim = config.l1_dim
        self.l2_dim = config.l2_dim

        self.input_norm = input_norm

        # wirefitting specific params
        self.app_points = config.app_points

        # not used basically
        # self.decay_rate = config.lr_decay_rate
        # self.decay_step = config.lr_decay_step

        # constants
        self.smooth_eps = 0.00001
        # self.adv_k = 1.0
        # self.n = 1
        self.dtype = tf.float32  # tf.float64

        # original network
        self.state_input, self.phase, self.interim_actions, self.interim_q_values, self.max_q, self.best_action = self.build_network("wf")
        self.action_input, self.q_val = self.create_interpolation("interplt", self.interim_actions, self.interim_q_values, self.max_q)

        # target network
        self.target_state_input, self.target_phase, self.target_interim_actions, self.target_interim_qvalues, self.target_max_q, self.target_best_action = self.build_network("target_wf")
        self.target_action_input, self.target_q_val = self.create_interpolation("target_interplt", self.target_interim_actions, self.target_interim_qvalues, self.target_max_q)

        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wf') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='interplt')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_wf') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_interplt')

        # Batchnorm Ops and Vars
        # TODO: Implement batchnorm ops

        # loss, update operation
        self.target_q_input, self.interplt_loss = self.define_loss("losses")

        # define optimization
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.interplt_loss)

        # Op for init. target network with original network weights
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx]) for idx in range(len(self.target_net_params))]

        # Op for periodically updating target network with original network weights
        self.update_target_net_params = [
            tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx]))
            for idx in range(len(self.target_net_params))]

    def define_loss(self, scopename):
        with tf.variable_scope(scopename):
            qtargets = tf.placeholder(self.dtype, [None])
            interplt_loss = tf.losses.mean_squared_error(qtargets, self.q_val)
        return qtargets, interplt_loss

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            state_input = tf.placeholder(self.dtype, shape=(None, self.state_dim))
            phase = tf.placeholder(tf.bool)
            # action = tf.placeholder(self.dtype, shape=(None, self.action_dim))

            # normalize state inputs if using "input_norm" or "layer" or "batch"
            if self.norm_type != 'none':
                state_input = tf.clip_by_value(self.input_norm.normalize(state_input), self.state_min, self.state_max)

            state_hidden1 = slim.fully_connected(state_input, self.l1_dim, activation_fn=None)
            state_hidden1_norm = self.apply_norm(state_hidden1, activation_fn=tf.nn.relu, phase=phase, layer_num=1)
            state_hidden2 = slim.fully_connected(state_hidden1_norm, self.l2_dim, activation_fn=None)
            state_hidden2_norm = self.apply_norm(state_hidden2, activation_fn=tf.nn.relu, phase=phase, layer_num=2)

            # state_hidden2_val = slim.fully_connected(state_hidden1, n_hidden1, activation_fn = tf.nn.relu)
            '''
            state_hidden1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(state_input, n_hidden1, activation_fn = None), center=True, scale=True, is_training=self.is_training))
            state_hidden2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(state_hidden1, n_hidden2, activation_fn = None), center=True, scale=True, is_training=self.is_training))
            '''
            w_init = tf.random_uniform_initializer(minval=-1., maxval=1.)
            interim_actions = slim.fully_connected(state_hidden2_norm, self.app_points * self.action_dim,
                                                activation_fn=tf.nn.tanh, weights_initializer=w_init) * self.action_max[0]
            # print 'interim action shape is :: ', interim_actions.shape
            # w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            interim_q_values = slim.fully_connected(state_hidden2_norm, self.app_points, activation_fn=None,
                                                   weights_initializer=w_init)
            # print 'interim q values shape is :: ', interim_q_values.shape
            max_q_val = tf.reduce_max(interim_q_values, axis=1)

            # get best action
            maxind = tf.argmax(interim_q_values, axis=1)
            rowinds = tf.range(0, tf.cast(tf.shape(state_input)[0], tf.int64), 1)
            maxind_nd = tf.concat([tf.reshape(rowinds, [-1, 1]), tf.reshape(maxind, [-1, 1])], axis=1)
            # print 'max id shape is :: ', maxind_nd.shape

            best_action = tf.gather_nd(tf.reshape(interim_actions, [-1, self .app_points, self.action_dim]), maxind_nd)

        return state_input, phase, interim_actions, interim_q_values, max_q_val, best_action

    def create_interpolation(self, scope_name, interim_actions, interim_qvalues, max_q):
        with tf.variable_scope(scope_name):
            action_input = tf.placeholder(self.dtype, [None, self.action_dim])
            tiled_action_input = tf.tile(action_input, [1, self.app_points])
            reshaped_action_input = tf.reshape(tiled_action_input, [-1, self.app_points, self.action_dim])
            # print 'reshaped tiled action shape is :: ', reshaped_action_input.shape
            reshaped_action_output = tf.reshape(interim_actions, [-1, self.app_points, self.action_dim])
            # distance is b * n mat, n is number of points to do interpolation
            act_distance = tf.reduce_sum(tf.square(reshaped_action_input - reshaped_action_output), axis=2)
            w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            smooth_c = tf.nn.sigmoid(tf.get_variable("smooth_c", [1, self.app_points], initializer=w_init, dtype=self.dtype))
            q_distance = smooth_c*(tf.reshape(max_q, [-1, 1]) - interim_qvalues)
            distance = act_distance + q_distance + self.smooth_eps
            # distance = tf.add(distance, self.smooth_eps)
            weight = 1.0/distance
            # weight sum is a matrix b*1, b is batch size
            weightsum = tf.reduce_sum(weight, axis=1, keep_dims=True)
            weight_final = weight/weightsum
            q_val = tf.reduce_sum(tf.multiply(weight_final, interim_qvalues), axis=1)
        return action_input, q_val

    def train(self, *args):
        # args (inputs, action, predicted_q_value, phase)
        return self.sess.run(self.optimize, feed_dict={
            self.state_input: args[0],
            self.action_input: args[1],
            self.target_q_input: args[2],
            self.phase: True
        })

    def predict_action(self, *args):

        inputs = args[0]
        best_action, actions = self.sess.run([self.best_action, self.interim_actions], feed_dict={
            self.state_input: inputs.reshape(-1, self.state_dim), self.phase: False
        })

        return best_action.reshape(-1), actions.reshape(self.app_points, self.action_dim)

    # TODO: This isn't used, and hasn't been verified
    def predict_q_target(self, *args):
        return self.sess.run(self.target_q_val, feed_dict={
            self.state_input: args[0],
            self.action_input: args[1],

        })

    def predict_max_q_target(self, *args):
        inputs = args[0]

        return self.sess.run(self.target_max_q, feed_dict={
            self.target_state_input: inputs,
            self.target_phase: False
        })

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params])  #, self.update_target_batchnorm_params])

    def getQFunction(self, state):
        # raise NotImplementedError
        return lambda action: self.sess.run(self.q_val, feed_dict={self.state_input: np.expand_dims(state, 0),
                                                                          self.action_input: np.expand_dims(action, 0),
                                                                          self.phase: False})
