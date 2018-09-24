import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np


class NAF_Network(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(NAF_Network, self).__init__(sess, config, config.learning_rate)

        self.l1_dim = config.l1_dim
        self.l2_dim = config.l2_dim

        self.input_norm = input_norm

        # naf specific params
        self.noise_scale = config.noise_scale

        # original network
        self.inputs, self.phase, self.action, self.q_val, self.max_q, self.best_action, self.Lmat_columns = self.build_network(scope_name="naf")
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='naf')

        # target network
        self.target_inputs, self.target_phase, self.target_action, self.target_q_val, self.target_max_q, self.target_best_action, self.target_Lmat_columns = self.build_network(
            scope_name="target_naf")
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_naf')

        # Batchnorm Ops and Vars
        if self.norm_type == 'batch':

            self.batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='naf/batchnorm')
            self.target_batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_naf/batchnorm')

            self.batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='naf/batchnorm')
            self.target_batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='target_naf/batchnorm')

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

            self.target_q_input = tf.placeholder(tf.float32, [None])
            self.loss = tf.reduce_sum(tf.square(self.target_q_input - tf.squeeze(self.q_val)))
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


        # Op for init. target network with original network weights
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx])
                                       for idx in range(len(self.target_net_params))]

        # Op for periodically updating target network with original network weights
        self.update_target_net_params = [tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx]))
                                         for idx in range(len(self.target_net_params))]

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            phase = tf.placeholder(tf.bool)
            action = tf.placeholder(tf.float32, shape=(None, self.action_dim))

            # normalize state inputs if using "input_norm" or "layer" or "batch"
            if self.norm_type is not 'none':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            q_val, max_q, best_action, Lmat_columns = self.network(inputs, action, phase)

        return inputs, phase, action, q_val, max_q, best_action, Lmat_columns

    def network(self, inputs, action, phase):

        state_hidden1 = tf.contrib.layers.fully_connected(inputs, self.l1_dim, activation_fn=None)

        # net, activation_fn, phase, layer_num):
        state_hidden1_norm = self.apply_norm(state_hidden1, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # action branch
        action_hidden2 = tf.contrib.layers.fully_connected(state_hidden1_norm, self.l2_dim, activation_fn=None)
        action_hidden2_norm = self.apply_norm(action_hidden2, activation_fn=tf.nn.relu, phase=phase, layer_num=2)
        best_action = tf.contrib.layers.fully_connected(action_hidden2_norm, self.action_dim, activation_fn=tf.nn.tanh) * self.action_max
        # should be within range
        # best_action = tf.clip_by_value(best_action, self.action_min, self.action_max)

        # value branch
        value_hidden = tf.contrib.layers.fully_connected(state_hidden1_norm, self.l2_dim, activation_fn=None)
        value_hidden_norm = self.apply_norm(value_hidden, activation_fn=tf.nn.relu, phase=phase, layer_num=3)
        value = tf.contrib.layers.fully_connected(value_hidden_norm, 1, activation_fn=None, weights_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))

        # Lmat branch
        act_mu_diff = action - best_action

        # Lmat_flattened = tf.contrib.layers.fully_connected(state_hidden1_norm, (1+self.action_dim)*self.action_dim/2, activation_fn = None)
        # Lmat_diag = [tf.exp(tf.contrib.layers.fully_connected(state_hidden1_norm, 1, activation_fn = None)) for _ in range(self.action_dim)]

        Lmat_diag = [tf.exp(tf.clip_by_value(tf.contrib.layers.fully_connected(state_hidden1_norm, 1, activation_fn=None), -5.0, 5.0))
                     for _ in range(self.action_dim)]  # clipping to prevent blowup
        Lmat_nondiag = [tf.contrib.layers.fully_connected(state_hidden1_norm, k - 1, activation_fn=None) for k in
                        range(self.action_dim, 1, -1)]

        # in Lmat_columns, if actdim = 1, first part is empty
        Lmat_columns = [tf.concat((Lmat_diag[id], Lmat_nondiag[id]), axis=1)
                        for id in range(len(Lmat_nondiag))] + [Lmat_diag[-1]]

        act_mu_diff_Lmat_prod = [tf.reduce_sum(tf.slice(act_mu_diff, [0, cid], [-1, -1]) * Lmat_columns[cid], axis=1, keepdims=True)
                                 for cid in range(len(Lmat_columns))]

        # prod_tensor should be dim: batchsize * action_dim
        prod_tensor = tf.concat(act_mu_diff_Lmat_prod, axis=1)

        adv_value = -0.5 * tf.reduce_sum(prod_tensor * prod_tensor, axis=1, keepdims=True)
        q_value = value + adv_value
        max_q = value

        return q_value, max_q, best_action, Lmat_columns

    def train(self, *args):

        inputs = args[0]
        action = args[1]
        target_q = args[2]

        self.sess.run(self.optimize, feed_dict={self.inputs: inputs,
                                                self.action: action,
                                                self.target_q_input: target_q,
                                                self.phase: True})

    def predict_max_q_target(self, *args):
        inputs = args[0]

        return self.sess.run(self.target_max_q, feed_dict={
            self.target_inputs: inputs,
            self.phase: True  # TODO: Double check why it should be True
        })

    def predict_action(self, *args):

        inputs = args[0]
        best_action = self.sess.run(self.best_action, feed_dict={self.inputs: inputs,
                                                                 self.phase: False})
        return best_action

    # return one sampled action
    def sample_action(self, *args):

        inputs = args[0]
        greedy_action = args[1]

        Lmat_columns = self.sess.run(self.Lmat_columns, feed_dict={self.inputs: inputs,
                                                                   self.phase: False})

        # compute covariance matrix
        Lmat = np.zeros((self.action_dim, self.action_dim))
        for i in range(self.action_dim):
            Lmat[i:, i] = np.squeeze(Lmat_columns[i])
        try:
            covmat = self.noise_scale * np.linalg.pinv(Lmat.dot(Lmat.T))
        except:
            print('Lmat', Lmat)
            print('\nLmat^2', Lmat.dot(Lmat.T))
            print('\n')
            print("error occurred!")
            exit()

        sampled_action = np.random.multivariate_normal(greedy_action.reshape(-1), covmat)
        sampled_action = np.clip(sampled_action, self.action_min, self.action_max)

        return sampled_action, covmat

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_val, feed_dict={self.inputs: np.expand_dims(state, 0),
                                                                   self.action: np.expand_dims(action, 0),
                                                                   self.phase: False})

    def getPolicyFunction(self, m, v):

        mean = np.squeeze(m)
        var = np.squeeze(v)

        assert(var >= 0)
        return lambda action: np.multiply(np.sqrt(1.0 / (2 * np.pi * var)), np.exp(-np.square(action - mean) / (2.0 * var)))

