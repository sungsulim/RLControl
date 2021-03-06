import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
from scipy.stats import norm
from utils.boundedvar_gaussian_mixture import BoundedVarGaussianMixture


class QTOPTNetwork(BaseNetwork):

    def __init__(self, sess, input_norm, config):
        super(QTOPTNetwork, self).__init__(sess, config, config.qnet_lr)

        self.rng = np.random.RandomState(config.random_seed)

        self.l1 = config.qnet_l1_dim
        self.l2 = config.qnet_l2_dim

        self.num_iter = config.num_iter
        self.num_samples = config.num_samples
        self.top_m = config.top_m

        self.input_norm = input_norm

        self.num_modal = config.num_modal


        # Q network
        self.inputs, self.phase, self.action, self.outputs = self.build_network(scope_name='qnet')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qnet')

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_outputs = self.build_network(scope_name='target_qnet')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_qnet')
        
        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = [tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx])) for idx in range(len(self.target_net_params))]

        # Op for init. target network with identical parameter as the original network
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx]) for idx in range(len(self.target_net_params))]

        if self.norm_type == 'batch':
            # Batchnorm Ops and Vars
            self.batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qnet/batchnorm')
            self.target_batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_qnet/batchnorm')

            self.batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='qnet/batchnorm')
            self.target_batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='target_qnet/batchnorm')

            self.update_target_batchnorm_params = [tf.assign(self.target_batchnorm_vars[idx], self.batchnorm_vars[idx])
                                                   for idx in range(len(self.target_batchnorm_vars))
                                                   if self.target_batchnorm_vars[idx].name.endswith('moving_mean:0')
                                                   or self.target_batchnorm_vars[idx].name.endswith('moving_variance:0')]

        else:
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()

        # Define loss and optimization Op
        with tf.control_dependencies(self.batchnorm_ops):
            self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.outputs))
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):

            inputs = tf.placeholder(tf.float32, shape=(None,self.state_dim))
            phase = tf.placeholder(tf.bool)
            action = tf.placeholder(tf.float32, [None, self.action_dim])

            # normalize state inputs if using "input_norm" or "layer" or "batch"
            if self.norm_type != 'none':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            outputs = self.network(inputs, action, phase)

        return inputs, phase, action, outputs

    def network(self, inputs, action, phase):
        # 1st fc
        net = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=None,
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        net = self.apply_norm(net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # 2nd fc
        net = tf.contrib.layers.fully_connected(tf.concat([net, action], 1), self.l2, activation_fn=None,
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        net = self.apply_norm(net, activation_fn=tf.nn.relu, phase=phase, layer_num=2)

        outputs = tf.contrib.layers.fully_connected(net, 1, activation_fn=None,
                                                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return outputs

    def predict_q(self, *args):
        # args  (inputs, action, phase)    
        inputs = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: phase
        })

    def predict_q_target(self, *args):
        # args  (inputs, action, phase)
        inputs = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.target_outputs, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_phase: phase
        })


    def iterate_cem_multidim(self, state_batch):
        batch_size = len(state_batch)

        action_samples_batch = None
        gmm_batch = None

        ## stack states
        stacked_state_batch = np.array(
            [np.tile(state, (self.num_samples, 1)) for state in state_batch])  # batch_size x self.num_samples x state_dim
        stacked_state_batch = np.reshape(stacked_state_batch, (batch_size * self.num_samples, self.state_dim))

        for i in range(self.num_iter):

            # sample batch_num x num_samples: (n,64)
            if action_samples_batch is None and gmm_batch is None:
                action_samples_batch = self.rng.uniform(self.action_min, self.action_max, size=(batch_size, self.num_samples, self.action_dim))

            else:
                # single gaussian (deprecated)
                # action_samples_batch = np.array([self.rng.multivariate_normal(mean, std, size=self.num_samples) for (mean, std) in mean_std_arr])

                # gaussian mixture
                # action_samples_batch = np.array([self.rng.])
                action_samples_batch = np.array([gmm.sample(n_samples=self.num_samples)[0] for gmm in gmm_batch])

            # evaluate Q-val

            ## reshape action samples
            action_samples_batch_reshaped = np.reshape(action_samples_batch, (batch_size * self.num_samples, self.action_dim))

            q_val = self.predict_q(stacked_state_batch, action_samples_batch_reshaped, True)
            q_val = np.reshape(q_val, (batch_size, self.num_samples))

            # select top-m
            selected_idxs = list(map(lambda x: x.argsort()[::-1][:self.top_m], q_val))

            selected_action_samples_batch = np.array(
                [action_samples_for_state[selected_idx_for_state] for action_samples_for_state, selected_idx_for_state
                 in zip(action_samples_batch, selected_idxs)])

            # fit gaussian mixture
            gmm_batch = [BoundedVarGaussianMixture(n_components=self.num_modal, random_state=self.rng, covariance_type="diag", tol=1e-2).fit(action_samples) for action_samples in selected_action_samples_batch]

        return gmm_batch

    def predict_action(self, state_batch):

        gmm_batch = self.iterate_cem_multidim(state_batch)
        final_action_mean_batch = np.array([gmm.means_[np.argmax(gmm.weights_)] for gmm in gmm_batch])
        return final_action_mean_batch

    def sample_action(self, state_batch):

        gmm_batch = self.iterate_cem_multidim(state_batch)
        final_action_samples_batch = np.array([gmm.sample(n_samples=1)[0] for gmm in gmm_batch])
        final_action_mean_batch = np.array([gmm.means_[np.argmax(gmm.weights_)] for gmm in gmm_batch])

        weight_mean_var_arr = [(gmm.weights_, gmm.means_, gmm.covariances_) for gmm in gmm_batch]

        return final_action_samples_batch, final_action_mean_batch, weight_mean_var_arr

    def train(self, *args):
        # args (inputs, action, predicted_q_value, phase)
        return self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.predicted_q_value: args[2],
            self.phase: True
        })

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])


    def print_variables(self, variable_list):
        variable_names = [v.name for v in variable_list]
        values = self.sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.outputs, {self.inputs: np.expand_dims(state, 0), 
                                            self.action: np.expand_dims([action], 0),
                                            self.phase: False})

    def getPolicyFunction(self, weight_mean_var_arr):

        weight, mean, var = weight_mean_var_arr
        mean = np.squeeze(mean, axis=1)
        var = np.squeeze(var, axis=1)

        if len(weight) == len(mean) == len(var) == 2:
            return lambda action: np.sum(weight * np.multiply(np.sqrt(1.0 / (2 * np.pi * var)), np.exp(
                -np.square(action - mean) / (2.0 * var))))
        else:
            return lambda action: np.multiply(
                np.sqrt(1.0 / (2 * np.pi * var[0])),
                np.exp(-np.square(action - mean[0]) / (2.0 * var[0])))

