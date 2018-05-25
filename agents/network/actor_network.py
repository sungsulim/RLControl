import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np

class ActorNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, layer_dim, state_dim, state_min, state_max, action_dim, action_min, action_max, learning_rate, tau, norm_type):
        super(ActorNetwork, self).__init__(sess, state_dim, action_dim, learning_rate, tau)

        self.l1 = layer_dim[0]
        self.l2 = layer_dim[1]

        self.state_min = state_min
        self.state_max = state_max

        self.action_min = action_min
        self.action_max = action_max

        self.input_norm = input_norm
        self.norm_type = norm_type

        # Actor network
        self.inputs, self.phase, self.outputs, self.scaled_outputs = self.build_network(scope_name = 'actor')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor') # tf.trainable_variables('actor')

        # Target network
        self.target_inputs, self.target_phase, self.target_outputs, self.target_scaled_outputs = self.build_network(scope_name = 'target_actor')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor') # tf.trainable_variables('target_actor')

        # Op for periodically updating target network with online network weights
        self.update_target_net_params  = [tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx])) for idx in range(len(self.target_net_params))]


        # Temporary placeholder action gradient
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])
        self.actor_gradients = tf.gradients(self.outputs, self.net_params, -self.action_gradients)

        self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)


        if self.norm_type == 'batch':
            # Batchnorm Ops and Vars
            self.batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/batchnorm')
            self.target_batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_actor/batchnorm')

            self.batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='actor/batchnorm')
            self.target_batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='target_actor/batchnorm')

            self.update_target_batchnorm_params = [tf.assign(self.target_batchnorm_vars[idx], \
                                                self.batchnorm_vars[idx]) for idx in range(len(self.target_batchnorm_vars)) \
                                                if self.target_batchnorm_vars[idx].name.endswith('moving_mean:0') or self.target_batchnorm_vars[idx].name.endswith('moving_variance:0')]

        else:
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()


        # Optimization Op
        with tf.control_dependencies(self.batchnorm_ops):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self.actor_gradients, self.net_params))


    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            phase = tf.placeholder(tf.bool)

            # normalize state inputs if using "input_norm" or "layer" or "batch"
            if self.norm_type == 'input_norm' or self.norm_type == 'layer':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            if self.norm_type == 'layer':
                outputs = self.layer_norm_network(inputs, phase)
            elif self.norm_type == 'batch':
                assert (self.input_norm is None)
                outputs = self.batch_norm_network(inputs, phase)
            elif self.norm_type == 'none' or self.norm_type == 'input_norm':
                assert (self.input_norm is None)
                outputs = self.no_norm_network(inputs, phase)
            else:
                raise Exception('WRONG NORM TYPE!!')



            scaled_outputs = tf.multiply(outputs, self.action_max) # Scale output to [-action_bound, action_bound]
            
        return inputs, phase, outputs, scaled_outputs


    def layer_norm_network(self, inputs, phase):

        # 1st fc
        net = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=None, \
                                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                            weights_regularizer=None, #]tf.contrib.layers.l2_regularizer(0.001), \
                                            biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        # layer norm
        net = tf.contrib.layers.layer_norm(net, center=True, scale=True, activation_fn=tf.nn.relu)

        # 2nd fc
        net = tf.contrib.layers.fully_connected(net, self.l2, activation_fn=None, \
                                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                            weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                            biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        # layer norm
        net = tf.contrib.layers.layer_norm(net, center=True, scale=True, activation_fn=tf.nn.relu)

        # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
        outputs = tf.contrib.layers.fully_connected(net, self.action_dim, activation_fn=tf.tanh, \
                                            weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
                                            weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                            biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        return outputs

    def batch_norm_network(self, inputs, phase):
        # state input -> bn
        net = tf.contrib.layers.batch_norm(inputs, fused=True, center=True, scale=True, activation_fn=None, \
                                            is_training=phase, scope='batchnorm_0')

        # 1st fc
        net = tf.contrib.layers.fully_connected(net, self.l1, activation_fn=None, \
                                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                            weights_regularizer=None, #]tf.contrib.layers.l2_regularizer(0.001), \
                                            biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        # bn -> relu
        net = tf.contrib.layers.batch_norm(net, fused=True, center=True, scale=True, activation_fn=tf.nn.relu, \
                                            is_training=phase, scope='batchnorm_1')


        # 2nd fc
        net = tf.contrib.layers.fully_connected(net, self.l2, activation_fn=None, \
                                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                            weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                            biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        # bn -> relu
        net = tf.contrib.layers.batch_norm(net, fused=True, center=True, scale=True, activation_fn=tf.nn.relu, \
                                            is_training=phase, scope='batchnorm_2')

        # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
        outputs = tf.contrib.layers.fully_connected(net, self.action_dim, activation_fn=tf.tanh, \
                                            weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
                                            weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                            biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return outputs


    def no_norm_network(self, inputs, phase):
        
        # 1st fc
        net = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=tf.nn.relu, \
                                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                            weights_regularizer=None, #]tf.contrib.layers.l2_regularizer(0.001), \
                                            biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        # 2nd fc
        net = tf.contrib.layers.fully_connected(net, self.l2, activation_fn=tf.nn.relu, \
                                            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                            weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                            biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
        outputs = tf.contrib.layers.fully_connected(net, self.action_dim, activation_fn=tf.tanh, \
                                            weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
                                            weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                            biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return outputs


    def train(self, *args):

        self.sess.run(self.optimize, feed_dict={
            self.inputs: args[0],
            self.action_gradients: args[1],
            self.phase: True
        })

        return

    def predict(self, *args):
        # args [inputs, phase]
        inputs = args[0]
        phase = args[1]

        return self.sess.run(self.scaled_outputs, feed_dict={
            self.inputs: inputs,
            self.phase: phase
        })

    def predict_target(self, *args):

        inputs = args[0]
        phase = args[1]

        return self.sess.run(self.target_scaled_outputs, feed_dict={
            self.target_inputs: inputs,
            self.target_phase: phase,
        })

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def print_variables(self, variable_list):
        variable_names = [v.name for v in variable_list]
        values = self.sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)
