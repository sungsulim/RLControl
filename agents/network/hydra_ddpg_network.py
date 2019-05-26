import tensorflow as tf
import numpy as np
from agents.network.base_network import BaseNetwork


class HydraDDPGNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(HydraDDPGNetwork, self).__init__(sess, config, [config.actor_lr, config.critic_lr])

        # self.l1 = config.actor_l1_dim
        # self.l2 = config.actor_l2_dim

        self.shared_layer_dim = config.shared_l1_dim
        self.actor_layer_dim = config.actor_l2_dim
        self.critic_layer_dim = config.critic_l2_dim

        self.input_norm = input_norm

        # Actor network
        self.inputs, self.phase, self.action, self.output_action, self.output_scaled_action, self.output_qval = self.build_network(scope_name='hydra_ddpg')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hydra_ddpg')

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_output_action, self.target_output_scaled_action, self.target_output_qval = self.build_network(scope_name='target_hydra_ddpg')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_hydra_ddpg')

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = [tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx])) for idx in range(len(self.target_net_params))]

        # Op for init. target network with identical parameter as the original network
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx]) for idx in range(len(self.target_net_params))]

        # For Actor Network
        # Temporary placeholder action gradient
        self.action_gradients_placeholder = tf.placeholder(tf.float32, [None, self.action_dim])
        self.actor_gradients = tf.gradients(self.output_action, self.net_params, -self.action_gradients_placeholder)

        # For Critic Network
        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)

        if self.norm_type == 'batch':
            # Batchnorm Ops and Vars
            self.batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hydra_ddpg/batchnorm')
            self.target_batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_hydra_ddpg/batchnorm')

            self.batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='hydra_ddpg/batchnorm')
            self.target_batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='target_hydra_ddpg/batchnorm')

            self.update_target_batchnorm_params = [tf.assign(self.target_batchnorm_vars[idx],
                                                   self.batchnorm_vars[idx]) for idx in range(len(self.target_batchnorm_vars))
                                                   if self.target_batchnorm_vars[idx].name.endswith('moving_mean:0')
                                                   or self.target_batchnorm_vars[idx].name.endswith('moving_variance:0')]

        else:
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()

        # Optimization Op
        with tf.control_dependencies(self.batchnorm_ops):

            # Actor Update
            self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate[0]).apply_gradients(zip(self.actor_gradients, self.net_params))

            # Critic Update
            self.critic_loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.output_qval))
            self.critic_optimize = tf.train.AdamOptimizer(self.learning_rate[1]).minimize(self.critic_loss)

        # Get the gradient of the critic w.r.t. the action
        self.action_grads = tf.gradients(self.output_qval, self.action)
        self.action_grads_target = tf.gradients(self.target_output_qval, self.target_action)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):

            inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            phase = tf.placeholder(tf.bool)
            action = tf.placeholder(tf.float32, [None, self.action_dim])

            # normalize state inputs if using "input_norm" or "layer" or "batch"
            if self.norm_type != 'none':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            output_action, output_qval = self.network(inputs, action, phase)

            # TODO: Currently assumes actions are symmetric around 0.
            output_scale_action = tf.multiply(output_action, self.action_max)  # Scale output to [-action_bound, action_bound]


        return inputs, phase, action, output_action, output_scale_action, output_qval

    def network(self, inputs, action, phase):

        # 1st fc
        shared_net = tf.contrib.layers.fully_connected(inputs, self.shared_layer_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        shared_net = self.apply_norm(shared_net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # Actor Branch
        # 2nd fc
        actor_net = tf.contrib.layers.fully_connected(shared_net, self.actor_layer_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=None,  # tf.contrib.layers.l2_regularizer(0.001),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        actor_net = self.apply_norm(actor_net, activation_fn=tf.nn.relu, phase=phase, layer_num=2)

        # Final layer weight are initialized to Uniform[-3e-3, 3e-3]
        output_action = tf.contrib.layers.fully_connected(actor_net, self.action_dim, activation_fn=tf.tanh,
                                                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                    weights_regularizer=None, # tf.contrib.layers.l2_regularizer(0.001), \
                                                    biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        # Critic Branch
        # 2nd fc
        critic_net = tf.contrib.layers.fully_connected(tf.concat([shared_net, action], 1), self.critic_layer_dim, activation_fn=None,
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        critic_net = self.apply_norm(critic_net, activation_fn=tf.nn.relu, phase=phase, layer_num=3)

        output_qval = tf.contrib.layers.fully_connected(critic_net, 1, activation_fn=None,
                                                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return output_action, output_qval

    def train_actor(self, *args):
        # args (inputs, action_gradients)

        return self.sess.run(self.actor_optimize, feed_dict={
            self.inputs: args[0],
            self.action_gradients_placeholder: args[1],
            self.phase: True
        })

    def train_critic(self, *args):
        # args (inputs, action, predicted_q_value, phase)
        return self.sess.run([self.output_qval, self.critic_optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.predicted_q_value: args[2],
            self.phase: True
        })

    def predict_action(self, *args):

        # args [inputs, phase]
        inputs = args[0]
        phase = args[1]

        return self.sess.run(self.output_scaled_action, feed_dict={
            self.inputs: inputs,
            self.phase: phase
        })

    def predict_action_target(self, *args):

        inputs = args[0]
        phase = args[1]

        return self.sess.run(self.target_output_scaled_action, feed_dict={
            self.target_inputs: inputs,
            self.target_phase: phase,
        })

    def predict_qval(self, *args):
        # args  (inputs, action, phase)
        inputs = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.output_qval, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: phase
        })

    def predict_qval_target(self, *args):
        # args  (inputs, action, phase)
        inputs = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.target_output_qval, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_phase: phase
        })

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

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

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

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.output_qval, {self.inputs: np.expand_dims(state, 0),
                                            self.action: np.expand_dims(action, 0),
                                            self.phase: False})

