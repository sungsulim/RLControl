import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import itertools


class CriticNetwork(BaseNetwork):

    def __init__(self, sess, input_norm, layer_dim, state_dim, state_min, state_max, action_dim, action_min, action_max, learning_rate, tau, norm_type):
        super(CriticNetwork, self).__init__(sess, state_dim, action_dim, learning_rate, tau)

        self.l1 = layer_dim[0]
        self.l2 = layer_dim[1]

        self.state_min = state_min
        self.state_max = state_max

        self.action_min = action_min
        self.action_max = action_max

        self.input_norm = input_norm
        self.norm_type = norm_type

        # Critic network
        self.inputs, self.phase, self.action, self.outputs = self.build_network(scope_name = 'critic')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic') # tf.trainable_variables()[num_actor_vars:]

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_outputs = self.build_network(scope_name = 'target_critic')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic') # tf.trainable_variables()[len(self.net_params) + num_actor_vars:]
        
        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Op for periodically updating target network with online network weights
        self.update_target_net_params  = [tf.assign_add(self.target_net_params[idx], \
                                            self.tau * (self.net_params[idx] - self.target_net_params[idx])) for idx in range(len(self.target_net_params))]

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
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()

        # Define loss and optimization Op
        with tf.control_dependencies(self.batchnorm_ops):
            self.loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.outputs))
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the critic w.r.t. the action
        self.action_grads = tf.gradients(self.outputs, self.action)
        self.action_grads_target = tf.gradients(self.target_outputs, self.target_action)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):

            inputs = tf.placeholder(tf.float32, shape=(None,self.state_dim))
            phase = tf.placeholder(tf.bool)
            action = tf.placeholder(tf.float32, [None, self.action_dim])

            # normalize state inputs if using "input_norm" or "layer" or "batch"
            if self.norm_type is not 'none':
                inputs = self.input_norm.normalize(inputs)

            outputs = self.network(inputs, action, phase)

        return inputs, phase, action, outputs

    def network(self, inputs, action, phase):
        # 1st fc
        net = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=None,
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        net = self.apply_norm(net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # 2nd fc
        net = tf.contrib.layers.fully_connected(tf.concat([net, action], 1), self.l2, activation_fn=None,
                                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True),
                                                # tf.truncated_normal_initializer(),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                    factor=1.0, mode="FAN_IN", uniform=True))

        net = self.apply_norm(net, activation_fn=tf.nn.relu, phase=phase, layer_num=2)

        outputs = tf.contrib.layers.fully_connected(net, 1, activation_fn=None,
                                                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return outputs

    def predict(self, *args):
        # args  (inputs, action, phase)    
        inputs = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: phase
        })

    def predict_target(self, *args):
        # args  (inputs, action, phase)
        inputs = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.target_outputs, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action,
            self.target_phase: phase
        })

    def gradient_ascent(self, state, action_init, gd_alpha, gd_max_steps, gd_stop, is_training):
        
        action = np.copy(action_init)

        ascent_count = 0
        update_flag = np.ones([state.shape[0], self.action_dim])

        while np.any(update_flag > 0) and ascent_count < gd_max_steps:
            action_old = np.copy(action)

            gradients = self.action_gradients(state, action, is_training)[0]
            action += update_flag * gd_alpha * gradients
            action = np.clip(action, self.action_min, self.action_max)

            # stop if action diff. is small
            stop_idx = [idx for idx in range(len(action)) if np.mean(np.abs(action_old[idx] - action[idx])/self.action_max) <= gd_stop]
            update_flag[stop_idx] = 0
            # print(update_flag)

            ascent_count += 1
        return action

    def gradient_ascent_target(self, state, action_init, gd_alpha, gd_max_steps, gd_stop, is_training):
        
        action = np.copy(action_init)

        ascent_count = 0
        update_flag = np.ones([state.shape[0], self.action_dim])

        while np.any(update_flag > 0) and ascent_count < gd_max_steps:
            action_old = np.copy(action)

            gradients = self.action_gradients_target(state, action, is_training)[0]
            action += update_flag * gd_alpha * gradients
            action = np.clip(action, self.action_min, self.action_max)

            # stop if action diff. is small
            stop_idx = [idx for idx in range(len(action)) if np.mean(np.abs(action_old[idx] - action[idx])/self.action_max) <= gd_stop]
            update_flag[stop_idx] = 0

            ascent_count += 1
        return action

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

    def train(self, *args):
        # args (inputs, action, predicted_q_value, phase)
        return self.sess.run([self.outputs, self.optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.predicted_q_value: args[2],
            self.phase: True
        })
        
    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    # For OMNISCIENT
    def get_argmax_q(self, inputs, gd_alpha, gd_max_steps, gd_stop, mode):
        # mode == 0 : return argmax Q
        # mode == 1 : return argmax Q (target)


        # Later find a way to compute state representation beforehand
        #state_representation = self.sess.run(self.)@@@@@@@

        # print(inputs)
        # print(self.sess.run(tf.shape(self.argmax_stacked_state), feed_dict={
        #     self.argmax_input_state: inputs
        #     }))
        # print(self.sess.run(self.outputs, feed_dict={
        #     self.inputs: np.expand_dims(inputs,0),
        #     self.action: [[-1.0, -1.0]],
        #     self.phase: False
        #     }))

        # Stack states
        stacked_states = []
        stacked_states.extend(itertools.repeat(inputs, self.NUM_ACTION_SEGMENTS**self.action_dim))

        # original network
        if mode == 0:
            q_values = self.sess.run(self.outputs, feed_dict={
                self.inputs: stacked_states,
                self.action: self.searchable_actions,
                self.phase: False
            })
        # target network
        elif mode == 1:
            q_values = self.sess.run(self.target_outputs, feed_dict={
                self.target_inputs: stacked_states,
                self.target_action: self.searchable_actions,
                self.target_phase: False
            })

        # argmax index
        max_idx = np.argmax(q_values)
        return self.searchable_actions[max_idx]

    def print_variables(self, variable_list):
        variable_names = [v.name for v in variable_list]
        values = self.sess.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.outputs, {self.inputs: np.expand_dims(state, 0), 
                                            self.action: np.expand_dims(action, 0), 
                                            self.phase: False})

