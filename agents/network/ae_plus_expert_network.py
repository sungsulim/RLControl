import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np


class ActorExpert_Plus_Expert_Network(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(ActorExpert_Plus_Expert_Network, self).__init__(sess, config, config.expert_lr)

        self.rng = np.random.RandomState(config.random_seed)

        self.l1_dim = config.l1_dim
        self.l2_dim = config.l2_dim

        self.input_norm = input_norm


        # GA config during CEM
        self.gd_alpha = config.gd_alpha
        self.gd_max_steps = config.gd_max_steps
        self.gd_stop = config.gd_stop

        self.sigma_scale = 1.0  # config.sigma_scale

        self.use_uniform_sampling = False
        if config.use_uniform_sampling == "True":
            self.use_uniform_sampling = True
            self.uniform_sampling_ratio = 0.2  # config.uniform_sampling_ratio

        self.use_better_q_gd = False
        if config.use_better_q_gd == "True":
            self.use_better_q_gd = True
            self.better_q_gd_alpha = 1e-2  # config.better_q_gd_alpha
            self.better_q_gd_max_steps = 10  # config.better_q_gd_max_steps
            self.better_q_gd_stop = 1e-3  # config.better_q_gd_stop

        # Removed from config
        # "better_q_gd_alpha": [1e-2],
        # "better_q_gd_max_steps": [10],
        # "better_q_gd_stop": [1e-3],

        # currently not used
        self.use_policy_gd = False
        # if config.use_policy_gd == "True":
        #     self.use_policy_gd = True
        #     self.policy_gd_alpha = config.policy_gd_alpha
        #     self.policy_gd_max_steps = config.policy_gd_max_steps
        #     self.policy_gd_stop = config.policy_gd_stop

        # Removed from config
        # "use_policy_gd": ["False"],
        # "policy_gd_alpha": [1e-1],
        # "policy_gd_max_steps": [50],
        # "policy_gd_stop": [1e-3],

        self.equal_modal_selection = False
        # if config.equal_modal_selection == "True":
        #     self.equal_modal_selection = True

        # original network
        self.inputs, self.phase, self.action, self.q_prediction = self.build_network(scope_name='ae_expert')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ae_expert')

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_q_prediction = self.build_network(scope_name='target_ae_expert')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_ae_expert')

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = [
            tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx]))
            for idx in range(len(self.target_net_params))]

        # Op for init. target network with identical parameter as the original network
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx]) for idx in range(len(self.target_net_params))]

        # TODO: Currently doesn't support batchnorm
        if self.norm_type == 'batch':
            raise NotImplementedError

        else:
            assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
            self.batchnorm_ops = [tf.no_op()]
            self.update_target_batchnorm_params = tf.no_op()

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        self.actions = tf.placeholder(tf.float32, [None, self.action_dim])

        # Optimization Op
        with tf.control_dependencies(self.batchnorm_ops):

            # Expert Update
            self.expert_loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.q_prediction))
            self.expert_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.expert_loss)

        # Get the gradient of the expert w.r.t. the action
        self.action_grads = tf.gradients(self.q_prediction, self.action)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            phase = tf.placeholder(tf.bool)
            action = tf.placeholder(tf.float32, shape=(None, self.action_dim))

            # normalize inputs
            if self.norm_type != 'none':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            q_prediction = self.network(inputs, action, phase)

        return inputs, phase, action, q_prediction

    def network(self, inputs, action, phase):
        # shared net
        net = tf.contrib.layers.fully_connected(inputs, self.l1_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        net = self.apply_norm(net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # Q branch
        q_net = tf.contrib.layers.fully_connected(tf.concat([net, action], 1), self.l2_dim,
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

    def q_gradient_ascent(self, state, action_init, is_training, is_better_q_gd=False):

        if is_better_q_gd:
            gd_alpha = self.better_q_gd_alpha
            gd_max_steps = self.better_q_gd_max_steps
            gd_stop = self.better_q_gd_stop

        else:
            gd_alpha = self.gd_alpha
            gd_max_steps = self.gd_max_steps
            gd_stop = self.gd_stop

        action = np.copy(action_init)

        ascent_count = 0
        update_flag = np.ones([state.shape[0], self.action_dim])  # batch_size * action_dim

        while np.any(update_flag > 0) and ascent_count < gd_max_steps:
            action_old = np.copy(action)

            gradients = self.q_action_gradients(state, action, is_training)[0]
            action += update_flag * gd_alpha * gradients
            action = np.clip(action, self.action_min, self.action_max)

            # stop if action diff. is small
            stop_idx = [idx for idx in range(len(action)) if
                        np.mean(np.abs(action_old[idx] - action[idx]) / self.action_max) <= gd_stop]
            update_flag[stop_idx] = 0
            # print(update_flag)

            ascent_count += 1

        # print('ascent count:', ascent_count)
        return action

    def q_action_gradients(self, inputs, action, is_training):

        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: is_training
        })

    def train_expert(self, *args):
        # args (inputs, action, predicted_q_value)
        return self.sess.run([self.q_prediction, self.expert_optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.predicted_q_value: args[2],
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

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_prediction, feed_dict={self.inputs: np.expand_dims(state, 0),
                                                                          self.action: np.expand_dims([action], 0),
                                                                          self.phase: False})