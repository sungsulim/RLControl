import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np


class ActorExpert_Plus_Network(BaseNetwork):
    def __init__(self, sess, input_norm, config):
        super(ActorExpert_Plus_Network, self).__init__(sess, config, [config.actor_lr, config.expert_lr])

        self.shared_layer_dim = config.shared_l1_dim
        self.actor_layer_dim = config.actor_l2_dim
        self.expert_layer_dim = config.expert_l2_dim

        self.input_norm = input_norm

        # ae_plus specific params
        self.rho = config.rho
        self.num_samples = config.num_samples
        self.num_modal = config.num_modal
        self.gd_alpha = config.gd_alpha
        self.gd_max_steps = config.gd_max_steps
        self.gd_stop = config.gd_stop
        self.actor_output_dim = self.num_modal * (1 + 2 * self.action_dim)

        self.policy_gd_alpha = config.policy_gd_alpha
        self.policy_gd_max_steps = config.policy_gd_max_steps
        self.policy_gd_stop = config.policy_gd_stop
        self.use_policy_gd = False
        if config.use_policy_gd == "True":
            self.use_policy_gd = True


        # original network
        self.inputs, self.phase, self.action, self.action_prediction_mean, self.action_prediction_sigma, self.action_prediction_alpha, self.q_prediction = self.build_network(
            scope_name='actorexpert')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actorexpert')

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_action_prediction_mean, self.target_action_prediction_sigma, self.target_action_prediction_alpha, self.target_q_prediction = self.build_network(
            scope_name='target_actorexpert')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actorexpert')

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
            self.expert_optimize = tf.train.AdamOptimizer(self.learning_rate[1]).minimize(self.expert_loss)

            # Actor update
            self.actor_loss = self.get_lossfunc(self.action_prediction_alpha, self.action_prediction_sigma,
                                                self.action_prediction_mean, self.actions)
            self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.actor_loss)

        # Get the gradient of the expert w.r.t. the action
        self.action_grads = tf.gradients(self.q_prediction, self.action)

        # # Get the gradient of the policy w.r.t. the action
        self.temp_alpha, self.temp_mean, self.temp_sigma, self.temp_action, self.policy_func = self.get_policyfunc()
        self.policy_action_grads = tf.gradients(self.policy_func, self.temp_action)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            phase = tf.placeholder(tf.bool)
            action = tf.placeholder(tf.float32, shape=(None, self.action_dim))

            # normalize inputs
            if self.norm_type is not 'none':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            action_prediction_mean, action_prediction_sigma, action_prediction_alpha, q_prediction = self.network(
                inputs, action, phase)

        return inputs, phase, action, action_prediction_mean, action_prediction_sigma, action_prediction_alpha, q_prediction

    def network(self, inputs, action, phase):
        # shared net
        shared_net = tf.contrib.layers.fully_connected(inputs, self.shared_layer_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        shared_net = self.apply_norm(shared_net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # action branch
        action_net = tf.contrib.layers.fully_connected(shared_net, self.actor_layer_dim, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       # tf.truncated_normal_initializer(),
                                                       weights_regularizer=None,
                                                       # tf.contrib.layers.l2_regularizer(0.001),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        action_net = self.apply_norm(action_net, activation_fn=tf.nn.relu, phase=phase, layer_num=2)

        action_prediction_mean = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim,
                                                                   activation_fn=tf.tanh,
                                                                   weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True),
                                                                   # tf.random_uniform_initializer(-3e-3, 3e-3),
                                                                   weights_regularizer=None,
                                                                   # tf.contrib.layers.l2_regularizer(0.001),
                                                                   biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                                       factor=1.0, mode="FAN_IN", uniform=True))
        # tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_sigma = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim,
                                                                    activation_fn=tf.tanh,
                                                                    weights_initializer=tf.random_uniform_initializer(0,
                                                                                                                      3e-3),
                                                                    weights_regularizer=None,
                                                                    # tf.contrib.layers.l2_regularizer(0.001),
                                                                    biases_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3))

        action_prediction_alpha = tf.contrib.layers.fully_connected(action_net, self.num_modal, activation_fn=tf.tanh,
                                                                    weights_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3),
                                                                    weights_regularizer=None,
                                                                    # tf.contrib.layers.l2_regularizer(0.001),
                                                                    biases_initializer=tf.random_uniform_initializer(
                                                                        -3e-3, 3e-3))

        # reshape output
        action_prediction_mean = tf.reshape(action_prediction_mean, [-1, self.num_modal, self.action_dim])
        action_prediction_sigma = tf.reshape(action_prediction_sigma, [-1, self.num_modal, self.action_dim])
        action_prediction_alpha = tf.reshape(action_prediction_alpha, [-1, self.num_modal, 1])

        # scale mean to env. action domain
        action_prediction_mean = tf.multiply(action_prediction_mean, self.action_max)

        # exp. sigma
        action_prediction_sigma = tf.exp(action_prediction_sigma)

        # mean: [None, num_modal, action_dim]  : [None, 1]
        # sigma: [None, num_modal, action_dim] : [None, 1]
        # alpha: [None, num_modal, 1]              : [None, 1]

        # compute softmax prob. of alpha
        max_alpha = tf.reduce_max(action_prediction_alpha, axis=1, keepdims=True)
        action_prediction_alpha = tf.subtract(action_prediction_alpha, max_alpha)
        action_prediction_alpha = tf.exp(action_prediction_alpha)

        normalize_alpha = tf.reciprocal(tf.reduce_sum(action_prediction_alpha, axis=1, keepdims=True))
        action_prediction_alpha = tf.multiply(normalize_alpha, action_prediction_alpha)

        # Q branch
        q_net = tf.contrib.layers.fully_connected(tf.concat([shared_net, action], 1), self.expert_layer_dim,
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

        return action_prediction_mean, action_prediction_sigma, action_prediction_alpha, q_prediction

    def tf_normal(self, y, mu, sigma):

        # y: batch x action_dim
        # mu: batch x num_modal x action_dim
        # sigma: batch x num_modal x action_dim

        # stacked y: batch x num_modal x action_dim
        stacked_y = tf.expand_dims(y, 1)
        stacked_y = tf.tile(stacked_y, [1, self.num_modal, 1])

        return tf.reduce_prod(
            tf.sqrt(1.0 / (2 * np.pi * tf.square(sigma))) * tf.exp(-tf.square(stacked_y - mu) / (2 * tf.square(sigma))),
            axis=2)

    def get_policyfunc(self):

        alpha = tf.placeholder(tf.float32, shape=(None, self.num_modal, 1), name='temp_alpha')
        mean = tf.placeholder(tf.float32, shape=(None, self.num_modal, self.action_dim), name='temp_mean')
        sigma = tf.placeholder(tf.float32, shape=(None, self.num_modal, self.action_dim), name='temp_sigma')
        action = tf.placeholder(tf.float32, shape=(None, self.action_dim), name='temp_action')

        result = self.tf_normal(action, mean, sigma)
        result = tf.multiply(result, tf.squeeze(alpha, axis=2))
        result = tf.reduce_sum(result, 1, keepdims=True)

        return alpha, mean, sigma, action, result

    def get_lossfunc(self, alpha, sigma, mu, y):
        # alpha: batch x num_modal x 1
        # sigma: batch x num_modal x action_dim
        # mu: batch x num_modal x action_dim
        # y: batch x action_dim
        result = self.tf_normal(y, mu, sigma)

        result = tf.multiply(result, tf.squeeze(alpha, axis=2))
        result = tf.reduce_sum(result, 1, keepdims=True)
        result = -tf.log(result)

        return tf.reduce_mean(result)

    def policy_gradient_ascent(self, alpha, mean, sigma, action_init):

        action = np.copy(action_init)

        ascent_count = 0
        update_flag = np.ones([alpha.shape[0], self.action_dim])  # batch_size * action_dim

        while np.any(update_flag > 0) and ascent_count < self.policy_gd_max_steps:
            action_old = np.copy(action)

            # print(np.shape(self.policy_action_gradients(alpha, mean, sigma, action)))
            # exit()
            gradients = self.policy_action_gradients(alpha, mean, sigma, action)[0]
            action += update_flag * self.policy_gd_alpha * gradients
            action = np.clip(action, self.action_min, self.action_max)

            # stop if action diff. is small
            stop_idx = [idx for idx in range(len(action)) if
                        np.mean(np.abs(action_old[idx] - action[idx]) / self.action_max) <= self.policy_gd_stop]
            update_flag[stop_idx] = 0
            # print(update_flag)

            ascent_count += 1

        # print('ascent count:', ascent_count)
        return action

    def policy_action_gradients(self, alpha, mean, sigma, action):

        return self.sess.run(self.policy_action_grads, feed_dict={
            self.temp_alpha: alpha,
            self.temp_mean: mean,
            self.temp_sigma: sigma,
            self.temp_action: action
        })

    def train_expert(self, *args):
        # args (inputs, action, predicted_q_value)
        return self.sess.run([self.q_prediction, self.expert_optimize], feed_dict={
            self.inputs: args[0],
            self.action: args[1],
            self.predicted_q_value: args[2],
            self.phase: True
        })

    def train_actor(self, *args):
        # args [inputs, actions, phase]
        return self.sess.run(self.actor_optimize, feed_dict={
            self.inputs: args[0],
            self.actions: args[1],
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

    def predict_action(self, *args):
        inputs = args[0]
        phase = args[1]

        # alpha: batchsize x num_modal x 1
        # mean: batchsize x num_modal x action_dim
        alpha, mean, sigma = self.sess.run(
            [self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
                self.inputs: inputs,
                self.phase: phase
            })

        self.setModalStats(alpha[0], mean[0], sigma[0])

        max_idx = np.argmax(np.squeeze(alpha, axis=2), axis=1)

        # best_mean = []
        # for idx, m in zip(max_idx, mean):
        #     best_mean.append(m[idx])
        # best_mean = np.asarray(best_mean)

        best_mean = [m[idx] for idx, m in zip(max_idx, mean)]

        old_best_mean = best_mean
        if self.use_policy_gd:
            best_mean = self.policy_gradient_ascent(alpha, mean, sigma, best_mean)

        return old_best_mean, best_mean

    def predict_action_target(self, *args):
        inputs = args[0]
        phase = args[1]

        # batchsize x num_modal x action_dim
        alpha, mean, sigma = self.sess.run([self.target_action_prediction_alpha, self.target_action_prediction_mean, self.target_action_prediction_sigma],
                                           feed_dict={
                                                self.target_inputs: inputs,
                                                self.target_phase: phase
                                            })

        max_idx = np.argmax(np.squeeze(alpha, axis=2), axis=1)

        # best_mean = []
        # for idx, m in zip(max_idx, mean):
        #     best_mean.append(m[idx])
        # best_mean = np.asarray(best_mean)
        best_mean = [m[idx] for idx, m in zip(max_idx, mean)]

        old_best_mean = best_mean
        if self.use_policy_gd:
            best_mean = self.policy_gradient_ascent(alpha, mean, sigma, best_mean)

        return best_mean

    # Should return n actions
    def sample_action(self, *args):
        # args [inputs]

        inputs = args[0]
        phase = args[1]

        # batchsize x action_dim
        alpha, mean, sigma = self.sess.run(
            [self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
                self.inputs: inputs,
                self.phase: phase
            })

        alpha = np.squeeze(alpha, axis=2)

        self.setModalStats(alpha[0], mean[0], sigma[0])

        # for each transition in batch
        # sampled_actions = []
        # for prob, m, s in zip(alpha, mean, sigma):
        #     modal_idx = np.random.choice(self.num_modal, self.num_samples, p=prob)
        #     # print(modal_idx)
        #     actions = list(map(lambda idx: np.random.normal(m[idx], s[idx]), modal_idx))
        #     sampled_actions.append(np.clip(actions, self.action_min, self.action_max))

        # TODO: Check multi-dimensional action case. Is it sampling correctly
        modal_idx_list = [np.random.choice(self.num_modal, self.num_samples, p=prob) for prob in alpha]
        sampled_actions = [np.clip(np.random.normal(m[idx], s[idx]), self.action_min, self.action_max) for idx, m, s in zip(modal_idx_list, mean, sigma)]

        return sampled_actions

    # Should return n actions
    def sample_action_target(self, *args):

        inputs = args[0]
        phase = args[1]

        alpha, mean, sigma = self.sess.run([self.target_action_prediction_alpha, self.target_action_prediction_mean,
                                            self.target_action_prediction_sigma], feed_dict={
            self.target_inputs: inputs,
            self.target_phase: phase
        })

        alpha = np.squeeze(alpha, axis=2)

        assert (self.num_modal == np.shape(alpha)[1])

        # for each transition in batch
        # sampled_actions = []
        # for prob, m, s in zip(alpha, mean, sigma):
        #     modal_idx = np.random.choice(self.num_modal, self.num_samples, p=prob)
        #     actions = list(map(lambda idx: np.random.normal(m[idx], s[idx]), modal_idx))
        #     sampled_actions.append(np.clip(actions, self.action_min, self.action_max))
        # TODO: Check multi-dimensional action case. Is it sampling correctly
        modal_idx_list = [np.random.choice(self.num_modal, self.num_samples, p=prob) for prob in alpha]
        sampled_actions = [np.clip(np.random.normal(m[idx], s[idx]), self.action_min, self.action_max) for idx, m, s in zip(modal_idx_list, mean, sigma)]

        return sampled_actions

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def gradient_ascent(self, state, action_init, is_training):

        action = np.copy(action_init)

        ascent_count = 0
        update_flag = np.ones([state.shape[0], self.action_dim])  # batch_size * action_dim

        while np.any(update_flag > 0) and ascent_count < self.gd_max_steps:
            action_old = np.copy(action)

            gradients = self.action_gradients(state, action, is_training)[0]
            action += update_flag * self.gd_alpha * gradients
            action = np.clip(action, self.action_min, self.action_max)

            # stop if action diff. is small
            stop_idx = [idx for idx in range(len(action)) if
                        np.mean(np.abs(action_old[idx] - action[idx]) / self.action_max) <= self.gd_stop]
            update_flag[stop_idx] = 0
            # print(update_flag)

            ascent_count += 1

        # print('ascent count:', ascent_count)
        return action

    def action_gradients(self, inputs, action, is_training):

        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.phase: is_training
        })

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_prediction, feed_dict={self.inputs: np.expand_dims(state, 0),
                                                                          self.action: np.expand_dims(action, 0),
                                                                          self.phase: False})

    def getPolicyFunction(self, alpha, mean, sigma):

        # alpha = np.squeeze(alpha, axis=1)
        mean = np.squeeze(mean, axis=1)
        sigma = np.squeeze(sigma, axis=1)

        return lambda action: np.sum(alpha * np.multiply(np.sqrt(1.0 / (2 * np.pi * np.square(sigma))),
                                                         np.exp(-np.square(action - mean) / (2.0 * np.square(sigma)))))

    def setModalStats(self, alpha, mean, sigma):
        self.saved_alpha = alpha
        self.saved_mean = mean
        self.saved_sigma = sigma

    def getModalStats(self):
        return self.saved_alpha, self.saved_mean, self.saved_sigma

