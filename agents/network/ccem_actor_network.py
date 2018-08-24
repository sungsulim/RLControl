import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class CCEM_ActorNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, layer_dim, state_dim, state_min, state_max, action_dim, action_min, action_max,
                 actor_lr, tau, rho, num_samples, num_modal, action_selection, norm_type):
        super(CCEM_ActorNetwork, self).__init__(sess, state_dim, action_dim, actor_lr, tau)

        self.l1 = layer_dim[0]
        self.l2 = layer_dim[1]

        self.state_min = state_min
        self.state_max = state_max

        self.action_min = action_min
        self.action_max = action_max

        self.input_norm = input_norm
        self.norm_type = norm_type

        self.rho = rho
        self.num_samples = num_samples
        self.num_modal = num_modal
        self.actor_output_dim = self.num_modal * (1 + 2 * self.action_dim)

        self.action_selection = action_selection

        # CEM Hydra network
        self.inputs, self.phase, self.action_prediction_mean, self.action_prediction_sigma, self.action_prediction_alpha = self.build_network(
            scope_name='actor')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Target network
        self.target_inputs, self.target_phase, self.target_action_prediction_mean, self.target_action_prediction_sigma, self.target_action_prediction_alpha = self.build_network(
            scope_name='target_actor')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = [
            tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx]))
            for idx in range(len(self.target_net_params))]

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

            # Actor update
            # self.actor_loss = tf.reduce_mean(tf.squared_difference(self.actions, self.action_prediction))
            # self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.actor_loss)
            self.actor_loss = self.get_lossfunc(self.action_prediction_alpha, self.action_prediction_sigma,
                                                self.action_prediction_mean, self.actions)

            # self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate)  # tf.train.AdamOptimizer(self.learning_rate)
            self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss)

        # self.predict_action_op = self.predict_action_func(self.action_prediction_alpha, self.action_prediction_mean)
        # self.predict_action_target_op = self.predict_action_target_func(self.target_action_prediction_alpha, self.target_action_prediction_mean)

        # self.sample_action_op = self.sample_action_func()
        # self.sample_action_target_op = self.sample_action_target_func()

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            phase = tf.placeholder(tf.bool)

            # normalize inputs
            if self.norm_type is not 'none':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            action_prediction_mean, action_prediction_sigma, action_prediction_alpha = self.network(
                inputs, phase)

        return inputs, phase, action_prediction_mean, action_prediction_sigma, action_prediction_alpha

    def batch_norm_network(self, inputs, phase):
        raise NotImplementedError

    def network(self, inputs, phase):
        # net
        net = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       # tf.truncated_normal_initializer(), \
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        net = self.apply_norm(net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # action branch
        action_net = tf.contrib.layers.fully_connected(net, self.l2, activation_fn=None,
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
                                                                    weights_initializer=tf.random_uniform_initializer(0, 3e-3),
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
        # action_prediction_sigma = tf.exp(tf.clip_by_value(action_prediction_sigma, -3.0, 3.0))
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

        return action_prediction_mean, action_prediction_sigma, action_prediction_alpha

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

    def get_lossfunc(self, alpha, sigma, mu, y):
        # alpha: batch x num_modal x 1
        # sigma: batch x num_modal x action_dim
        # mu: batch x num_modal x action_dim
        # y: batch x action_dim
        # print(np.shape(y), np.shape(mu), np.shape(sigma))
        result = self.tf_normal(y, mu, sigma)
        # print(np.shape(result), np.shape(alpha))

        result = tf.multiply(result, tf.squeeze(alpha, axis=2))
        # print('get_lossfunc1',np.shape(result))
        result = tf.reduce_sum(result, 1, keep_dims=True)
        # print('get_lossfunc2',np.shape(result))
        result = -tf.log(result)
        # exit()
        return tf.reduce_mean(result)

    def train_actor(self, *args):
        # args [inputs, actions, phase]
        return self.sess.run([self.actor_loss, self.actor_optimize], feed_dict={
            self.inputs: args[0],
            self.actions: args[1],
            self.phase: True
        })

    def predict_action(self, *args):
        inputs = args[0]
        phase = args[1]

        if self.action_selection == 'highest_alpha':

            # alpha: batchsize x num_modal x 1
            # mean: batchsize x num_modal x action_dim
            alpha, mean, sigma = self.sess.run(
                [self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
                    self.inputs: inputs,
                    self.phase: phase
                })

            self.setModalStats(alpha[0], mean[0], sigma[0])

            max_idx = np.argmax(np.squeeze(alpha, axis=2), axis=1)
            # print('predict action')
            # print('alpha', alpha, np.shape(alpha))
            # print('max_idx', max_idx)
            # print("mean", mean)

            best_mean = []
            for idx, m in zip(max_idx, mean):
                # print(idx, m)
                # print(m[idx])
                best_mean.append(m[idx])
            best_mean = np.asarray(best_mean)
            # print('best mean', best_mean, np.shape(best_mean))

            # input()

        elif self.action_selection == 'highest_q_val':

            raise NotImplementedError

        return best_mean

        #######################

    def predict_action_target(self, *args):
        inputs = args[0]
        phase = args[1]

        # batchsize x num_modal x action_dim
        alpha, mean = self.sess.run([self.target_action_prediction_alpha, self.target_action_prediction_mean],
                                    feed_dict={
                                        self.target_inputs: inputs,
                                        self.target_phase: phase
                                    })

        max_idx = np.argmax(np.squeeze(alpha, axis=2), axis=1)
        # print('predict action target')
        # print('alpha', alpha, np.shape(alpha))
        # print('max_idx', max_idx)
        # print("mean", mean)

        best_mean = []
        for idx, m in zip(max_idx, mean):
            # print(idx, m)
            # print(m[idx])
            best_mean.append(m[idx])

        best_mean = np.asarray(best_mean)
        # print('best mean', best_mean, np.shape(best_mean))

        # input()
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
        # print('alpha', alpha[0], np.shape(alpha[0]))
        # print('mean', mean[0], np.shape(mean[0]))
        # print('sigma', sigma[0], np.shape(sigma[0]))
        # input()

        # for each transition in batch
        sampled_actions = []
        for prob, m, s in zip(alpha, mean, sigma):
            modal_idx = np.random.choice(self.num_modal, self.num_samples, p=prob)
            # print(modal_idx)
            actions = list(map(lambda idx: np.random.normal(m[idx], s[idx]), modal_idx))
            sampled_actions.append(np.clip(actions, self.action_min,self.action_max))

        # print(sampled_actions, np.shape(sampled_actions))
        # input()
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

        # print('sample action target')
        # print('alpha', alpha, np.shape(alpha))
        # print("mean", mean, np.shape(mean))

        assert (self.num_modal == np.shape(alpha)[1])

        # for each transition in batch
        sampled_actions = []
        for prob, m, s in zip(alpha, mean, sigma):
            modal_idx = np.random.choice(self.num_modal, self.num_samples, p=prob)
            # print(modal_idx)
            actions = list(map(lambda idx: np.random.normal(m[idx], s[idx]), modal_idx))
            sampled_actions.append(np.clip(actions, self.action_min, self.action_max))

        # print(sampled_actions, np.shape(sampled_actions))

        # input()
        return sampled_actions

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    def getPolicyFunction(self, alpha, mean, sigma):

        mean = np.squeeze(mean, axis=1)
        sigma = np.squeeze(sigma, axis=1)

        return lambda action: np.sum(alpha * np.multiply(np.sqrt(1.0 / (2 * np.pi * np.square(sigma))),
                                                         np.exp(-np.square(action - mean) / (2.0 * np.square(sigma)))))

    def setModalStats(self, alpha, mean, sigma):
        self.temp_alpha = alpha
        self.temp_mean = mean
        self.temp_sigma = sigma

    def getModalStats(self):
        return self.temp_alpha, self.temp_mean, self.temp_sigma

    def printGradient(self, state, action):
        grads_and_vars = self.actor_optimizer.compute_gradients(self.actor_loss, var_list=self.net_params)
        # grads_and_vars = self.expert_optimizer.compute_gradients(self.expert_loss)

        # print(grads_and_vars)

        for gv in grads_and_vars:
            print(str(self.sess.run(gv[0], feed_dict={self.inputs: state,
                                                      self.actions: action,
                                                      self.phase: True})) + " - " + gv[1].name)