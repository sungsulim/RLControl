import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt


class ExpertNetwork(BaseNetwork):
    def __init__(self, sess, input_norm, layer_dim, state_dim, state_min, state_max,
                 action_dim, action_min, action_max, expert_lr, tau, norm_type):
        super(ExpertNetwork, self).__init__(sess, state_dim, action_dim, expert_lr, tau)

        self.l1 = layer_dim[0]
        self.l2 = layer_dim[1]

        self.state_min = state_min
        self.state_max = state_max

        self.action_min = action_min
        self.action_max = action_max

        self.input_norm = input_norm
        self.norm_type = norm_type

        # CEM Hydra network
        self.inputs, self.phase, self.action, self.q_prediction = self.build_network(
            scope_name='expert')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='expert')

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_q_prediction = self.build_network(
            scope_name='target_expert')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_expert')

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

            ### TODO Update loss

            # expert update
            self.expert_loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.q_prediction))
            self.expert_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.expert_optimize = self.expert_optimizer.minimize(self.expert_loss)


        # self.predict_action_op = self.predict_action_func(self.action_prediction_alpha, self.action_prediction_mean)
        # self.predict_action_target_op = self.predict_action_target_func(self.target_action_prediction_alpha, self.target_action_prediction_mean)

        # self.sample_action_op = self.sample_action_func()
        # self.sample_action_target_op = self.sample_action_target_func()

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            inputs = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            phase = tf.placeholder(tf.bool)
            action = tf.placeholder(tf.float32, [None, self.action_dim])

            # normalize inputs
            if self.norm_type is not 'none':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)

            q_prediction = self.network(inputs, action, phase)

        return inputs, phase, action, q_prediction

    def batch_norm_network(self, inputs, action, phase):
        raise NotImplementedError

    def network(self, inputs, action, phase):
        # shared net
        net = tf.contrib.layers.fully_connected(inputs, self.l1, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True),
                                                       # tf.truncated_normal_initializer(), \
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                       biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                           factor=1.0, mode="FAN_IN", uniform=True))

        net = self.apply_norm(net, activation_fn=tf.nn.relu, phase=phase, layer_num=1)

        # Q branch
        q_net = tf.contrib.layers.fully_connected(tf.concat([net, action], 1), self.l2,
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

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    # Buggy
    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_prediction, feed_dict={self.inputs: np.expand_dims(state, 0),
                                                                          self.action: np.expand_dims(action, 0),
                                                                          self.phase: False})

    def plotFunction(self, func1, func2, state, mean, x_min, x_max, resolution=1e2, display_title='', save_title='',
                     save_dir='', linewidth=2.0, ep_count=0, grid=True, show=False, equal_aspect=False):

        fig, ax = plt.subplots(2, sharex=True)
        # fig, ax = plt.subplots(figsize=(10, 5))

        x = np.linspace(x_min, x_max, resolution)
        y1 = []
        y2 = []

        max_point_x = x_min
        max_point_y = np.float('-inf')

        for point_x in x:
            point_y1 = np.squeeze(func1([point_x]))  # reduce dimension
            point_y2 = func2(point_x)

            if point_y1 > max_point_y:
                max_point_x = point_x
                max_point_y = point_y1

            y1.append(point_y1)
            y2.append(point_y2)

        ax[0].plot(x, y1, linewidth=linewidth)
        ax[1].plot(x, y2, linewidth=linewidth)
        # plt.ylim((-0.5, 1.6))
        if equal_aspect:
            ax.set_aspect('auto')

        if grid:
            ax[0].grid(True)
            # ax[0].axhline(y=0, linewidth=1.5, color='darkslategrey')
            # ax[0].axvline(x=0, linewidth=1.5, color='darkslategrey')

            ax[1].grid(True)
            ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')
            ax[1].axvline(x=0, linewidth=1.5, color='darkslategrey')

        if display_title:

            display_title += ", maxA: {:.3f}".format(max_point_x) + ", maxQ: {:.3f}".format(
                max_point_y) + "\n state: " + str(state)
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

            mode_string = ""
            for i in range(len(mean)):
                mode_string += "{:.3f}".format(np.squeeze(mean[i])) + ", "
            ax[1].set_title("modes: " + mode_string)

        else:
            top_margin = 1.0

        if show:
            plt.show()
        else:
            # print(save_title)
            save_dir = save_dir + '/figures/' + str(ep_count) + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_dir + save_title)
            plt.close()
