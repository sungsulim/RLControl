import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np
import os


import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

from matplotlib import pyplot as plt

class AE_CCEM_Network(BaseNetwork):
    def __init__(self, sess, input_norm, shared_layer_dim, actor_layer_dim, expert_layer_dim, state_dim, state_min, state_max, action_dim, action_min, action_max, actor_lr, expert_lr, tau, rho, num_samples, num_modal, action_selection, norm_type):
        super(AE_CCEM_Network, self).__init__(sess, state_dim, action_dim, [actor_lr, expert_lr], tau)

        self.shared_layer_dim = shared_layer_dim

        self.actor_layer_dim = actor_layer_dim
        self.expert_layer_dim = expert_layer_dim


        self.state_min = state_min
        self.state_max = state_max

        self.action_min = action_min
        self.action_max = action_max

        self.input_norm = input_norm
        self.norm_type = norm_type


        self.rho = rho
        self.num_samples = num_samples
        #self.var = np.ones(self.action_dim) * var_scale

        ### ###
        self.num_modal = num_modal
        self.actor_output_dim =  self.num_modal * (1 + 2 * self.action_dim)

        # print('1st cutoff', int(self.num_modal * self.action_dim))
        # print('2nd cutoff', int(2 * self.num_modal * self.action_dim))
        # print('total actor_output_dim', self.actor_output_dim)
        ### ###

        self.action_selection = action_selection

        # CEM Hydra network
        self.inputs, self.phase, self.action, self.action_prediction_mean, self.action_prediction_sigma, self.action_prediction_alpha , self.q_prediction= self.build_network(scope_name = 'cem_hydra')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cem_hydra')

        # Target network
        self.target_inputs, self.target_phase, self.target_action, self.target_action_prediction_mean, self.target_action_prediction_sigma, self.target_action_prediction_alpha , self.target_q_prediction= self.build_network(scope_name = 'target_cem_hydra')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_cem_hydra')

        # Op for periodically updating target network with online network weights
        self.update_target_net_params  = [tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx])) for idx in range(len(self.target_net_params))]

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
            self.expert_optimize = tf.train.AdamOptimizer(self.learning_rate[1]).minimize(self.expert_loss)


            # Actor update
            # self.actor_loss = tf.reduce_mean(tf.squared_difference(self.actions, self.action_prediction))
            # self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.actor_loss)
            self.actor_loss = self.get_lossfunc(self.action_prediction_alpha, self.action_prediction_sigma, self.action_prediction_mean, self.actions)
            self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.actor_loss)

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
            if self.norm_type == 'input_norm' or self.norm_type == 'layer':
                inputs = tf.clip_by_value(self.input_norm.normalize(inputs), self.state_min, self.state_max)


            if self.norm_type == 'layer':
                action_prediction_mean, action_prediction_sigma, action_prediction_alpha, q_prediction = self.layer_norm_network(inputs, action, phase)

            elif self.norm_type == 'batch':
                assert (self.input_norm is None)
                raise NotImplementedError

            else:
                assert( self.norm_type == 'none' or self.norm_type == 'input_norm')
                action_prediction_mean, action_prediction_sigma, action_prediction_alpha, q_prediction = self.no_norm_network(inputs, action, phase)

        return inputs, phase, action, action_prediction_mean, action_prediction_sigma, action_prediction_alpha, q_prediction

    def batch_norm_network(self, inputs, action, phase):
        raise NotImplementedError

    def layer_norm_network(self, inputs, action, phase):
        # shared net
        shared_net = tf.contrib.layers.fully_connected(inputs, self.shared_layer_dim, activation_fn=None, \
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))
        # layer norm
        shared_net = tf.contrib.layers.layer_norm(shared_net, center=True, scale=True, activation_fn=tf.nn.relu)

        # action branch
        action_net = tf.contrib.layers.fully_connected(shared_net, self.actor_layer_dim, activation_fn=None, \
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))
        # layer norm
        action_net = tf.contrib.layers.layer_norm(action_net, center=True, scale=True, activation_fn=tf.nn.relu)

        # action_outputs = tf.contrib.layers.fully_connected(action_net, self.actor_output_dim, activation_fn=tf.tanh, \
        #                                 weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
        #                                 weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
        #                                 biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        # # mean, sigma, coeff
        # action_prediction_mean, action_prediction_sigma, action_prediction_alpha = tf.split(action_outputs, [self.num_modal * self.action_dim, self.num_modal * self.action_dim, self.num_modal], 1)
        
        action_prediction_mean = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim, activation_fn=tf.tanh, \
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True),#tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True)) #tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_sigma = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim, activation_fn=None, \
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_alpha = tf.contrib.layers.fully_connected(action_net, self.num_modal, activation_fn=tf.tanh, \
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        # reshape output
        action_prediction_mean = tf.reshape(action_prediction_mean, [-1, self.num_modal, self.action_dim])
        action_prediction_sigma = tf.reshape(action_prediction_sigma, [-1, self.num_modal, self.action_dim])
        action_prediction_alpha = tf.reshape(action_prediction_alpha, [-1, self.num_modal, 1])

        # scale mean to env. action domain
        action_prediction_mean = tf.multiply(action_prediction_mean, self.action_max)

        # exp. sigma
        action_prediction_sigma = tf.exp(tf.clip_by_value(action_prediction_sigma, 0.0, 5.0))
        
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
        q_net = tf.contrib.layers.fully_connected(tf.concat([shared_net, action], 1), self.expert_layer_dim, activation_fn=None, \
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))
        # layer norm
        q_net = tf.contrib.layers.layer_norm(q_net, center=True, scale=True, activation_fn=tf.nn.relu)


        q_prediction = tf.contrib.layers.fully_connected(q_net, 1, activation_fn = None, \
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return action_prediction_mean, action_prediction_sigma, action_prediction_alpha, q_prediction



    def no_norm_network(self, inputs, action, phase):

        # shared net
        shared_net = tf.contrib.layers.fully_connected(inputs, self.shared_layer_dim, activation_fn=tf.nn.relu, \
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        # action branch
        action_net = tf.contrib.layers.fully_connected(shared_net, self.actor_layer_dim, activation_fn=tf.nn.relu, \
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))
        
        # action_outputs = tf.contrib.layers.fully_connected(action_net, self.actor_output_dim, activation_fn=tf.tanh, \
        #                                 weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
        #                                 weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
        #                                 biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        # # mean, sigma, coeff
        # action_prediction_mean, action_prediction_sigma, action_prediction_alpha = tf.split(action_outputs, [self.num_modal * self.action_dim, self.num_modal * self.action_dim, self.num_modal], 1)
        
        action_prediction_mean = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim, activation_fn=tf.tanh, \
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True),#tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True)) #tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_sigma = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim, activation_fn=None, \
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_alpha = tf.contrib.layers.fully_connected(action_net, self.num_modal, activation_fn=tf.tanh, \
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        # reshape output
        action_prediction_mean = tf.reshape(action_prediction_mean, [-1, self.num_modal, self.action_dim])
        action_prediction_sigma = tf.reshape(action_prediction_sigma, [-1, self.num_modal, self.action_dim])
        action_prediction_alpha = tf.reshape(action_prediction_alpha, [-1, self.num_modal, 1])

        # scale mean to env. action domain
        action_prediction_mean = tf.multiply(action_prediction_mean, self.action_max)

        # exp. sigma
        action_prediction_sigma = tf.exp(tf.clip_by_value(action_prediction_sigma, 0.0, 5.0))
        
        # mean: [None, num_modal, action_dim]  : [None, 1]
        # sigma: [None, num_modal, action_dim] : [None, 1]
        # alpha: [None, num_modal, 1]              : [None, 1]

        # compute softmax prob. of alpha
        max_alpha = tf.reduce_max(action_prediction_alpha, axis=1, keep_dims=True)
        action_prediction_alpha = tf.subtract(action_prediction_alpha, max_alpha)
        action_prediction_alpha = tf.exp(action_prediction_alpha)

        normalize_alpha = tf.reciprocal(tf.reduce_sum(action_prediction_alpha, axis=1, keep_dims=True))
        action_prediction_alpha = tf.multiply(normalize_alpha, action_prediction_alpha)

        
        # Q branch
        q_net = tf.contrib.layers.fully_connected(tf.concat([shared_net, action], 1), self.expert_layer_dim, activation_fn=tf.nn.relu, \
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(), \
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        q_prediction = tf.contrib.layers.fully_connected(q_net, 1, activation_fn = None, \
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01), \
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return action_prediction_mean, action_prediction_sigma, action_prediction_alpha, q_prediction

    
    def tf_normal(self, y, mu, sigma):

        # y: batch x action_dim 
        # mu: batch x num_modal x action_dim
        # sigma: batch x num_modal x action_dim

        # stacked y: batch x num_modal x action_dim
        stacked_y = tf.expand_dims(y, 1)
        stacked_y = tf.tile(stacked_y, [1,self.num_modal,1])

        return tf.reduce_prod(tf.sqrt(1.0 / (2 * np.pi * tf.square(sigma))) * tf.exp(-tf.square(stacked_y - mu) / (2 * tf.square(sigma))), axis=2)


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




        if self.action_selection == 'highest_alpha':

            # alpha: batchsize x num_modal x 1
            # mean: batchsize x num_modal x action_dim
            alpha, mean, sigma = self.sess.run([self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
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
            for idx,m in zip(max_idx, mean):
                # print(idx, m)
                # print(m[idx])
                best_mean.append(m[idx])
            best_mean = np.asarray(best_mean)
            # print('best mean', best_mean, np.shape(best_mean))

            # input()

        elif self.action_selection == 'highest_q_val':
            # mean: batchsize x num_modal x action_dim
            alpha, mean, sigma = self.sess.run([self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
                self.inputs: inputs,
                self.phase: phase
            })

            self.setModalStats(alpha[0], mean[0], sigma[0])

            # print('mean shape', np.shape(mean))
            mean_reshaped = np.reshape(mean, (np.shape(mean)[0] * np.shape(mean)[1], np.shape(mean)[2]))
            # print('mean_reshaped', np.shape(mean_reshaped))


            stacked_state_batch = None

            for state in inputs:
                stacked_one_state = np.tile(state, (self.num_modal, 1))

                if stacked_state_batch is None:
                    stacked_state_batch = stacked_one_state
                else:
                    stacked_state_batch = np.concatenate((stacked_state_batch, stacked_one_state), axis=0)

            # print('stacked_states', np.shape(stacked_state_batch))

            q_prediction = np.expand_dims(self.predict_q(stacked_state_batch, mean_reshaped, True), axis=0)

            # print('q_pred', np.shape(q_prediction))
            # print( (np.shape(mean)[0], np.shape(mean)[1], -1))

            q_prediction = np.squeeze(np.reshape(q_prediction, (np.shape(mean)[0], np.shape(mean)[1], -1)), axis=2)

            # print('q_pred', q_prediction)
            best_mean = [mean[b][np.argmax(q_prediction[b])] for b in range(len(q_prediction))]
            # print('mean', mean)
            # print(best_mean)
            # input()


        return best_mean

        #######################
    def predict_action_target(self, *args):
        inputs = args[0]
        phase = args[1]


        # batchsize x num_modal x action_dim
        alpha, mean = self.sess.run([self.target_action_prediction_alpha, self.target_action_prediction_mean], feed_dict={
            self.target_inputs: inputs,
            self.target_phase: phase
        })

        max_idx = np.argmax(np.squeeze(alpha, axis=2), axis=1)
        # print('predict action target')
        # print('alpha', alpha, np.shape(alpha))
        # print('max_idx', max_idx)
        # print("mean", mean)

        best_mean = []
        for idx,m in zip(max_idx, mean):
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
        alpha, mean, sigma = self.sess.run([self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
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
            modal_idx = np.random.choice(self.num_modal, self.num_samples, p = prob)
            # print(modal_idx)
            actions = list(map(lambda idx: np.random.normal(m[idx], s[idx]), modal_idx ))
            sampled_actions.append(actions)
        
        # print(sampled_actions, np.shape(sampled_actions))
        # input()
        return sampled_actions

        

    # Should return n actions
    def sample_action_target(self, *args):

        inputs = args[0]
        phase = args[1]

        alpha, mean, sigma = self.sess.run([self.target_action_prediction_alpha, self.target_action_prediction_mean, self.target_action_prediction_sigma], feed_dict={
            self.target_inputs: inputs,
            self.target_phase: phase
        })

        alpha = np.squeeze(alpha, axis=2)

        # print('sample action target')
        # print('alpha', alpha, np.shape(alpha))
        # print("mean", mean, np.shape(mean))

        assert(self.num_modal == np.shape(alpha)[1])

        # for each transition in batch
        sampled_actions = []
        for prob, m, s in zip(alpha, mean, sigma):
            modal_idx = np.random.choice(self.num_modal, self.num_samples, p = prob)
            # print(modal_idx)
            actions = list(map(lambda idx: np.random.normal(m[idx], s[idx]), modal_idx ))
            sampled_actions.append(actions)
        
        # print(sampled_actions, np.shape(sampled_actions))

        # input()
        return sampled_actions



    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])



    # Buggy
    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_prediction, feed_dict={self.inputs: np.expand_dims(state, 0), 
                                            self.action: np.expand_dims(action, 0), 
                                            self.phase:False})

    def getPolicyFunction(self, alpha, mean, sigma):

        mean = np.squeeze(mean, axis=1)
        sigma = np.squeeze(sigma, axis=1)

        return lambda action: np.sum(alpha * np.multiply(np.sqrt(1.0 / (2 * np.pi * np.square(sigma))), np.exp(-np.square(action - mean) / (2.0 * np.square(sigma)))))

    def plotFunction(self, func1, func2, state, mean, x_min, x_max, resolution=1e2, display_title='', save_title='', save_dir='', linewidth=2.0, ep_count=0, grid=True, show=False, equal_aspect=False):

        fig, ax = plt.subplots(2, sharex=True)
        # fig, ax = plt.subplots(figsize=(10, 5))

        x = np.linspace(x_min, x_max, resolution)
        y1 = []
        y2 = []

        max_point_x = x_min
        max_point_y = np.float('-inf')

        for point_x in x:
            point_y1 = np.squeeze(func1([point_x])) # reduce dimension
            point_y2 = func2(point_x)

            if point_y1 > max_point_y:
                max_point_x = point_x
                max_point_y = point_y1

            y1.append(point_y1)
            y2.append(point_y2)

        ax[0].plot(x, y1, linewidth = linewidth)
        ax[1].plot(x, y2, linewidth = linewidth)
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

            display_title += ", maxA: {:.3f}".format(max_point_x) + ", maxQ: {:.3f}".format(max_point_y) + "\n state: " + str(state)
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95


            mode_string=""
            for i in range(len(mean)):
                mode_string+= "{:.3f}".format(np.squeeze(mean[i])) +", "
            ax[1].set_title("modes: " + mode_string)

        else: 
            top_margin = 1.0


        #plt.tight_layout(rect=(0,0,1, top_margin))

        if show:
            plt.show()
        else:
            #print(save_title)
            save_dir = save_dir+'/figures/'+str(ep_count)+'/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_dir+save_title)
            plt.close()

    def setModalStats(self, alpha, mean, sigma):
        self.temp_alpha = alpha
        self.temp_mean = mean
        self.temp_sigma = sigma

    def getModalStats(self):
        return self.temp_alpha, self.temp_mean, self.temp_sigma

