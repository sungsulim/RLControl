import tensorflow as tf
from agents.network.base_network import BaseNetwork
import numpy as np

tfd = tf.contrib.distributions

from matplotlib import pyplot as plt

class AE_CCEM_Network(BaseNetwork):
    def __init__(self, sess, input_norm, shared_layer_dim, actor_layer_dim, expert_layer_dim, state_dim, state_min, state_max, action_dim, action_min, action_max, actor_lr, expert_lr, tau, rho, num_samples, num_modal, norm_type):
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
        # self.var = np.ones(self.action_dim) * var_scale

        ### ###
        self.num_modal = num_modal
        self.actor_output_dim = self.num_modal * (1 + 2 * self.action_dim)

        # print('1st cutoff', int(self.num_modal * self.action_dim))
        # print('2nd cutoff', int(2 * self.num_modal * self.action_dim))
        # print('total actor_output_dim', self.actor_output_dim)
        ### ###

        # CEM Hydra network
        self.state, self.phase, self.action, self.action_prediction_mean, self.action_prediction_sigma, self.action_prediction_alpha, self.gauss_mix, self.q_prediction = self.build_network(scope_name = 'cem_hydra')
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cem_hydra')

        # Target network
        self.target_state, self.target_phase, self.target_action, self.target_action_prediction_mean, self.target_action_prediction_sigma, self.target_action_prediction_alpha, self.target_gauss_mix, self.target_q_prediction = self.build_network(scope_name = 'target_cem_hydra')
        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_cem_hydra')


        # Op for predict_action
        max_idx = tf.argmax(self.action_prediction_alpha, axis=1)
        self.max_alpha_action = tf.map_fn(lambda x: x[1][x[0]], [max_idx, self.action_prediction_mean],
                                          dtype=tf.float32)

        # Op for predict_action_target
        max_idx_target = tf.argmax(self.target_action_prediction_alpha, axis=1)
        self.max_alpha_action_target = tf.map_fn(lambda x: x[1][x[0]],
                                                 [max_idx_target, self.target_action_prediction_mean], dtype=tf.float32)

        # Op for sampling action
        self.current_num_sample = tf.placeholder(tf.int32)
        self.sample_action_op = tf.transpose(self.gauss_mix.sample(sample_shape=(self.current_num_sample)), [1, 0, 2])


        # Op for periodically updating target network with online network weights
        self.update_target_net_params = [tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx])) for idx in range(len(self.target_net_params))]

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

            # expert update
            self.expert_loss = tf.reduce_mean(tf.squared_difference(self.predicted_q_value, self.q_prediction))
            self.expert_optimize = tf.train.AdamOptimizer(self.learning_rate[1]).minimize(self.expert_loss)


            # Actor update
            # self.actor_loss = tf.reduce_mean(tf.squared_difference(self.actions, self.action_prediction))
            # self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.actor_loss)

            self.actor_loss = tf.reduce_mean(-self.gauss_mix.log_prob(self.actions))
            # self.actor_loss = self.get_lossfunc(self.action_prediction_alpha, self.action_prediction_sigma,
            #                                     self.action_prediction_mean, self.actions)
            self.actor_optimize = tf.train.AdamOptimizer(self.learning_rate[0]).minimize(self.actor_loss)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):
            state = tf.placeholder(tf.float32, shape=(None, self.state_dim))
            phase = tf.placeholder(tf.bool)

            action = tf.placeholder(tf.float32, [None, self.action_dim])

            # normalize state
            if self.norm_type == 'input_norm' or self.norm_type == 'layer':
                state = tf.clip_by_value(self.input_norm.normalize(state), self.state_min, self.state_max)

            if self.norm_type == 'layer':
                action_prediction_mean, action_prediction_sigma, action_prediction_alpha, gauss_mix, q_prediction = self.layer_norm_network(state, action, phase)

            elif self.norm_type == 'batch':
                assert (self.input_norm is None)
                raise NotImplementedError

            else:
                assert( self.norm_type == 'none' or self.norm_type == 'input_norm')
                action_prediction_mean, action_prediction_sigma, action_prediction_alpha, gauss_mix, q_prediction = self.no_norm_network(state, action, phase)

        return state, phase, action, action_prediction_mean, action_prediction_sigma, action_prediction_alpha, gauss_mix, q_prediction

    def batch_norm_network(self, state, action, phase):
        raise NotImplementedError

    def layer_norm_network(self, state, action, phase):
        # shared net
        shared_net = tf.contrib.layers.fully_connected(state, self.shared_layer_dim, activation_fn=None,
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))
        # layer norm
        shared_net = tf.contrib.layers.layer_norm(shared_net, center=True, scale=True, activation_fn=tf.nn.relu)

        # action branch
        action_net = tf.contrib.layers.fully_connected(shared_net, self.actor_layer_dim, activation_fn=None,
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(),
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001),
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))
        # layer norm
        action_net = tf.contrib.layers.layer_norm(action_net, center=True, scale=True, activation_fn=tf.nn.relu)

        
        action_prediction_mean = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim, activation_fn=tf.tanh,
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True),#tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True)) #tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_sigma = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim, activation_fn=None, #tf.tanh,
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001),
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_alpha = tf.contrib.layers.fully_connected(action_net, self.num_modal, activation_fn=tf.tanh,
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001),
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        # reshape output
        action_prediction_mean = tf.reshape(action_prediction_mean, [-1, self.num_modal, self.action_dim])
        action_prediction_sigma = tf.reshape(action_prediction_sigma, [-1, self.num_modal, self.action_dim])
        action_prediction_alpha = tf.reshape(action_prediction_alpha, [-1, self.num_modal, 1])

        # scale mean to env. action domain
        action_prediction_mean = tf.multiply(action_prediction_mean, self.action_max)


        # mean: [None, num_modal, action_dim]  : [None, 1]
        # sigma: [None, num_modal, action_dim] : [None, 1]
        # alpha: [None, num_modal, 1]              : [None, 1]

        # compute softmax prob. of alpha
        max_alpha = tf.reduce_max(action_prediction_alpha, axis=1, keepdims=True)
        action_prediction_alpha = tf.subtract(action_prediction_alpha, max_alpha)
        action_prediction_alpha = tf.exp(action_prediction_alpha)

        normalize_alpha = tf.reciprocal(tf.reduce_sum(action_prediction_alpha, axis=1, keepdims=True))
        action_prediction_alpha = tf.multiply(normalize_alpha, action_prediction_alpha)
        action_prediction_alpha = tf.squeeze(action_prediction_alpha, axis=2)

        bimix_gauss = tfd.Mixture(
            cat = tfd.Categorical(probs=action_prediction_alpha),
            components = [tfd.MultivariateNormalDiagWithSoftplusScale(loc=action_prediction_mean[:,i,:], scale_diag=action_prediction_sigma[:,i,:]) for i in range(self.num_modal)])


        # Q branch
        q_net = tf.contrib.layers.fully_connected(tf.concat([shared_net, action], 1), self.expert_layer_dim, activation_fn=None,
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))
        # layer norm
        q_net = tf.contrib.layers.layer_norm(q_net, center=True, scale=True, activation_fn=tf.nn.relu)


        q_prediction = tf.contrib.layers.fully_connected(q_net, 1, activation_fn = None,
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return action_prediction_mean, action_prediction_sigma, action_prediction_alpha, bimix_gauss, q_prediction



    def no_norm_network(self, state, action, phase):
        # shared net
        shared_net = tf.contrib.layers.fully_connected(state, self.shared_layer_dim, activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        # action branch
        action_net = tf.contrib.layers.fully_connected(shared_net, self.actor_layer_dim, activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(),
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001),
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        
        action_prediction_mean = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim, activation_fn=tf.tanh,
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True),#tf.random_uniform_initializer(-3e-3, 3e-3), \
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001), \
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True)) #tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_sigma = tf.contrib.layers.fully_connected(action_net, self.num_modal * self.action_dim, activation_fn=None, #tf.tanh,
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001),
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        action_prediction_alpha = tf.contrib.layers.fully_connected(action_net, self.num_modal, activation_fn=tf.tanh,
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                        weights_regularizer=None, #tf.contrib.layers.l2_regularizer(0.001),
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        # reshape output
        action_prediction_mean = tf.reshape(action_prediction_mean, [-1, self.num_modal, self.action_dim])
        action_prediction_sigma = tf.reshape(action_prediction_sigma, [-1, self.num_modal, self.action_dim])
        action_prediction_alpha = tf.reshape(action_prediction_alpha, [-1, self.num_modal, 1])

        # scale mean to env. action domain
        action_prediction_mean = tf.multiply(action_prediction_mean, self.action_max)


        # mean: [None, num_modal, action_dim]  : [None, 1]
        # sigma: [None, num_modal, action_dim] : [None, 1]
        # alpha: [None, num_modal, 1]              : [None, 1]

        # compute softmax prob. of alpha
        max_alpha = tf.reduce_max(action_prediction_alpha, axis=1, keepdims=True)
        action_prediction_alpha = tf.subtract(action_prediction_alpha, max_alpha)
        action_prediction_alpha = tf.exp(action_prediction_alpha)

        normalize_alpha = tf.reciprocal(tf.reduce_sum(action_prediction_alpha, axis=1, keepdims=True))
        action_prediction_alpha = tf.multiply(normalize_alpha, action_prediction_alpha)
        action_prediction_alpha = tf.squeeze(action_prediction_alpha, axis=2)

        bimix_gauss = tfd.Mixture(
            cat = tfd.Categorical(probs=action_prediction_alpha),
            components = [tfd.MultivariateNormalDiagWithSoftplusScale(loc=action_prediction_mean[:,i,:], scale_diag=action_prediction_sigma[:,i,:]) for i in range(self.num_modal)])


        # Q branch
        q_net = tf.contrib.layers.fully_connected(tf.concat([shared_net, action], 1), self.expert_layer_dim, activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True), #tf.truncated_normal_initializer(),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                        biases_initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_IN", uniform=True))

        q_prediction = tf.contrib.layers.fully_connected(q_net, 1, activation_fn = None,
                                        weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                        biases_initializer=tf.random_uniform_initializer(-3e-3, 3e-3))

        return action_prediction_mean, action_prediction_sigma, action_prediction_alpha, bimix_gauss, q_prediction

    def tf_normal(self, y, mu, sigma):


        # y: batch x action_dim 
        # mu: batch x num_modal x action_dim
        # sigma: batch x num_modal x action_dim

        # stacked y: batch x num_modal x action_dim
        stacked_y = tf.expand_dims(y, 1)
        stacked_y = tf.tile(stacked_y, [1, self.num_modal, 1])

        # oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)  # normalisation factor for gaussian
        # result = tf.subtract(stacked_y, mu)
        # result = tf.multiply(result, tf.reciprocal(sigma))
        # result = -tf.square(result)/2
        # return tf.reduce_prod(tf.multiply(tf.exp(result), tf.reciprocal(sigma))*oneDivSqrtTwoPI, axis=2)

        # Product along action_dim
        return tf.reduce_prod(tf.sqrt(1.0 / (2 * np.pi * tf.square(sigma))) * tf.exp(-tf.square(stacked_y - mu) / (2 * tf.square(sigma))), axis=2)

    # def get_lossfunc(self, alpha, sigma, mu, y):
    #     # alpha: batch x num_modal
    #     # sigma: batch x num_modal x action_dim
    #     # mu: batch x num_modal x action_dim
    #     # y: batch x action_dim
    #     # print(np.shape(y), np.shape(mu), np.shape(sigma))
    #
    #     # shape: batch x num_modal
    #     result = self.tf_normal(y, mu, sigma)
    #     # print(np.shape(result))
    #     result = tf.multiply(result, alpha)
    #     # print('get_lossfunc1',np.shape(result))
    #     result = tf.reduce_sum(result, 1, keep_dims=True)
    #     # print('get_lossfunc2',np.shape(result))
    #     result = -tf.log(result)
    #     # exit()
    #     return tf.reduce_mean(result)


    def train_expert(self, state_batch, action_batch, target_batch):
        # predict_q_target(next_state_batch, next_action_batch_final_target, True)
        # self.target_q_prediction
        # compute target: r + gamma_batch * target_q
        # train_expert

        return self.sess.run([self.q_prediction, self.expert_optimize], feed_dict={
            self.state: state_batch,
            self.action: action_batch,
            self.predicted_q_value: target_batch,
            self.phase: True
        })

    def train_actor(self, *args):
        # args [state, actions, phase]
        return self.sess.run(self.actor_optimize, feed_dict={
            self.state: args[0],
            self.actions: args[1],
            self.phase: True
        })

    def predict_q(self, *args):
        # args  (state, action, phase)
        state = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.q_prediction, feed_dict={
            self.state: state,
            self.action: action,
            self.phase: phase
        })

    def predict_q_target(self, *args):
        # args  (state, action, phase)
        state = args[0]
        action = args[1]
        phase = args[2]

        return self.sess.run(self.target_q_prediction, feed_dict={
            self.target_state: state,
            self.target_action: action,
            self.target_phase: phase
        })

    # def predict_action_func(self, alpha, mean):
    #     # alpha: batch_size x num_modal x 1
    #     # mean: batch_size x num_modal x action_dim
    #     max_idx = tf.argmax(alpha, axis=1)
    #     return mean[max_idx]

    # def predict_action_target_func(self, ):  
    #     pass

    def predict_action(self, state, phase):

        # alpha: batchsize x num_modal x 1
        # mean: batchsize x num_modal x action_dim

        best_mean = self.sess.run(self.max_alpha_action, feed_dict={
            self.state: state,
            self.phase: phase
        })

        return best_mean

    def predict_action_target(self, state, phase):

        # batchsize x num_modal x action_dim
        best_mean_target = self.sess.run(self.max_alpha_action_target, feed_dict={
            self.target_state: state,
            self.target_phase: phase
        })

        # Old code segment
        # # Get dominating alpha idx
        # max_idx = np.argmax(np.squeeze(alpha, axis=2), axis=1)
        # # print('predict action target')
        # # print('alpha', alpha, np.shape(alpha))
        # # print('max_idx', max_idx)
        # # print("mean", mean)
        #
        # best_mean = []
        # for idx, m in zip(max_idx, mean):
        #     # print(idx, m)
        #     # print(m[idx])
        #     best_mean.append(m[idx])
        #
        # best_mean = np.asarray(best_mean)
        # # print('best mean', best_mean, np.shape(best_mean))
        #
        # # input()
        return best_mean_target




    # Should return n actions
    def sample_action(self, batch_size, state, phase, do_single_sample):

        if do_single_sample:
            num_sample = 1
        else:
            num_sample = self.num_samples

        sampled_actions, mean, sigma, alpha = self.sess.run([self.sample_action_op, self.action_prediction_mean, self.action_prediction_sigma, self.action_prediction_alpha], feed_dict={
            self.state: state,
            self.phase: phase,
            self.current_num_sample: num_sample
        })

        self.setAlpha1(alpha[0])
        self.setMean1(mean[0])
        self.setSigma1(sigma[0])

        #print(alpha)
        #print(mean)
        #print(sigma)
        #input()

        # print(sampled_actions)
        # print(np.shape(sampled_actions))
        # print('batch_size', batch_size)
        # input()

        return sampled_actions
        ''' # old code segment
        # batchsize x action_dim
        alpha, mean, sigma = self.sess.run([self.action_prediction_alpha, self.action_prediction_mean, self.action_prediction_sigma], feed_dict={
            self.state: state,
            self.phase: phase
        })


        # for each transition in batch
        # tfd = tf.contrib.distributions
        sampled_actions = []
        for prob, m, s in zip(alpha, mean, sigma):

            modal_idx = np.random.choice(self.num_modal, self.num_samples, p = prob)

            # actions = list(map(lambda idx: np.random.multivariatenormal(m[idx], np.diag(s[idx])), modal_idx))
            actions = list(map(lambda idx: np.random.normal(m[idx], s[idx]), modal_idx))

            # actions = np.ones((self.num_samples, 1))
            sampled_actions.append(actions)

        # print(sampled_actions, np.shape(sampled_actions))
        # input()
        return sampled_actions
        '''
        

    # Should return n actions
    # Not used
    def sample_action_target(self, state, phase):

        alpha, mean, sigma = self.sess.run([self.target_action_prediction_alpha, self.target_action_prediction_mean, self.target_action_prediction_sigma], feed_dict={
            self.target_state: state,
            self.target_phase: phase
        })

        # alpha = np.squeeze(alpha, axis=2)

        # print('sample action target')
        # print('alpha', alpha, np.shape(alpha))
        # print("mean", mean, np.shape(mean))

        assert(self.num_modal == np.shape(alpha)[1])

        # TODO: Use Map instead?
        # for each transition in batch
        sampled_actions = []
        for prob, m, s in zip(alpha, mean, sigma):
            modal_idx = np.random.choice(self.num_modal, self.num_samples, p = prob)
            # print(modal_idx)
            actions = list(map(lambda idx: np.random.normal(m[idx], s[idx]), modal_idx))
            # actions = list(map(lambda idx: np.random.multivariate_normal(m[idx], np.diag(s[idx])), modal_idx))
            sampled_actions.append(actions)
        
        # print(sampled_actions, np.shape(sampled_actions))

        # input()
        return sampled_actions

    def update_target_network(self):
        self.sess.run([self.update_target_net_params, self.update_target_batchnorm_params])

    # Buggy
    def getQFunction(self, state):
        return lambda action: self.sess.run(self.q_prediction, feed_dict={self.state: np.expand_dims(state, 0),
                                            self.action: np.expand_dims(action, 0), 
                                            self.phase:False})

    # Buggy
    def plotFunc(self,func, x_min, x_max, resolution=1e2, display_title='', save_title='', linewidth=2.0, grid=True, show=True, equal_aspect=False):

        fig, ax = plt.subplots(figsize=(10, 5))    

        x = np.linspace(x_min, x_max, resolution)
        y = []

        max_point_x = x_min
        max_point_y = np.float('-inf')

        for point_x in x:
            point_y = np.squeeze(func([point_x])) # reduce dimension

            if point_y > max_point_y:
                max_point_x = point_x
                max_point_y = point_y

            y.append(point_y)

        ax.plot(x, y, linewidth = linewidth)
        plt.ylim((-0.5, 1.6))
        if equal_aspect:
            ax.set_aspect('equal')

        if grid:
            ax.grid(True)
            ax.axhline(y=0, linewidth=1.5, color='darkslategrey')
            ax.axvline(x=0, linewidth=1.5, color='darkslategrey')

        if display_title:
            display_title+= ", true_maxA: " + str(max_point_x) + ', true_maxQ: '+str(max_point_y)
            fig.suptitle(display_title, fontsize=11, fontweight='bold')
            top_margin = 0.95

        else: 
            top_margin = 1.0


        #plt.tight_layout(rect=(0,0,1, top_margin))

        if show:
            plt.show()
        else:
            #print(save_title)
            plt.savefig('./figures/'+save_title)
            plt.close()


    def setMean1(self, mean):
        self.temp_mean1 = mean

    def getMean1(self):
        return self.temp_mean1

    def setMean2(self, mean):
        self.temp_mean2 = mean

    def getMean2(self):
        return self.temp_mean2

    def setMean3(self, mean):
        self.temp_mean3 = mean

    def getMean3(self):
        return self.temp_mean3

    # set sigma (for tf Summary)
    def setSigma1(self, sigma):
        self.temp_sigma1 = sigma

    def getSigma1(self):
        return self.temp_sigma1

    def setSigma2(self, sigma):
        self.temp_sigma2 = sigma

    def getSigma2(self):
        return self.temp_sigma2

    def setSigma3(self, sigma):
        self.temp_sigma3 = sigma

    def getSigma3(self):
        return self.temp_sigma3

    # set sigma (for tf Summary)
    def setAlpha1(self, alpha):
        self.temp_alpha1 = alpha

    def getAlpha1(self):
        return self.temp_alpha1

    def setAlpha2(self, alpha):
        self.temp_alpha2 = alpha

    def getAlpha2(self):
        return self.temp_alpha2

    def setAlpha3(self, alpha):
        self.temp_alpha3 = alpha

    def getAlpha3(self):
        return self.temp_alpha3
