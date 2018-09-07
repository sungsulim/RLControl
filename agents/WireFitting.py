import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.running_mean_std import RunningMeanStd

from agents.base_agent import BaseAgent # for python3
#from base_agent import BaseAgent # for python2
import utils.plot_utils


class WireFittingNetwork(object):
    def __init__(self, state_dim, state_min, state_max, action_dim, action_min, action_max, config, random_seed):

        self.write_log = config.write_log
        self.write_plot = config.write_plot

        self.train_global_steps = 0
        self.eval_global_steps = 0
        self.train_ep_count = 0
        self.eval_ep_count = 0


        self.interplt_lr = config.interplt_lr
        #self.interim_NN_lc = params['interim_nn_lc']
        
        self.app_points = config.app_points
        self.n_h1 = config.l1_dim
        self.n_h2 = config.l2_dim
        self.smooth_eps = 0.00001
        self.tau = config.tau
        
        self.decay_rate = config.lr_decay_rate
        self.decay_step = config.lr_decay_step
        
        # self.adv_k = 1.0
        # self.n = 1

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = action_max

        self.dtype = tf.float64

        self.norm_type = config.norm
        if self.norm_type is not 'none':
            self.input_norm = RunningMeanStd(self.state_dim)
        else:
            assert(self.norm_type == 'none')
            self.input_norm = None


        self.g = tf.Graph()
        
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.is_training = tf.placeholder(tf.bool, [])
            self.state_input, self.interim_actions, self.interim_qvalues, self.max_q, self.bestact, self.tvars_in_actnn = self.create_act_q_nn("actqNN", self.state_dim, self.n_h1, self.n_h2)
            self.tar_state_input, self.tar_interim_actions, self.tar_interim_qvalues, self.tar_max_q, self.tar_bestact, self.tar_tvars_in_actnn = self.create_act_q_nn("target_actqNN", self.state_dim, self.n_h1, self.n_h2)

            self.action_input, self.qvalue, self.tvars_interplt = self.create_interpolation("interpolation", self.interim_actions, self.interim_qvalues, self.max_q)
            self.tar_action_input, self.tar_qvalue, self.tar_tvars_interplt = self.create_interpolation("target_interpolation", self.tar_interim_actions, self.tar_interim_qvalues, self.tar_max_q)

            # one list includes all vars
            self.tvars = self.tvars_in_actnn + self.tvars_interplt
            self.tar_tvars = self.tar_tvars_in_actnn + self.tar_tvars_interplt

            # define loss operation
            self.qtarget_input, self.interplt_loss = self.define_loss("losses")

            # define optimization
            self.global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.interplt_lr, self.global_step, self.decay_step, self.decay_rate, staircase=True)

            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="actqNN")
            # with tf.control_dependencies(update_ops):

            self.params_update = tf.train.AdamOptimizer(learning_rate).minimize(self.interplt_loss, global_step = self.global_step)
            self.step_add = tf.assign_add(self.global_step, 1)

            # update target network
            self.init_target = [tf.assign(self.tar_tvars[idx], self.tvars[idx]) for idx in range(len(self.tar_tvars))]
            self.update_target = [tf.assign_add(self.tar_tvars[idx], self.tau*(self.tvars[idx] - self.tar_tvars[idx])) for idx in range(len(self.tvars))]
            
            # init session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.init_target)

    def define_loss(self, scopename):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                qtargets = tf.placeholder(self.dtype, [None])
                # print 'qvalue shape is :: ', self.qvalue.shape
                interplt_loss = tf.losses.mean_squared_error(qtargets, self.qvalue)

        return qtargets, interplt_loss

    def create_act_q_nn(self, scopename, n_input, n_hidden1, n_hidden2):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                state_input = tf.placeholder(self.dtype, [None, n_input])
                if self.norm_type is not 'none':
                    state_input = self.input_norm.normalize(state_input)

                state_hidden1 = slim.fully_connected(state_input, n_hidden1, activation_fn = None)
                state_hidden1_norm = self.apply_norm(state_hidden1, activation_fn=tf.nn.relu, phase=True, layer_num=1) ## TODO: Hacky way. Not using batchnorm so just set phase to True. (Need to implement self.phase in the class)
                state_hidden2 = slim.fully_connected(state_hidden1_norm, n_hidden2, activation_fn = None)
                state_hidden2_norm = self.apply_norm(state_hidden2, activation_fn=tf.nn.relu, phase=True, layer_num=2)

                #state_hidden2_val = slim.fully_connected(state_hidden1, n_hidden1, activation_fn = tf.nn.relu)
                '''
                state_hidden1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(state_input, n_hidden1, activation_fn = None), center=True, scale=True, is_training=self.is_training))
                state_hidden2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(state_hidden1, n_hidden2, activation_fn = None), center=True, scale=True, is_training=self.is_training))
                '''
                w_init = tf.random_uniform_initializer(minval=-1., maxval=1.)
                interim_acts = slim.fully_connected(state_hidden2_norm, self.app_points * self.action_dim, activation_fn = tf.nn.tanh, weights_initializer=w_init) * self.action_max
                # print 'interim action shape is :: ', interim_acts.shape
                # w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
                interim_qvalues = slim.fully_connected(state_hidden2_norm, self.app_points, activation_fn = None, weights_initializer=w_init)
                # print 'interim q values shape is :: ', interim_qvalues.shape
                maxqvalue = tf.reduce_max(interim_qvalues, axis=1)

                # get best action
                maxind = tf.argmax(interim_qvalues, axis = 1)
                rowinds = tf.range(0, tf.cast(tf.shape(state_input)[0], tf.int64), 1)
                maxind_nd = tf.concat([tf.reshape(rowinds, [-1, 1]), tf.reshape(maxind, [-1, 1])], axis = 1)
                # print 'max id shape is :: ', maxind_nd.shape

                bestacts = tf.gather_nd(tf.reshape(interim_acts, [-1, self.app_points, self.action_dim]), maxind_nd)
                # get variables
                tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)

            return state_input, interim_acts, interim_qvalues, maxqvalue, bestacts, tvars

    def create_interpolation(self, scopename, interim_actions, interim_qvalues, max_q):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                action_input = tf.placeholder(self.dtype, [None, self.action_dim])
                tiled_action_input = tf.tile(action_input, [1, self.app_points])
                reshaped_action_input = tf.reshape(tiled_action_input, [-1, self.app_points, self.action_dim])
                # print 'reshaped tiled action shape is :: ', reshaped_action_input.shape
                reshaped_action_output = tf.reshape(interim_actions, [-1, self.app_points, self.action_dim])
                # distance is b * n mat, n is number of points to do interpolation
                act_distance = tf.reduce_sum(tf.square(reshaped_action_input - reshaped_action_output), axis = 2)
                w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
                smooth_c = tf.nn.sigmoid(tf.get_variable("smooth_c", [1, self.app_points], initializer=w_init, dtype=self.dtype))
                q_distance = smooth_c*(tf.reshape(max_q, [-1,1]) - interim_qvalues)
                distance = act_distance + q_distance + self.smooth_eps
                # distance = tf.add(distance, self.smooth_eps)
                weight = 1.0/distance
                # weight sum is a matrix b*1, b is batch size
                weightsum = tf.reduce_sum(weight, axis = 1, keep_dims=True)
                weight_final = weight/weightsum
                qvalue = tf.reduce_sum(tf.multiply(weight_final, interim_qvalues), axis = 1)
                tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
            return action_input, qvalue, tvars

    def create_interpolation_test(self, scopename):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                action_input = tf.placeholder(self.dtype, [None, self.action_dim])
                tiled_action_input = tf.tile(action_input, [1, self.app_points])
                reshaped_action_input = tf.reshape(tiled_action_input, [-1, self.app_points, self.action_dim])
                # print 'reshaped tiled action shape is :: ', reshaped_action_input.shape
                reshaped_action_output = tf.reshape(self.interim_actions, [-1, self.app_points, self.action_dim])
                # distance is b * n mat, n is number of points to do interpolation
                act_distance = tf.reduce_sum(tf.square(reshaped_action_input - reshaped_action_output), axis = 2)
                # w_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
                # smooth_c = tf.nn.sigmoid(tf.get_variable("smooth_c", [1, self.app_points], initializer=w_init, dtype=self.dtype))

                smooth_c = 0.1
                q_distance = smooth_c*(tf.reshape(self.max_q, [-1,1]) - self.interim_qvalues)
                distance = act_distance + q_distance + self.smooth_eps
                weight = 1.0/distance

                # weight sum is a matrix b*1, b is batch size
                weightsum = tf.reduce_sum(weight, axis = 1, keep_dims=True)
                weight_final = weight/weightsum
                qvalue = tf.reduce_sum(tf.multiply(weight_final, self.interim_qvalues), axis = 1)
            # tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)

            return action_input, reshaped_action_input, reshaped_action_output, q_distance, qvalue

    # currently a hack. Should later incorporate inheriting from base_network.apply_norm()
    def apply_norm(self, net, activation_fn, phase, layer_num):

        if self.norm_type == 'layer':
            norm_net = tf.contrib.layers.layer_norm(net, center=True, scale=True, activation_fn=activation_fn)
        elif self.norm_type == 'batch':
            norm_net = tf.contrib.layers.batch_norm(net, fused=True, center=True, scale=True, activation_fn=activation_fn,
                                                    is_training=phase, scope='batchnorm_'+str(layer_num))
        elif self.norm_type == 'none' or self.norm_type == 'input_norm':
            norm_net = activation_fn(net)
        else:
            raise ValueError('unknown norm type')

        return norm_net

    '''return an action to take for each state, NOTE this action is in [-1, 1]'''
    def take_action(self, state, is_train):

        # Very hacky
        if is_train:
            self.train_global_steps += 1
        else:
            self.eval_global_steps += 1
        bestact, acts = self.sess.run([self.bestact, self.interim_actions], {self.state_input: state.reshape(-1, self.state_dim), self.is_training: False})
        #print bestact.shape, acts.shape
        return bestact.reshape(-1), acts.reshape(self.app_points, self.action_dim)

    # similar to takeAction(), except this is for targets, returns QVal instead, and calculates in batches
    def computeQtargets(self, state, action, state_tp, reward, gamma):
        with self.g.as_default():
            Sp_qmax = self.sess.run(self.tar_max_q, {self.tar_state_input: state_tp, self.is_training: False})
            # this is a double Q DDQN learning rule
            # Sp_bestacts = self.sess.run(self.bestact, {self.state_input: state_tp})
            # Sp_qmax = self.sess.run(self.tar_qvalue, {self.tar_state_input: state_tp, self.tar_action_input: Sp_bestacts})
            qTargets = reward + gamma*np.squeeze(Sp_qmax)

            return qTargets
    
    def update_vars(self, state, action, state_tp, reward, gamma):
        with self.g.as_default():
            qtargets = self.computeQtargets(state, action, state_tp, reward, gamma)
            self.sess.run(self.step_add)
            self.sess.run(self.params_update, feed_dict = {self.state_input: state.reshape(-1, self.state_dim), self.action_input: action.reshape(-1, self.action_dim), self.qtarget_input: qtargets.reshape(-1), self.is_training: True})
            self.sess.run(self.update_target)

        #print gdstep
        return None
    '''
    def performtest(self, state, action, state_tp, reward, gamma):
        with self.g.as_default():
            qtargets = self.computeQtargets(state, action, state_tp, reward, gamma)
            test_a_input, reshaped_a_input, interimacts, test_a_output, q_distance, qvs = self.sess.run([self.test_a_input, self.reshaped_a_input, self.interim_actions, self.test_a_output, self.q_distance, self.qvs], feed_dict = {self.state_input: state.reshape(-1, self.stateDim), self.test_a_input: action.reshape(-1, self.actionDim)})
            print 'test a input is :: ', test_a_input
            print ' ---------------------------------- '
            print 'test a reshaped input is :: ', reshaped_a_input
            print ' ---------------------------------- '
            print 'interim actions :: ', interimacts
            print ' ---------------------------------- '
            print 'test a out:: ', test_a_output
            print ' ---------------------------------- '
            print 'q dist :: ', q_distance
            print ' ---------------------------------- '
            print 'qvs :: ', qvs
        #print gdstep
        return None
    '''
    def reset(self):
        pass

    def getQFunction(self, state):
        return lambda action: self.sess.run(self.qvalue, feed_dict={self.state_input: np.expand_dims(state, 0),
                                                                   self.action_input: np.expand_dims(action, 0),
                                                                   self.is_training: False})


class WireFitting(BaseAgent):
    def __init__(self, env, config, random_seed):
        super(WireFitting, self).__init__(env, config)


        np.random.seed(random_seed)  # Random action selection
        random.seed(random_seed)  # Experience Replay Buffer

        # state_dim, state_min, state_max, action_dim, action_min, action_max, config, random_seed

        self.network = WireFittingNetwork(self.state_dim, self.state_min, self.state_max,
                                          self.action_dim, self.action_min, self.action_max,
                                          config, random_seed)
        
        self.cum_steps = 0  # cumulative steps across episodes

    def start(self, state, is_train):
        return self.take_action(state, is_train, is_start=True)

    def step(self, state, is_train):
        # print self.next_action
        return self.take_action(state, is_train, is_start=False)

    def take_action(self, state, is_train, is_start):

        # random action during warmup
        if self.cum_steps < self.warmup_steps:
            action = np.random.uniform(self.action_min, self.action_max)

        else:
            action, allacts = self.network.take_action(state, is_train)

            # Train
            if is_train:
                greedy_action = action
                # if using an external exploration policy
                if self.use_external_exploration:
                    # print('greedy action', action)
                    action = self.exploration_policy.generate(greedy_action, self.cum_steps)
                    # print("noise added action", action)
                    # input()
                # only increment during training, not evaluation
                self.cum_steps += 1

                # HACKY WAY
                if self.write_plot:

                    if is_start:
                        self.network.train_ep_count += 1

                    func1 = self.network.getQFunction(state)

                    utils.plot_utils.plotFunction("WireFitting", [func1], state, greedy_action, action, self.action_min,
                                                  self.action_max,
                                                  display_title='ep: ' + str(
                                                      self.network.train_ep_count) + ', steps: ' + str(
                                                      self.network.train_global_steps),
                                                  save_title='steps_' + str(self.network.train_global_steps),
                                                  save_dir=self.writer.get_logdir(),
                                                  ep_count=self.network.train_ep_count,
                                                  show=False)

            action = np.clip(action, self.action_min, self.action_max)

        return action

    def update(self, state, next_state, reward, action, is_terminal, is_truncated):

        # Add to experience replay buffer
        if not is_truncated:
            if not is_terminal:
                self.replay_buffer.add(state, action, reward, next_state, self.gamma)
            else:
                self.replay_buffer.add(state, action, reward, next_state, 0.0)

        # TODO: WF has no normalization currently
        # # update running mean/std
        # if self.network.norm_type == 'layer' or self.network.norm_type == 'input_norm':
        #     self.network.input_norm.update(np.array([state]))

        if self.network.norm_type is not 'none':
            self.network.input_norm.update(np.array([state]))

        self.learn()

    def learn(self):

        if self.replay_buffer.get_size() > max(self.warmup_steps, self.batch_size):
            state, action, reward, next_state, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.network.update_vars(state, action, next_state, reward, gamma)
        else:
            return

    def reset(self):
        self.network.reset()
        if self.use_external_exploration:
            self.exploration_policy.reset()

