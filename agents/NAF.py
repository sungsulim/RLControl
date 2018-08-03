from agents.base_agent import BaseAgent # for python3
#from base_agent import BaseAgent # for python2

import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils.running_mean_std import RunningMeanStd
from experiment import write_summary

class NAF_Network:
    def __init__(self, state_dim, state_min, state_max, action_dim, action_min, action_max, use_external_exploration, config, random_seed):

        self.write_log = config.write_log

        self.state_dim = state_dim
        self.state_min = state_min
        self.state_max = state_max

        self.action_dim  = action_dim
        self.action_min = action_min
        self.action_max = action_max

        self.use_external_exploration = use_external_exploration
        # type of normalization: 'none', 'batch', 'layer', 'input_norm'
        self.norm_type = config.norm

        if self.norm_type == 'input_norm' or self.norm_type == 'layer' or self.norm_type == 'batch':
            self.input_norm = RunningMeanStd(self.state_dim)
        else:
            assert(self.norm_type == 'none')
            self.input_norm = None

        self.network_layer_dim = [config.l1_dim, config.l2_dim]

        self.learning_rate = config.learning_rate
        self.tau = config.tau
        self.noise_scale = config.noise_scale

        #record step n for tf Summary
        self.train_global_steps = 0
        self.eval_global_steps = 0
        

        self.dtype = tf.float32

        self.g = tf.Graph()
        
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.phase = tf.placeholder(tf.bool, [])
            self.state_input, self.action_input, self.q_val, self.max_q, self.best_action, self.Lmat_columns, self.tvars = self.build_network("naf")
            self.target_state_input, self.target_action_input, self.target_q_val, self.target_max_q, self.target_best_action, _, self.target_tvars = self.build_network("target_naf")
            

            # Batchnorm Ops and Vars
            if self.norm_type == 'batch':
            
                self.batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='naf/batchnorm')
                self.target_batchnorm_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_naf/batchnorm')

                self.batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='naf/batchnorm')
                self.target_batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='target_naf/batchnorm')

                self.update_target_batchnorm_params = [tf.assign(self.target_batchnorm_vars[idx],
                                                       self.batchnorm_vars[idx]) for idx in range(len(self.target_batchnorm_vars))
                                                       if self.target_batchnorm_vars[idx].name.endswith('moving_mean:0')
                                                       or self.target_batchnorm_vars[idx].name.endswith('moving_variance:0')]

            else:
                assert (self.norm_type == 'none' or self.norm_type == 'layer' or self.norm_type == 'input_norm')
                self.batchnorm_ops = [tf.no_op()]
                self.update_target_batchnorm_params = tf.no_op()

            # define loss and update operation
            with tf.control_dependencies(self.batchnorm_ops):

                self.target_q_input = tf.placeholder(self.dtype, [None])
                self.loss = tf.reduce_sum(tf.square(self.target_q_input - tf.squeeze(self.q_val)))
                self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            # update target network
            self.init_target = [tf.assign(self.target_tvars[idx], self.tvars[idx]) for idx in range(len(self.target_tvars))]
            self.update_target = [tf.assign_add(self.target_tvars[idx], self.tau * (self.tvars[idx] - self.target_tvars[idx])) for idx in range(len(self.tvars))]
            
            # init session
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.init_target)

    def build_network(self, scopename):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                state_input = tf.placeholder(self.dtype, [None, self.state_dim])
                action_input = tf.placeholder(self.dtype, [None, self.action_dim])
                
                # normalize state inputs if using "input_norm" or "layer" or "batch"
                if self.norm_type == 'input_norm' or self.norm_type == 'layer' or self.norm_type == 'batch':
                    state_input = tf.clip_by_value(self.input_norm.normalize(state_input), self.state_min, self.state_max)

                # layer norm network
                if self.norm_type == 'layer':
                    q_value, max_q, action, Lmat_columns = self.layer_norm_network(state_input, action_input)

                # batch norm network
                elif self.norm_type == 'batch':
                    q_value, max_q, action, Lmat_columns = self.batch_norm_network(state_input, action_input)

                # no norm network
                else:
                    assert(self.norm_type == 'none' or self.norm_type == 'input_norm')
                    q_value, max_q, action, Lmat_columns = self.no_norm_network(state_input, action_input)

            # get variables
            tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
            return state_input, action_input, q_value, max_q, action, Lmat_columns, tvars

    def layer_norm_network(self, state_input, action_input):

        state_hidden1 = slim.fully_connected(state_input, self.network_layer_dim[0], activation_fn = None)
        state_hidden1_layernorm1 = tf.contrib.layers.layer_norm(state_hidden1, center=True, scale=True, activation_fn=tf.nn.relu)

        # three branch: first output action
        action_hidden2 = slim.fully_connected(state_hidden1_layernorm1, self.network_layer_dim[1], activation_fn = None)
        action_hidden2_layernorm1 = tf.contrib.layers.layer_norm(action_hidden2, center=True, scale=True, activation_fn=tf.nn.relu)

        action = slim.fully_connected(action_hidden2_layernorm1, self.action_dim, activation_fn = tf.nn.tanh)*self.action_max


        # value branch
        value_hidden = slim.fully_connected(state_hidden1_layernorm1, self.network_layer_dim[1], activation_fn = None)
        value_hidden_layernorm1 = tf.contrib.layers.layer_norm(value_hidden, center=True, scale=True, activation_fn=tf.nn.relu)

        value = slim.fully_connected(value_hidden_layernorm1, 1, activation_fn = None,
                                     weights_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
        
        # Lmat branch
        act_mu_diff = action_input - action
        
        # Lmat_flattened = slim.fully_connected(state_hidden1_layernorm1, (1+self.action_dim)*self.action_dim/2, activation_fn = None)
        Lmat_diag = [tf.exp(slim.fully_connected(state_hidden1_layernorm1, 1, activation_fn = None)) for _ in range(self.action_dim)]
        Lmat_nondiag = [slim.fully_connected(state_hidden1_layernorm1, k-1, activation_fn = None) for k in range(self.action_dim, 1, -1)]

        # in Lmat_columns, if actdim = 1, first part is empty
        Lmat_columns = [tf.concat((Lmat_diag[id], Lmat_nondiag[id]),axis=1) for id in range(len(Lmat_nondiag))] + [Lmat_diag[-1]]
        act_mu_diff_Lmat_prod = [tf.reduce_sum(tf.slice(act_mu_diff,[0,cid],[-1,-1])*Lmat_columns[cid], axis=1, keep_dims=True) for cid in range(len(Lmat_columns))]
        # prod_tensor should be dim: batchsize*action_dim
        prod_tensor = tf.concat(act_mu_diff_Lmat_prod, axis = 1)
        # print 'prod tensor shape is :: ', prod_tensor.shape
        adv_value = -0.5 * tf.reduce_sum(prod_tensor*prod_tensor, axis = 1, keep_dims=True)
        q_value = value + adv_value
        max_q = value
        
        return q_value, max_q, action, Lmat_columns

    def batch_norm_network(self, state_input, action_input):

        state_hidden1 = slim.fully_connected(state_input, self.network_layer_dim[0], activation_fn = None)

        state_hidden1_batchnorm1 = tf.contrib.layers.batch_norm(state_hidden1, fused=True, center=True, scale=True, activation_fn=tf.nn.relu, \
                                            is_training=self.phase, scope='batchnorm_1')


        # three branch: first output action
        action_hidden2 = slim.fully_connected(state_hidden1_batchnorm1, self.network_layer_dim[1], activation_fn = None)
        action_hidden2_batchnorm1 = tf.contrib.layers.batch_norm(action_hidden2, fused=True, center=True, scale=True, activation_fn=tf.nn.relu, \
                                            is_training=self.phase, scope='batchnorm_2')

        action = slim.fully_connected(action_hidden2_batchnorm1, self.action_dim, activation_fn = tf.nn.tanh)*self.action_max


        # value branch
        value_hidden = slim.fully_connected(state_hidden1_batchnorm1, self.network_layer_dim[1], activation_fn = None)
        value_hidden_batchnorm1 = tf.contrib.layers.batch_norm(value_hidden, fused=True, center=True, scale=True, activation_fn=tf.nn.relu, \
                                            is_training=self.phase, scope='batchnorm_3')

        value = slim.fully_connected(value_hidden_batchnorm1, 1, activation_fn = None,
                                                        weights_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
        
        # Lmat branch
        act_mu_diff = action_input - action
        
        #Lmat_flattened = slim.fully_connected(state_hidden1_batchnorm1, (1+self.action_dim)*self.action_dim/2, activation_fn = None)
        Lmat_diag = [tf.exp(slim.fully_connected(state_hidden1_batchnorm1, 1, activation_fn = None)) for _ in range(self.action_dim)]
        Lmat_nondiag = [slim.fully_connected(state_hidden1_batchnorm1, k-1, activation_fn = None) for k in range(self.action_dim, 1, -1)]

        #in Lmat_columns, if actdim = 1, first part is empty
        Lmat_columns = [tf.concat((Lmat_diag[id], Lmat_nondiag[id]),axis=1) for id in range(len(Lmat_nondiag))] + [Lmat_diag[-1]]
        act_mu_diff_Lmat_prod = [tf.reduce_sum(tf.slice(act_mu_diff,[0,cid],[-1,-1])*Lmat_columns[cid], axis=1, keep_dims=True) for cid in range(len(Lmat_columns))]
        #prod_tensor should be dim: batchsize*action_dim
        prod_tensor = tf.concat(act_mu_diff_Lmat_prod, axis = 1)
        # print 'prod tensor shape is :: ', prod_tensor.shape
        adv_value = -0.5 * tf.reduce_sum(prod_tensor*prod_tensor, axis = 1, keep_dims=True)
        q_value = value + adv_value
        max_q = value
        
        return q_value, max_q, action, Lmat_columns

    def no_norm_network(self, state_input, action_input):

        state_hidden1 = slim.fully_connected(state_input, self.network_layer_dim[0], activation_fn = tf.nn.relu)
        #three branch: first output action
        action_hidden2 = slim.fully_connected(state_hidden1, self.network_layer_dim[1], activation_fn = tf.nn.relu)
        action = slim.fully_connected(action_hidden2, self.action_dim, activation_fn = tf.nn.tanh)*self.action_max
        #value branch
        value_hidden = slim.fully_connected(state_hidden1, self.network_layer_dim[1], activation_fn = tf.nn.relu)
        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        value = slim.fully_connected(value_hidden, 1, activation_fn = None, weights_initializer=w_init)
        
        #Lmat branch
        act_mu_diff = action_input - action
        #Lmat_flattened = slim.fully_connected(state_hidden1, (1+self.action_dim)*self.action_dim/2, activation_fn = None)
        Lmat_diag = [tf.exp(slim.fully_connected(state_hidden1, 1, activation_fn = None)) for _ in range(self.action_dim)]
        Lmat_nondiag = [slim.fully_connected(state_hidden1, k-1, activation_fn = None) for k in range(self.action_dim, 1, -1)]
        #in Lmat_columns, if actdim = 1, first part is empty
        Lmat_columns = [tf.concat((Lmat_diag[id], Lmat_nondiag[id]),axis=1) for id in range(len(Lmat_nondiag))] + [Lmat_diag[-1]]
        act_mu_diff_Lmat_prod = [tf.reduce_sum(tf.slice(act_mu_diff,[0,cid],[-1,-1])*Lmat_columns[cid], axis=1, keep_dims=True) for cid in range(len(Lmat_columns))]
        #prod_tensor should be dim: batchsize*action_dim
        prod_tensor = tf.concat(act_mu_diff_Lmat_prod, axis = 1)
        # print 'prod tensor shape is :: ', prod_tensor.shape
        adv_value = -0.5 * tf.reduce_sum(prod_tensor*prod_tensor, axis = 1, keep_dims=True)
        q_value = value + adv_value
        max_q = value
                
        return q_value, max_q, action, Lmat_columns

    '''return an action to take for each state'''
    def take_action(self, state, is_train):

        best_action, Lmat_columns = self.sess.run([self.best_action, self.Lmat_columns], {self.state_input: state.reshape(-1, self.state_dim), self.phase: False})
        
        # train
        if is_train:

            # just return the best action
            if self.use_external_exploration:
                chosen_action = np.clip(best_action.reshape(-1), self.action_min, self.action_max)

            else:
                #compute covariance matrix
                Lmat = np.zeros((self.action_dim, self.action_dim))
                #print 'the Lmat columns are --------------------------------------------- '
                for i in range(self.action_dim):
                    #print Lmat_columns[i]
                    Lmat[i:, i] = np.squeeze(Lmat_columns[i])
                try:
                    covmat = self.noise_scale * np.linalg.pinv(Lmat.dot(Lmat.T))
                except:
                    print('Lmat', Lmat)
                    print('\nLmat^2', Lmat.dot(Lmat.T))
                    exit()

                sampled_act = np.random.multivariate_normal(best_action.reshape(-1), covmat)
                #print 'sampled act is ------------------ ', sampled_act
                
                # if self.n % 1000 == 0:
                #     print 'covmat is :: '
                #     print covmat
                #     print 'Lmat is :: '
                #     print Lmat
                # print('train', sampled_act)
                chosen_action = np.clip(sampled_act, self.action_min, self.action_max)

                if self.write_log:
                    self.train_global_steps += 1
                    write_summary(self.writer, self.train_global_steps, covmat[0][0], tag='train/covmat00') # logging only top-left element!

            if self.write_log:
                write_summary(self.writer, self.train_global_steps, chosen_action[0], tag='train/action_taken')

            return chosen_action

        # eval
        else:

            chosen_action = np.clip(best_action.reshape(-1), self.action_min, self.action_max)

            if self.write_log:
                self.eval_global_steps += 1
                write_summary(self.writer, self.eval_global_steps, chosen_action[0], tag='eval/action_taken')

            return chosen_action

    # similar to take_action(), except this is for targets, returns QVal instead, and calculates in batches
    def compute_target_Q(self, state, action, next_state, reward, gamma):
        with self.g.as_default():
            Sp_qmax = self.sess.run(self.target_max_q, {self.target_state_input: next_state, self.phase: True})
            # this is a double Q DDQN learning rule
            #Sp_bestacts = self.sess.run(self.bestact, {self.state_input: next_state})
            #Sp_qmax = self.sess.run(self.target_q_val, {self.target_state_input: next_state, self.target_action_input: Sp_bestacts})
            qTargets = reward + gamma * np.squeeze(Sp_qmax)
            return qTargets
    
    def update(self, state, action, next_state, reward, gamma):
        with self.g.as_default():
            target_q = self.compute_target_Q(state, action, next_state, reward, gamma)
            self.sess.run(self.optimize, feed_dict = {self.state_input: state.reshape(-1, self.state_dim), self.action_input: action.reshape(-1, self.action_dim), self.target_q_input: target_q.reshape(-1), self.phase: True})
            self.sess.run( [self.update_target, self.update_target_batchnorm_params])
        return None

    def reset(self):
        pass

class NAF(BaseAgent):
    def __init__(self, env, config, random_seed):
        super(NAF, self).__init__(env, config)

        np.random.seed(random_seed)
        random.seed(random_seed)

        self.network = NAF_Network(self.state_dim, self.state_min, self.state_max,
                                   self.action_dim, self.action_min, self.action_max,
                                   self.use_external_exploration, config, random_seed=random_seed)

        self.cum_steps = 0  # cumulative steps across episodes
    
    def start(self, state, is_train):
        return self.take_action(state, is_train)

    def step(self, state, is_train):
        return self.take_action(state, is_train)

    def take_action(self, state, is_train):

        # random action during warmup
        if self.cum_steps < self.warmup_steps:
            action = np.random.uniform(self.action_min, self.action_max)

        else:
            action = self.network.take_action(state, is_train)

            # Train
            if is_train:

                # if using an external exploration policy
                if self.use_external_exploration:
                    action = self.exploration_policy.generate(action, self.cum_steps)

                # only increment during training, not evaluation
                self.cum_steps += 1

            action = np.clip(action, self.action_min, self.action_max) 
        return action

    def update(self, state, next_state, reward, action, is_terminal):

        if not is_terminal:
            self.replay_buffer.add(state, action, reward, next_state, self.gamma)
        else:
            self.replay_buffer.add(state, action, reward, next_state, 0.0)

        if self.network.norm_type == 'input_norm' or self.network.norm_type == 'layer':
            self.network.input_norm.update(np.array([state]))
        self.learn()

    def learn(self):

        if self.replay_buffer.get_size() > max(self.warmup_steps, self.batch_size):
            state, action, reward, next_state, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.network.update(state, action, next_state, reward, gamma)
        else:
            return

    def reset(self):
        self.network.reset()
        if self.exploration_policy:
            self.exploration_policy.reset()




