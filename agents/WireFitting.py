from agents.agents import Agent # for python3
#from agents import Agent # for python2
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from collections import deque

from utils.replaybuffer import ReplayBuffer
import utils.exploration_policy

class wirefittingnn:
    def __init__(self, env, params, random_seed):
        
        self.interplt_lc = params['interplt_lc']
        #self.interim_NN_lc = params['interim_nn_lc']
        
        self.app_points = params['app_points']
        self.n_h1 = params['l1_dim']
        self.n_h2 = params['l2_dim']
        self.smooth_eps = 0.00001
        self.tau = params['tau']
        
        self.use_doubleQ = True
        
        self.decay_rate = params['lc_decay_rate']
        self.decay_step = params['lc_decay_step']
        
        # self.adv_k = 1.0
        # self.n = 1

        self.stateDim = env.stateDim # 2
        self.actionDim  = env.actionDim # 1
        self.actionBound = env.actionBound[0]

        self.dtype = tf.float64

        self.g = tf.Graph()
        
        with self.g.as_default():
            tf.set_random_seed(random_seed)
            self.is_training = tf.placeholder(tf.bool, [])
            self.state_input, self.interim_actions, self.interim_qvalues, self.max_q, self.bestact, self.tvars_in_actnn = self.create_act_q_nn("actqNN", self.stateDim, self.n_h1, self.n_h2)
            self.tar_state_input, self.tar_interim_actions, self.tar_interim_qvalues, self.tar_max_q, self.tar_bestact, self.tar_tvars_in_actnn = self.create_act_q_nn("target_actqNN", self.stateDim, self.n_h1, self.n_h2)
            self.action_input, self.qvalue, self.tvars_interplt = self.create_interpolation("interpolation", self.interim_actions, self.interim_qvalues, self.max_q)
            self.tar_action_input, self.tar_qvalue, self.tar_tvars_interplt = self.create_interpolation("target_interpolation", self.tar_interim_actions, self.tar_interim_qvalues, self.tar_max_q)
            #one list includes all vars
            self.tvars = self.tvars_in_actnn + self.tvars_interplt
            self.tar_tvars = self.tar_tvars_in_actnn + self.tar_tvars_interplt
            #define loss operation
            self.qtarget_input, self.interplt_loss = self.define_loss("losses")
            #define optimization
            self.global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.interplt_lc, self.global_step, self.decay_step, self.decay_rate, staircase=True)
            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="actqNN")
            #with tf.control_dependencies(update_ops):
            self.params_update = tf.train.AdamOptimizer(learning_rate).minimize(self.interplt_loss, global_step = self.global_step)
            self.step_add = tf.assign_add(self.global_step, 1)
            #update target network
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
                #print 'qvalue shape is :: ', self.qvalue.shape
                interplt_loss = tf.losses.mean_squared_error(qtargets, self.qvalue)
        return qtargets, interplt_loss

    def create_act_q_nn(self, scopename, n_input, n_hidden1, n_hidden2):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                state_input = tf.placeholder(self.dtype, [None, n_input])
                
                state_hidden1 = slim.fully_connected(state_input, n_hidden1, activation_fn = tf.nn.relu)
                state_hidden2 = slim.fully_connected(state_hidden1, n_hidden2, activation_fn = tf.nn.relu)
                
                #state_hidden2_val = slim.fully_connected(state_hidden1, n_hidden1, activation_fn = tf.nn.relu)
                '''
                state_hidden1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(state_input, n_hidden1, activation_fn = None), center=True, scale=True, is_training=self.is_training))
                state_hidden2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.contrib.layers.fully_connected(state_hidden1, n_hidden2, activation_fn = None), center=True, scale=True, is_training=self.is_training))
                '''
                w_init = tf.random_uniform_initializer(minval=-1., maxval=1.)
                interim_acts = slim.fully_connected(state_hidden2, self.app_points*self.actionDim, activation_fn = tf.nn.tanh, weights_initializer=w_init)*self.actionBound
                #print 'interim action shape is :: ', interim_acts.shape
                #w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
                interim_qvalues = slim.fully_connected(state_hidden2, self.app_points, activation_fn = None, weights_initializer=w_init)
                #print 'interim q values shape is :: ', interim_qvalues.shape
                maxqvalue = tf.reduce_max(interim_qvalues, axis=1)
                # get best action
                maxind = tf.argmax(interim_qvalues, axis = 1)
                rowinds = tf.range(0, tf.cast(tf.shape(state_input)[0], tf.int64), 1)
                maxind_nd = tf.concat([tf.reshape(rowinds, [-1, 1]), tf.reshape(maxind, [-1, 1])], axis = 1)
                #print 'max id shape is :: ', maxind_nd.shape
                bestacts = tf.gather_nd(tf.reshape(interim_acts, [-1, self.app_points, self.actionDim]), maxind_nd)
                #get variables
                tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
            return state_input, interim_acts, interim_qvalues, maxqvalue, bestacts, tvars

    def create_interpolation(self, scopename, interim_actions, interim_qvalues, max_q):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                action_input = tf.placeholder(self.dtype, [None, self.actionDim])
                tiled_action_input = tf.tile(action_input, [1, self.app_points])
                reshaped_action_input = tf.reshape(tiled_action_input, [-1, self.app_points, self.actionDim])
                #print 'reshaped tiled action shape is :: ', reshaped_action_input.shape
                reshaped_action_output = tf.reshape(interim_actions, [-1, self.app_points, self.actionDim])
                #distance is b * n mat, n is number of points to do interpolation
                act_distance = tf.reduce_sum(tf.square(reshaped_action_input - reshaped_action_output), axis = 2)
                w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
                smooth_c = tf.nn.sigmoid(tf.get_variable("smooth_c", [1, self.app_points], initializer=w_init, dtype=self.dtype))
                q_distance = smooth_c*(tf.reshape(max_q, [-1,1]) - interim_qvalues)
                distance = act_distance + q_distance + self.smooth_eps
                #distance = tf.add(distance, self.smooth_eps)
                weight = 1.0/distance
                #weight sum is a matrix b*1, b is batch size
                weightsum = tf.reduce_sum(weight, axis = 1, keep_dims=True)
                weight_final = weight/weightsum
                qvalue = tf.reduce_sum(tf.multiply(weight_final, interim_qvalues), axis = 1)
                tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
            return action_input, qvalue, tvars

    def create_interpolation_test(self, scopename):
        with self.g.as_default():
            with tf.variable_scope(scopename):
                action_input = tf.placeholder(self.dtype, [None, self.actionDim])
                tiled_action_input = tf.tile(action_input, [1, self.app_points])
                reshaped_action_input = tf.reshape(tiled_action_input, [-1, self.app_points, self.actionDim])
                #print 'reshaped tiled action shape is :: ', reshaped_action_input.shape
                reshaped_action_output = tf.reshape(self.interim_actions, [-1, self.app_points, self.actionDim])
                #distance is b * n mat, n is number of points to do interpolation
                act_distance = tf.reduce_sum(tf.square(reshaped_action_input - reshaped_action_output), axis = 2)
                w_init = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
                #smooth_c = tf.nn.sigmoid(tf.get_variable("smooth_c", [1, self.app_points], initializer=w_init, dtype=self.dtype))
                smooth_c = 0.1
                q_distance = smooth_c*(tf.reshape(self.max_q, [-1,1]) - self.interim_qvalues)
                distance = act_distance + q_distance + self.smooth_eps
                weight = 1.0/distance
                #weight sum is a matrix b*1, b is batch size
                weightsum = tf.reduce_sum(weight, axis = 1, keep_dims=True)
                weight_final = weight/weightsum
                qvalue = tf.reduce_sum(tf.multiply(weight_final, self.interim_qvalues), axis = 1)
            #tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scopename)
            return action_input, reshaped_action_input, reshaped_action_output, q_distance, qvalue

    '''return an action to take for each state, NOTE this action is in [-1, 1]'''
    def takeAction(self, state):
        bestact, acts = self.sess.run([self.bestact, self.interim_actions], {self.state_input: state.reshape(-1, self.stateDim), self.is_training: False})
        #print bestact.shape, acts.shape
        return bestact.reshape(-1), acts.reshape(self.app_points, self.actionDim)

    # similar to takeAction(), except this is for targets, returns QVal instead, and calculates in batches
    def computeQtargets(self, state, action, state_tp, reward, gamma):
        with self.g.as_default():
            Sp_qmax = self.sess.run(self.tar_max_q, {self.tar_state_input: state_tp, self.is_training: False})
            # this is a double Q DDQN learning rule
            #Sp_bestacts = self.sess.run(self.bestact, {self.state_input: state_tp})
            #Sp_qmax = self.sess.run(self.tar_qvalue, {self.tar_state_input: state_tp, self.tar_action_input: Sp_bestacts})
            qTargets = reward + gamma*np.squeeze(Sp_qmax)
            return qTargets
    
    def update_vars(self, state, action, state_tp, reward, gamma):
        with self.g.as_default():
            qtargets = self.computeQtargets(state, action, state_tp, reward, gamma)
            self.sess.run(self.step_add)
            self.sess.run(self.params_update, feed_dict = {self.state_input: state.reshape(-1, self.stateDim), self.action_input: action.reshape(-1, self.actionDim), self.qtarget_input: qtargets.reshape(-1), self.is_training: True})
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
        with self.g.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.init_target)

class Wire_fitting(Agent):
    def __init__(self, env, params, random_seed):
        super(Wire_fitting, self).__init__(env)

        np.random.seed(random_seed) # Random action selection
        random.seed(random_seed) # Experience Replay Buffer

        # self.epsilon = params['epsilon'] # 0.3
        # self.epsilon_decay = params['epsilon_decay'] # 0.9
        # self.epsilon_decay_step = params['epsilon_decay_step'] # 100


        self.policyfunc = wirefittingnn(env, params, random_seed)


        self.exploration_policy = utils.exploration_policy.OrnsteinUhlenbeckProcess(self.actionDim, params['ou_theta'],\
                                                            mu = params['ou_mu'], \
                                                            sigma = params['ou_sigma'])


        #self.bufferSize = params['buffer_size']
        #self.erbuffer = deque(maxlen = self.bufferSize)
        self.replay_buffer = ReplayBuffer(params['buffer_size'])
        self.batch_size = params['batch_size']

        self.gamma = params['gamma'] # 0.99

        self.warmup_steps = params['warmup_steps'] 

        self.action_is_greedy = None
        self.eps_decay = True
        
        # self.cum_steps = 0 # cumulative steps across episodes
    
        #print('agent params gamma, epsilon', self.gamma, self.epsilon)

    def update(self, S, Sp, r, a, episodeEnd = None):
        # self.cum_steps += 1
        # epsilon decay
        # if self.eps_decay and self.cum_steps % self.epsilon_decay_step == 0:
        #     self.epsilon *= self.epsilon_decay

        if not episodeEnd:
            self.replay_buffer.add(S, a, r, Sp, self.gamma)
            self.learn()
            #self.next_action = self.policy(Sp)
        else:
            self.replay_buffer.add(S, a, r, Sp, 0.0)
            self.learn()

    # more stable, converges faster
    # def sampleBatch(self):
    #     #startind = max(len(self.erbuffer)-self.recency, 0)
    #     randinds = np.random.randint(0, len(self.erbuffer)-1, self.batchSize)
    #     #print len(self.erbuffer)
    #     subset = [self.erbuffer[i] for i in randinds]
    #     subset[0] = self.erbuffer[len(self.erbuffer)-1]
    #     buff = np.array(subset)
    #     s, sp, r, a, gamma = [np.stack(buff[:,i]) for i in range(5)]
    #     #print s, sp, r
    #     return s, a, sp, np.squeeze(r), np.squeeze(gamma)
    
    def learn(self):
        # if len(self.erbuffer) < self.batchSize:
        #     return
        # s, a, sp, r, gamma = self.sampleBatch()

        if self.replay_buffer.getSize() > max(self.warmup_steps, self.batch_size):
            s, a, r, sp, gamma = self.replay_buffer.sample_batch(self.batch_size)
            self.policyfunc.update_vars(s, a, sp, r, gamma)
        else:
            return
        #print r
        #self.policyfunc.performtest(s, a, sp, r, gamma)

    def takeAction(self, state, isTrain):
        # epsilon greedy
        greedy_action, allacts = self.policyfunc.takeAction(state)
        # if np.random.uniform(0.,1.) < self.epsilon:
        #     self.action_is_greedy = False
        #     return np.random.uniform(-self.actionBound[0], self.actionBound[0], self.actionDim)
        #     #ind = np.random.randint(0, allacts.shape[0])
        #     #return allacts[ind,:]
        # else:
        #     self.action_is_greedy = True
        #     #self.noise_t += np.random.normal(np.zeros(self.actionDim), 0.2*np.ones(self.actionDim)) - self.noise_t*0.15
        #     #print bestact.shape, self.noise_t.shape
        #     #act =  bestact + self.noise_t
        #     return bestact

        if isTrain == 0:
            noise = self.exploration_policy.generate()

            action = np.clip( greedy_action + noise, self.actionMin, self.actionMax) 

            return action

        elif isTrain == 2:
            return np.clip( greedy_action, self.actionMin, self.actionMax) 

    def step(self, state, isTrain):
        #print self.next_action
        self.next_action = self.takeAction(state, isTrain)
        return self.next_action, None
    
    '''
    def getQfunction(self, state):
        return lambda action: self.policyfunc.sess.run(self.policyfunc.qvalue, {self.policyfunc.state_input: state.reshape(-1, self.policyfunc.stateDim), self.policyfunc.action_input: action.reshape(-1, self.policyfunc.actionDim), self.policyfunc.is_training:False})
    '''
    
    def start(self, obs, isTrain):
        self.next_action = self.takeAction(obs, isTrain)
        return self.next_action

    def reset(self):
        # self.erbuffer = [] # maybe do not reset erbuffer
        self.action_is_greedy = None
        self.exploration_policy.reset()
        # self.policyfunc.reset() # This shouldn't be reset!

        # set writer
    def setWriter(self, writer):
        self.policyfunc.writer = writer
