from baselines.common import tf_util
from baselines.ppo1.mlp_policy import MlpPolicy
import pickle
import numpy as np
import tensorflow as tf
import gym
from upb.envs.UPEnv import UPObservationSpace, UPActionSpace

class MLPAgent(MlpPolicy):
    def __init__(self, name, *args, **kwargs):
        super(MLPAgent, self).__init__(name, *args, **kwargs)
        self._hid_size = kwargs['hid_size']
        self._num_hid_layers = kwargs['num_hid_layers']
        self._name = name
        
    def save(self, filename):
        py_vars = {}
        for tf_var in self.get_variables():
            py_vars[tf_var.name] = tf_var.eval()
        py_vars['hid_size'] = self._hid_size
        py_vars['num_hid_layers'] = self._num_hid_layers
        with open(filename, 'wb') as f:
            pickle.dump(py_vars, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            py_vars = pickle.load(f)
        tf_util.initialize()
        for tf_var in self.get_variables():
            tf_var.load(py_vars[tf_var.name]) 
            
    def save_and_check_reload(self, filename):
        """Save and check that loading restores the same state.
        """
        # Initial state of variables
        py_vars_out = {}
        for tf_var in self.get_variables():
            py_vars_out[tf_var.name] = tf_var.eval()
        
        # Save
        self.save(filename)
        
        # Clear
        for tf_var in self.get_variables():
            tf_var = tf.clip_by_value(tf_var, 0.0, 0.0)
            
        # Load and compare
        self.load(filename)
        for tf_var in self.get_variables():
            py_var_out = py_vars_out[tf_var.name]
            py_var_in = tf_var.eval()
            if not np.array_equal(py_var_out, py_var_in):
                raise Exception("Variables not equal!")
                
    def load_and_check(self, filename, load_only_observation_scaling=False):
        with open(filename, 'rb') as f:
            py_vars = pickle.load(f)
        tf_util.initialize()
        for tf_var in self.get_variables():
            do_load = True
            if load_only_observation_scaling:
                if 'obfilter' not in tf_var.name:
                    do_load = False
            
            if do_load:
                # Strip the variable name of its scope
                var_name = tf_var.name
                end_scopename_idx = var_name.find("/")
                var_name_scopeless = var_name[end_scopename_idx+1:]
                
                # Find the name in the py dict, ignoring scope
                found_name = None
                for name in py_vars.keys():
                    if name.endswith(var_name_scopeless):
                        found_name = name
                        break
                if found_name == None:
                    raise Exception("Could not find {}.".format(var_name_scopless))
                
                # Load the variable                   
                tf_var.load(py_vars[found_name])
                if not np.array_equal(py_vars[found_name], tf_var.eval()):
                    raise Exception("Variables not equal!")
                
    def check_is_same_as(self, filename):
        with open(filename, 'rb') as f:
            py_vars = pickle.load(f)
        for tf_var in self.get_variables():
            if not np.array_equal(py_vars[tf_var.name], tf_var.eval()):
                raise Exception("Variables not equal!")
    
    def getVarNp(self, var_name, dtype=tf.float32):
        r"""Get variable as numpy array.
        """
        with tf.variable_scope(self.scope, reuse=True, dtype=dtype):
            weights_var = tf.get_variable(var_name)
            weights_np = weights_var.eval()
        return weights_np
    
    def getPolicyVars(self, i, kind):
        if i == self.num_hid_layers:
            return self.getVarNp("polfinal/{}".format(kind))
        else:
            return self.getVarNp("polfc{}/{}".format(i+1, kind))
    
    def getPolicyWeights(self, i):        
        r"""Get the weights of the policy.
        
        :param i: Which set of weights. i=0 => weights between input and first hidden layer, etc... 
        :type i: int
        :returns: weights -- A rank 2 numpy ndarray of the weights.
        """
        return self.getPolicyVars(i, 'w')
    
    def getPolicyBiases(self, i):
        return self.getPolicyVars(i, 'b')
        
    def getPolicyNetwork(self):
        """Return policy network as numpy arrays.
        """
        network = {'weights':[],'biases':[]}
        for i in range(self.num_hid_layers+1):
            network['weights'].append(self.getPolicyWeights(i))
            network['biases'].append(self.getPolicyBiases(i))
        return network
        
    def getObservationMeans(self):
        sums = self.getVarNp("obfilter/runningsum", dtype=tf.float64)
        counts = self.getVarNp("obfilter/count", dtype=tf.float64)
        means = sums/counts
        return means
        
    def getActionProbabilities(self, ob, ac_avail):
        with tf.variable_scope(self.scope):
            stochastic = True
            sequence_length = None
            ob_tfvar = tf_util.get_placeholder_cached(name=self.scope+"ob")
            ac_avail_tfvar = tf_util.get_placeholder_cached(name=self.scope+"acavail")
            logits = self.pd.logits.eval(feed_dict={ob_tfvar:ob[None], ac_avail_tfvar:ac_avail[None]})
            probs = np.exp(logits)/np.sum(np.exp(logits))
            return probs[0]
            
    @property
    def name(self):
        return self._name

def load_mlp_agent_topology(filename):
    with open(filename, 'rb') as f:
        py_vars = pickle.load(f)
    return py_vars['hid_size'], py_vars['num_hid_layers']
    
def load_mlp_agent(filename, agent_name, ob_space, ac_space):
    hid_size, num_hid_layers = load_mlp_agent_topology(filename)
    agent = MLPAgent(name=agent_name, ob_space=ob_space, 
                 ac_space=ac_space, hid_size=hid_size, num_hid_layers=num_hid_layers)
    agent.load_and_check(filename)
    return agent

class MultiStageAgent:
    def __init__(self, agents, initial_stage=0):
        self._agents = agents
        self._nstages = len(self._agents)
        self.stage = initial_stage
    
    @property
    def stage(self):
        return self._stage    
    
    @stage.setter
    def stage(self, value):
        if value < self._nstages:
            self._stage = value
        else:
            raise Exception("No stage {} for a {}-stage agent (zero indexed).".format(value, self._nstages))
    
    def act(self, ob):
        return self._agents[self._stage].act(ob)

class MLPAgentWIP(object):
    """With separate policy and value networks that have the same number
    of hidden layers.
    """
    def __init__(self, ob_space, ac_space, hidden_sizes):
        raise NotImplementedError()
        
        # Observation space type
        if isinstance(ob_space, gym.spaces.discrete.Discrete):
            self._obtype = 'discrete'
        elif isinstance(ob_space, gym.spaces.box.Box):
            self._obtype='box'
        else:
            raise NotImplementedError("No implementation for this type of space.")
        
        # Action space type
        if isinstance(ac_space, gym.spaces.discrete.Discrete):
            self._actype = 'discrete'
        elif isinstance(ac_space, gym.spaces.box.Box):
            self._actype = 'box'
        else:
            raise NotImplementedError("No implementation for this type of space.")
        
        # Layer sizes
        l_sizes = []
        
        # Input layer size
        if self._obtype == 'discrete':
            self._oblength = ob_space.n
        elif self._obtype == 'box':
            self._oblength = ob_space.shape[0]
        else:
            raise NotImplementedError("No implementation for this type of space.")
        l_sizes.append(self._oblength)
        
        # Hidden layers sizes
        for size in hidden_sizes:
            l_sizes.append(size)
        
        # Output layer size
        if self._actype == 'discrete':
            self._aclength = ac_space.n
        elif self._actype == 'box':
            self._aclength = ac_space.shape[0]
        else:
            raise NotImplementedError("No implementation for this type of space.")
        l_sizes.append(self._aclength)
        
        # Add layers with tanh activation except for 
        # - softmax in final layer for policy
        # - linear in final single-neuron layer for value function
        self._policy = keras.models.Sequential()
        self._value_fn = keras.models.Sequential()
        n_layers = len(l_sizes)
        if n_layers == 2:
            self._policy.add(keras.layers.Dense(l_sizes[1], activation='softmax', input_dim=l_sizes[0]))
            self._value_fn.add(keras.layers.Dense(1, activation='linear', input_dim=l_sizes[0]))
        else:
            self._policy.add(keras.layers.Dense(l_sizes[1], activation='tanh', input_dim=l_sizes[0]))
            self._value_fn.add(keras.layers.Dense(l_sizes[1], activation='tanh', input_dim=l_sizes[0]))
            for i in range(2, n_layers-1):
                self._policy.add(keras.layers.Dense(l_sizes[i], activation='tanh'))
                self._value_fn.add(keras.layers.Dense(l_sizes[i], activation='tanh'))
            self._policy.add(keras.layers.Dense(l_sizes[i], activation='softmax'))
            self._value_fn.add(keras.layers.Dense(1, activation='linear'))
        
        # Build models
        self._policy.build()
        self._value_fn.build()
        
        # Prepare prediction functions
        self._policy.model._make_predict_function()
        self._value_fn.model._make_predict_function()
    
    def get_action_and_value(self, ob):
        """Returns a chosen action and the value of the current state.
        """
        # Get action
        if self._actype == 'discrete':            
            action_prob = self._policy.predict(ob)
            action = np.random.choice(np.arange(self._aclength), p=action_prob)
        elif self._actype == 'box':
            action = self._policy.predict(ob)
        else:
            raise NotImplementedError("No implementation for this type of space.")
            
        # Get prediction
        value_prediction = self.get_value(ob)
        
        # Return
        return action, value_prediction
        
    @property
    def aclength(self):
        return self._aclength
    
    @property
    def oblength(self):
        return self._oblength
