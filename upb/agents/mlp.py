from baselines.common import tf_util
from baselines.ppo1.mlp_policy import MlpPolicy
import pickle
import numpy as np
import tensorflow as tf
import keras
import gym

class MLPAgent(MlpPolicy):
    def __init__(self, name, *args, **kwargs):
        super(MLPAgent, self).__init__(name, *args, **kwargs)
        
    def save(self, filename):
        py_vars = {}
        for tf_var in self.get_variables():
            py_vars[tf_var.name] = tf_var.eval()
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
                
    def load_and_check(self, filename):
        with open(filename, 'rb') as f:
            py_vars = pickle.load(f)
        tf_util.initialize()
        for tf_var in self.get_variables():
            tf_var.load(py_vars[tf_var.name])
            if not np.array_equal(py_vars[tf_var.name], tf_var.eval()):
                raise Exception("Variables not equal!")
                
    def check_is_same_as(self, filename):
        with open(filename, 'rb') as f:
            py_vars = pickle.load(f)
        for tf_var in self.get_variables():
            if not np.array_equal(py_vars[tf_var.name], tf_var.eval()):
                raise Exception("Variables not equal!")

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