from baselines.common import tf_util
from baselines.ppo1.mlp_policy import MlpPolicy
import pickle
import tensorflow as tf
import numpy as np
import keras

class MLPPolicySaveable(MlpPolicy):
    def __init__(self, name, *args, **kwargs):
        super(MLPPolicySaveable, self).__init__(name, *args, **kwargs)
        
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
            print("{} out =".format(tf_var.name), py_var_out)
            print("{} in =".format(tf_var.name), py_var_in)
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
            print("{} =".format(tf_var.name), tf_var.eval())
            print("{} from file =".format(tf_var.name), py_vars[tf_var.name])
            if not np.array_equal(py_vars[tf_var.name], tf_var.eval()):
                raise Exception("Variables not equal!")

class MLPPolicy(object):
    def __init__(self, ob_space, action_space, hidden_sizes, optimizer='adam', loss='kullback_leibler_divergence'):        
        # Initial model
        self._model = keras.models.Sequential()
        
        # Layer sizes
        l_sizes = []
        l_sizes.append(ob_space.shape)
        for size in hidden_sizes:
            l_sizes.append(size)
        l_sizes.append(action_space.shape)
        
        # Add first hidden layer
        self._model.add(Dense(l_sizes[1], activation='tanh', input_dim=l_sizes[0]))
        
        # Add subsequent layers        
        for i in range(2, len(l_sizes)):
            self._model.add(Dense(l_sizes[i], activation='tanh'))
            
        # Compile
        self._model.compile(optimizer=optimizer, loss=loss)
            
    def fit():
        pass
    
    def save():
        pass
        
    def load():
        pass
