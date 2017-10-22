from baselines.common import tf_util
from baselines.ppo1.mlp_policy import MlpPolicy
import pickle
import tensorflow as tf
import numpy as np

class MlpPolicySaveable(MlpPolicy):
    def __init__(self, name, *args, **kwargs):
        super(MlpPolicySaveable, self).__init__(name, *args, **kwargs)
        
    def save_state(self, filename):
        py_vars = {}
        for tf_var in self.get_variables():
            py_vars[tf_var.name] = tf_var.eval()
        with open(filename, 'wb') as f:
            pickle.dump(py_vars, f)
    
    def load_state(self, filename):
        with open(filename, 'rb') as f:
            py_vars = pickle.load(f)
        tf_util.initialize()
        for tf_var in self.get_variables():
            tf_var.load(py_vars[tf_var.name]) 
            
    def save_reload_verify(self, filename):
        """Check that saving and loading restores the same state.
        """
        # Initial state of variables
        py_vars_out = {}
        for tf_var in self.get_variables():
            py_vars_out[tf_var.name] = tf_var.eval()
        
        # Save
        self.save_state(filename)
        
        # Clear
        for tf_var in self.get_variables():
            tf_var = tf.clip_by_value(tf_var, 0.0, 0.0)
            
        # Load and compare
        self.load_state(filename)
        for tf_var in self.get_variables():
            py_var_out = py_vars_out[tf_var.name]
            py_var_in = tf_var.eval()
            if not np.array_equal(py_var_out, py_var_in):
                print("{} out =".format(tf_var.name), py_var_out)
                print("{} in =".format(tf_var.name), py_var_in)
                raise Exception("Arrays not equal!")         
