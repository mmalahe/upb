from baselines.common import tf_util
from baselines.ppo1 import mlp_policy
import pickle

class MLPPolicySaveable(mlp_policy):
    def __init__(self, name, *args, **kwargs):
        super(MLPPolicySaveable, self).__init__(name, *args, **kwargs)
        
    def save_state(self, filename):
        py_vars = {}
        for tf_var in self.get_trainable_variables():
            py_vars[tf_var.name] = tf_var.eval()
        with open(filename, 'wb') as f:
            pickle.dump(py_vars, f)
    
    def load_state(self, filename):
        with open(filename, 'rb') as f:
            py_vars = pickle.load(f)
        tf_util.initialize()
        for tf_var in self.get_trainable_variables():
            tf_var.load(py_vars[tf_var.name]) 
