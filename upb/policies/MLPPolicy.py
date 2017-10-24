from baselines.common import tf_util
from baselines.ppo1.mlp_policy import MlpPolicy
import pickle
import numpy as np
import keras
import gym

class MLPPolicy(object):
    def __init__(self, ob_space, ac_space, hidden_sizes, optimizer='adam', loss='kullback_leibler_divergence'):
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
        l_size.append(self._oblength)
        
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
        
        # Add layers with tanh activation except for softmax in final layer       
        self._model = keras.models.Sequential()
        n_layers = len(l_sizes)
        if n_layers == 2:
            self._model.add(keras.layers.Dense(l_sizes[1], activation='softmax', input_dim=l_sizes[0]))
        else:
            self._model.add(keras.layers.Dense(l_sizes[1], activation='tanh', input_dim=l_sizes[0]))
            for i in range(2, n_layers-1):
                self._model.add(keras.layers.Dense(l_sizes[i], activation='tanh'))
            self._model.add(keras.layers.Dense(l_sizes[i], activation='softmax'))
    
    def act(self, ob):
        if self._actype == 'discrete':            
            action_prob = self._model.predict(ob)
            return np.random.choice(action_prob, p=action_prob)
        elif self._actype == 'box':
            return self._model.predict(ob)
        else:
            raise NotImplementedError("No implementation for this type of space.")
