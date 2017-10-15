from UPGameHandler import *
from rllab.envs.base import Env
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
import numpy as np
from datetime import datetime
import time

# To microsecond precision
# @fixme Resets at the month boundary
def timeSeconds():
    dt = datetime.now()
    time_s = dt.day*86400.0
    time_s += dt.hour*3600.0
    time_s += dt.minute*60.0
    time_s += dt.second
    time_s += dt.microsecond/1000000.0;
    return time_s

class UPObservationSpace(Box):
    def __init__(self, observation_names):
        # Set up possible observations and their bounds
        self._all_observations = {
            'Unsold Inventory': [0, np.inf],
            'Price per Clip': [0, np.inf],
            'Public Demand': [0, np.inf],
            'Marketing Level': [0, np.inf],
            'Marketing Cost': [0, np.inf],
            'Manufacturing Clips per Second': [0, np.inf],
            'Wire Inches': [0, np.inf],
            'Wire Cost': [0, np.inf],
            'Number of Autoclippers': [0, np.inf],
            'Autoclipper Cost': [0, np.inf],
            'Autoclipper Purchasable': [0, 1],
            'Paperclips': [0, np.inf],
            'Available Funds': [0, np.inf]           
        }
        self._observations = {key: self._all_observations[key] for key in observation_names}
        self._keys = list(self._observations.keys())
        self._nkeys = len(self._keys)
        low = np.zeros(self._nkeys)
        high = np.zeros(self._nkeys)
        for i in range(self._nkeys):
            low[i] = self._observations[self._keys[i]][0]
            high[i] = self._observations[self._keys[i]][1]
            
        # Construct
        super(UPObservationSpace, self).__init__(low, high)
    
    def getPossibleObservations(self):
        return self._keys
        
    def observationAsArray(self, observation):
        obs_array = np.zeros(self._nkeys)
        for i in range(self._nkeys):
            obs_array[i] = observation[self._keys[i]]
        return obs_array

class UPActionSpace(Discrete):
    def __init__(self, action_names):
        # Set up possible actions
        self._all_actions = {
            'Make Paperclip': '',
            'Lower Price': '',
            'Raise Price': '',
            'Expand Marketing': '',
            'Buy Wire': '',
            'Buy Autoclipper': ''
        }
        self._actions = {key: self._all_actions[key] for key in action_names}
        self._keys = list(self._actions.keys())
        nkeys = len(self._keys)
            
        # Construct
        super(UPActionSpace, self).__init__(nkeys)
    
    # Converts action into a form that can be read by the game handler
    def actionAsString(self, action):
        return self._keys[action]

class UPEnv(Env):
    def __init__(self, url, observation_names, action_names, verbose=False):
        # Call base class constructor
        super(UPEnv, self).__init__()
        
        # Set url where the game is hosted
        self._url = url
        
        # Fresh game handler
        self._handler = UPGameHandler(self._url, verbose)
        
        # Action interval
        self._min_action_interval_s = 0.01
        
        # Spaces
        self._observation_names = observation_names
        self._action_names = action_names
        
        # Reset
        self.reset()
    
    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        
        # Reset page
        self._handler.reset()
        
        # Observe
        observation_from_handler = self._handler.makeObservation(self.observation_space.getPossibleObservations())
        observation = self.observation_space.observationAsArray(observation_from_handler)
        
        # Initial values
        self._prev_observation_from_handler = observation_from_handler
        self._prev_step_time = timeSeconds();
        
        # Return
        return observation
    
    def reward(self, observation_from_handler):
        return self.cashRateReward(observation_from_handler)
        #~ return self.clipRateReward(observation_from_handler)
    
    def cashRateReward(self, observation_from_handler):
        dcash = observation_from_handler['Available Funds'] - self._prev_observation_from_handler['Available Funds']
        return dcash
    
    def clipRateReward(self, observation_from_handler):
        dclips = observation_from_handler['Paperclips'] - self._prev_observation_from_handler['Paperclips']
        return dclips
    
    def getDt(self):
        return self._step_time - self._prev_step_time
    
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        # Wall clock action rate control
        self._step_time = timeSeconds();
        dt = self._step_time - self._prev_step_time
        print(dt)
        if self._min_action_interval_s - dt > 0:
            time.sleep(self._min_action_interval_s - dt)
        
        # Act
        action_for_handler = self.action_space.actionAsString(action)
        self._handler.takeAction(action_for_handler)
        
        # Observe
        observation_from_handler = self._handler.makeObservation(self.observation_space.getPossibleObservations())
        observation = self.observation_space.observationAsArray(observation_from_handler)
        
        # Get reward
        reward = self.reward(observation_from_handler)
        
        # Complete?
        done = False
        
        # Additional info
        info = {}
        
        # Update any additional state
        self._prev_observation_from_handler = observation_from_handler
        self._prev_step_time = self._step_time
        
        # Return
        return (observation, reward, done, info)

    @property
    def action_space(self):
        return UPActionSpace(self._action_names)

    @property
    def observation_space(self):
        return UPObservationSpace(self._observation_names)
