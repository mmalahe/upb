from UPGameHandler import *
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from datetime import datetime
import time

# Stage 1
up_observation_names_stage1 = [
    'Unsold Inventory', 
    'Price per Clip', 
    'Public Demand', 
    'Available Funds', 
    'Autoclipper Cost', 
    'Autoclipper Purchasable', 
    'Number of Autoclippers',
    'Wire Inches',
    'Wire Cost'
]
up_action_names_stage1 = [
    'Make Paperclip', 
    'Lower Price', 
    'Raise Price', 
    'Buy Autoclipper',
    'Buy Wire']

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
    def __init__(self,
                 url, 
                 observation_names,                  
                 action_names,
                 episode_length=None,
                 desired_action_interval=0.2,
                 webdriver_name='Chrome',
                 webdriver_path=None,
                 headless=False,
                 verbose=False):
        
        # Set url where the game is hosted
        self._url = url
        
        # Fresh game handler
        self._handler = UPGameHandler(self._url, 
                                      webdriver_name=webdriver_name,
                                      webdriver_path=webdriver_path,
                                      headless=headless, 
                                      verbose=verbose)
        
        # Spaces
        self._observation_names = observation_names
        self._action_names = action_names
        
        # Other
        self._episode_length = episode_length
        self._desired_action_interval = desired_action_interval
        self._verbose = verbose
        
        # Reset
        self.reset()
    
    def _reset(self):
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
        self._prev_act_time = None
        self._n_steps_taken = 0
        
        # Return
        return observation
    
    def reward(self, observation_from_handler):
        return self.cashReward(observation_from_handler) + self.assetsReward(observation_from_handler)
    
    def assetsReward(self, observation_from_handler):
        wire_per_spool = 1000.0
        dwire = observation_from_handler['Wire Inches'] - self._prev_observation_from_handler['Wire Inches']
        dautoclippers = observation_from_handler['Number of Autoclippers'] - self._prev_observation_from_handler['Number of Autoclippers']
        wire_cost = self._prev_observation_from_handler['Wire Cost']/wire_per_spool
        autoclipper_cost = self._prev_observation_from_handler['Autoclipper Cost']
        dassets = dwire*wire_cost + dautoclippers*autoclipper_cost
        return dassets
    
    def cashReward(self, observation_from_handler):
        dcash = observation_from_handler['Available Funds'] - self._prev_observation_from_handler['Available Funds']
        return dcash
    
    def clipReward(self, observation_from_handler):
        dclips = observation_from_handler['Paperclips'] - self._prev_observation_from_handler['Paperclips']
        return dclips
    
    def _step(self, action):
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
        if self._prev_act_time != None:
            dt = timeSeconds() - self._prev_act_time
            time_remaining = self._desired_action_interval - dt
            if time_remaining > 0:
                time.sleep(time_remaining)
            else:
                print("WARNING: Took {:1.2g} s for step, which is more than the desired {:1.2g} s.".format(dt, self._desired_action_interval))
        
        # Act
        self._prev_act_time = timeSeconds();
        action_for_handler = self.action_space.actionAsString(action)        
        self._handler.takeAction(action_for_handler)
        
        # Observe
        observation_from_handler = self._handler.makeObservation(self.observation_space.getPossibleObservations())
        observation = self.observation_space.observationAsArray(observation_from_handler)
        
        # Get reward
        reward = self.reward(observation_from_handler)
        
        # Update any additional state
        self._prev_observation_from_handler = observation_from_handler    
        self._n_steps_taken += 1
        
        # Complete?
        if self._episode_length == None:
            done = False
        else:
            done = self._n_steps_taken >= self._episode_length
        
        # Additional info
        info = {}
        
        # Return
        return (observation, reward, done, info)
    
    def save_screenshot(self, filename):
        self._handler.save_screenshot(filename)
    
    def _close(self):
        self._handler.quit()
        
    def _render(self, mode='human', close=False):
        return
        
    def _seed(self, seed=None):
        return []
    
    @property
    def action_space(self):
        return UPActionSpace(self._action_names)

    @property
    def observation_space(self):
        return UPObservationSpace(self._observation_names)
