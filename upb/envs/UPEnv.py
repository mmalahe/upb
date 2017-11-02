from upb.game.UPGameHandler import *
from upb.emu.UPEmulator import *
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from datetime import datetime
import time
from collections import OrderedDict

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
        # Set up approximate ranges for possible observations
        self._observation_ranges = {
            'Unsold Inventory': [0, 1.0e3],
            'Price per Clip': [0, 1.0],
            'Public Demand': [0, 1.0e3],
            'Marketing Level': [0, 1.0e2],
            'Marketing Cost': [0, 1.0e2],
            'Manufacturing Clips per Second': [0, 1.0e4],
            'Wire Inches': [0, 1.0e4],
            'Wire Cost': [0, 1.0e2],
            'Number of Autoclippers': [0, 1.0e2],
            'Autoclipper Cost': [0, 1.0e4],
            'Paperclips': [0, 1.0e6],
            'Available Funds': [0, 1.0e6],
            'Processors': [0, 1.0e2],
            'Memory': [0, 1.0e2],
            'Trust': [0, 1.0e2],
            'Next Trust': [0, 1.0e6],
            'Operations': [0, 1.0e4],
            'Creativity': [0, 1.0e4],
            'Improved AutoClippers Activated': [0,1.],
            'Beg for More Wire Activated': [0,1.],
            'Creativity Activated': [0,1.],
            'Even Better AutoClippers Activated': [0,1.],
            'Optimized AutoClippers Activated': [0,1.],
            'Limerick Activated': [0,1.],
            'Improved Wire Extrusion Activated': [0,1.],
            'Optimized Wire Extrusion Activated': [0,1.],
            'Microlattice Shapecasting Activated': [0,1.],
            'New Slogan Activated': [0,1.],
            'Catchy Jingle Activated': [0,1.],
            'Lexical Processing Activated': [0,1.],
            'Combinatory Harmonics Activated': [0,1.],
            'The Hadwiger Problem Activated': [0,1.],
            'The Toth Sausage Conjecture Activated': [0,1.],
            'Hadwiger Clip Diagrams Activated': [0,1.],
            'Donkey Space Activated': [0,1.],
            'Algorithmic Trading Activated': [0,1.],
            'WireBuyer Activated': [0,1.],
            'Hypno Harmonics Activated': [0,1.],
            'RevTracker Activated': [0,1.]    
        }

        # Order keys
        self._keys = observation_names
        self._nkeys = len(self._keys)
        low = np.zeros(self._nkeys)
        high = np.zeros(self._nkeys)
        for i in range(self._nkeys):
            low[i] = self._observation_ranges[self._keys[i]][0]
            high[i] = self._observation_ranges[self._keys[i]][1]
            
        # Construct
        super(UPObservationSpace, self).__init__(low, high)
    
    def getPossibleObservations(self):
        return self._keys
        
    def observationAsArray(self, observation):
        obs_array = np.zeros(self._nkeys)
        for i in range(self._nkeys):
            obs_array[i] = observation[self._keys[i]]
            # Normalise
            obs_array[i] /= self._observation_ranges[self._keys[i]][1]
        return obs_array
        
    def observationAsString(self, obs_array):
        obs = ""
        for i in range(self._nkeys):
            obs += "{}={:1.2g},".format(self._keys[i], obs_array[i])
        return obs

class UPActionSpace(Discrete):
    def __init__(self, action_names):
        # Construct
        self._keys = action_names
        nkeys = len(self._keys)
        super(UPActionSpace, self).__init__(nkeys)
    
    # Converts action into a form that can be read by the game handler
    def actionAsString(self, action):
        return self._keys[action]

class UPEnv(Env):
    _observation_names_stages = []
    _action_names_stages = []
    
    # Stage 0
    _observation_names_stages.append(
    [
        'Unsold Inventory', 
        'Price per Clip', 
        'Public Demand', 
        'Available Funds', 
        'Autoclipper Cost', 
        'Number of Autoclippers',
        'Wire Inches',
        'Wire Cost'
    ])
    _action_names_stages.append(
    [
        'Make Paperclip', 
        'Lower Price', 
        'Raise Price', 
        'Buy Autoclipper',
        'Buy Wire'
    ])
    
    # Stage 1
    _observation_names_stages.append(
    _observation_names_stages[-1]+
    [
        'Paperclips',
        'Marketing Level',
        'Marketing Cost',
        'Processors',
        'Memory',
        'Trust',
        'Next Trust',
        'Operations',
        'Creativity',
        'Improved AutoClippers Activated',
        #~ 'Beg for More Wire Activated',
        'Creativity Activated',
        'Even Better AutoClippers Activated',
        'Optimized AutoClippers Activated',
        'Limerick Activated',
        'Improved Wire Extrusion Activated',
        'Optimized Wire Extrusion Activated',
        'New Slogan Activated',
        'Catchy Jingle Activated',
        'Lexical Processing Activated',
        'Combinatory Harmonics Activated',
        'The Hadwiger Problem Activated',
        'The Toth Sausage Conjecture Activated',
        'Donkey Space Activated',
        'RevTracker Activated'
    ])
    _action_names_stages.append(
    _action_names_stages[-1]+
    [
        'Expand Marketing',
        'Add Processor',
        'Add Memory',
        'Activate Improved AutoClippers',
        #~ 'Activate Beg for More Wire',
        'Activate Creativity',
        'Activate Even Better AutoClippers',
        'Activate Optimized AutoClippers',
        'Activate Limerick',
        'Activate Improved Wire Extrusion',
        'Activate Optimized Wire Extrusion',
        'Activate New Slogan',
        'Activate Catchy Jingle',
        'Activate Lexical Processing',
        'Activate Combinatory Harmonics',
        'Activate The Hadwiger Problem',
        'Activate The Toth Sausage Conjecture',
        'Activate Donkey Space',
        'Activate RevTracker',
    ])
    
    # Stage 2
    _observation_names_stages.append(
    _observation_names_stages[-1]+
    [
        'Microlattice Shapecasting Activated',
        'Hadwiger Clip Diagrams Activated',
        'Algorithmic Trading Activated',
        'WireBuyer Activated',
        'Hypno Harmonics Activated',
    ])
    _action_names_stages.append(
    _action_names_stages[-1]+
    [
        'Activate Microlattice Shapecasting',
        'Activate Hadwiger Clip Diagrams',
        'Activate Algorithmic Trading',
        'Activate WireBuyer',
        'Activate Hypno Harmonics',
    ])
    
    # Stage 3
    _observation_names_stages.append(
    _observation_names_stages[-1]+
    [
        'Algorithmic Trading Activated',
    ])
    _action_names_stages.append(
    _action_names_stages[-1]+
    [
        'Activate Algorithmic Trading',
    ])    
       
    def __init__(self,
                 url,
                 initial_stage=0,
                 final_stage=None,
                 resetter_agents=[],
                 use_emulator=False,
                 episode_length=None,
                 desired_action_interval=0.2,
                 webdriver_name='Chrome',
                 webdriver_path=None,
                 headless=False,
                 verbose=False):
        
        # Set url where the game is hosted
        self._url = url
        
        # Fresh game handler
        self._use_emulator = use_emulator
        if use_emulator:
            self._handler = UPEmulator()
        else:
            self._handler = UPGameHandler(self._url, 
                                          webdriver_name=webdriver_name,
                                          webdriver_path=webdriver_path,
                                          headless=headless, 
                                          verbose=verbose)
        
        # Other
        self._episode_length = episode_length
        self._desired_action_interval = desired_action_interval
        self._verbose = verbose
        
        # Parameters for working up to a given stage
        if initial_stage != len(resetter_agents):
            raise Exception("Incorrect number of resetter agents ({}) for desired initial stage ({}). Should be equal.".format(len(resetter_agents), initial_stage))
        self._initial_stage = initial_stage
        self._final_stage = final_stage
        self._resetter_agents = resetter_agents
        
        # Reset
        self.reset()
    
    def _update_stage(self):
        stage_changed = False
        
        if self._stage == self._final_stage:
            return stage_changed
        
        # Update rule for stage 0 -> 1: onset of trust
        if self._stage == 0:
            observation_from_handler = self._handler.makeObservation(['Paperclips'])
            clips = observation_from_handler['Paperclips']
            if clips >= 2000:
                self._stage = 1
                if self._verbose:
                    print("Advancing from stage 0 to stage 1.")
                stage_changed = True
                
        # Update rule for stage 1 -> 2   
        if self._stage == 1:
            observation_from_handler = self._handler.makeObservation(['Memory'])
            memory = observation_from_handler['Memory']
            if memory >= 6:
                self._stage = 2
                if self._verbose:
                    print("Advancing from stage 1 to stage 2.")
                stage_changed = True
        
        # Update rule for stage 2 -> 3
        if self._stage == 2:
            observation_from_handler = self._handler.makeObservation(['Memory'])
            memory = observation_from_handler['Memory']
            if memory >= 10:
                self._stage = 3
                if self._verbose:
                    print("Advancing from stage 2 to stage 3.")
                stage_changed = True
                
         # Update rule for stage 3 -> 4
        if self._stage == 3:
            # No definition for stage 4 yet
            pass   
        
        if stage_changed:
            self._observation_names = self._observation_names_stages[self._stage]
            self._action_names = self._action_names_stages[self._stage]
        
        return stage_changed
    
    def _advance_to_stage(self, target_stage, agents):
        # Check number of supplied agents
        if target_stage > len(agents):
            raise Exception("Insufficient number of resetter agents ({}) for target stage ({}).".format(len(agents), target_stage))
        
        # Initial values
        self._prev_act_time = None
        self._n_steps_taken = 0
        
        # Initial observation
        ob_space = UPObservationSpace(self._observation_names_stages[self._stage])
        observation_from_handler = self._handler.makeObservation(ob_space.getPossibleObservations())
        self._prev_observation_from_handler = observation_from_handler        
        ob = ob_space.observationAsArray(observation_from_handler)
        
        # Advance
        stochastic = True
        prev_stage = self._stage
        max_n_steps = 10000
        while self._stage < target_stage:
            # Act
            agent = agents[self._stage]    
            ac, vpred = agent.act(stochastic, ob)  
            ob, rew, done, info = self._step(ac, self._stage)
            
            # Restart if failed to get to next stage
            if self._n_steps_taken > max_n_steps:
                print("WARNING: Timed out attempting to reach next stage. Resetting and trying fresh.")
                self._n_steps_taken = 0
                self._prev_act_time = None
                self._handler.reset()
                self._stage = 0
                ob_space = UPObservationSpace(self._observation_names_stages[self._stage])
                observation_from_handler = self._handler.makeObservation(ob_space.getPossibleObservations())
                self._prev_observation_from_handler = observation_from_handler        
                ob = ob_space.observationAsArray(observation_from_handler)
            
            # Report
            if self._verbose and self._stage > prev_stage:
                print("Advanced to stage {} after {} steps.".format(self._stage, self._n_steps_taken))
        if self._verbose:
            print("Completed initial stage advancement.")
    
    def _reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        
        # Reset page
        self._handler.reset()
        
        # Set correct initial stage
        self._stage = 0
        self._update_stage()
        
        # Advance to initial stage
        if self._initial_stage != 0:
            self._advance_to_stage(self._initial_stage, self._resetter_agents)
        
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
        if self._stage == 0:
            return self.assetsAndCashReward(observation_from_handler)
        elif self._stage == 1:
            return self.assetsAndCashReward(observation_from_handler)
        else:
            raise NotImplementedError("No definition for stage 2+.")
            
    def assetsAndCashReward(self, observation_from_handler):
        return self.cashReward(observation_from_handler) + self.assetsReward(observation_from_handler)
    
    def assetsReward(self, observation_from_handler):        
        dassets = 0.0
        
        wire_per_spool = 1000.0
        dwire = observation_from_handler['Wire Inches'] - self._prev_observation_from_handler['Wire Inches']
        wire_cost = self._prev_observation_from_handler['Wire Cost']/wire_per_spool
        dassets += dwire*wire_cost
        
        dautoclippers = observation_from_handler['Number of Autoclippers'] - self._prev_observation_from_handler['Number of Autoclippers']
        autoclipper_cost = self._prev_observation_from_handler['Autoclipper Cost']        
        dassets += dautoclippers*autoclipper_cost 
        
        if self._stage > 0:
            try:
                dmarketing = observation_from_handler['Marketing Level'] - self._prev_observation_from_handler['Marketing Level']
                marketing_cost = self._prev_observation_from_handler['Marketing Cost']
                dassets += dmarketing*marketing_cost
            except:
                pass
        return dassets
    
    def cashReward(self, observation_from_handler):
        dcash = observation_from_handler['Available Funds'] - self._prev_observation_from_handler['Available Funds']
        return dcash
    
    def clipReward(self, observation_from_handler):
        dclips = observation_from_handler['Paperclips'] - self._prev_observation_from_handler['Paperclips']
        return dclips
    
    def _step(self, action, stage=None):
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
        if not self._use_emulator:       
            if self._prev_act_time != None:
                dt = timeSeconds() - self._prev_act_time
                time_remaining = self._desired_action_interval - dt
                if time_remaining > 0:
                    time.sleep(time_remaining)
                else:
                    print("WARNING: Took {:1.2g} s for step, which is more than the desired {:1.2g} s.".format(dt, self._desired_action_interval))
        
        # Act
        self._prev_act_time = timeSeconds();
        if stage == None:            
            action_for_handler = self.action_space.actionAsString(action)
        else:
            action_for_handler = UPActionSpace(self._action_names_stages[stage]).actionAsString(action) 
        self._handler.takeAction(action_for_handler)
        if self._verbose:
            print("Took action {}.".format(action_for_handler))
        
        # Advance time half way to resolve purchases, etc.
        if self._use_emulator:
            self._handler.advanceTime(self._desired_action_interval/2.0)
        
        # Update stage
        stage_changed = self._update_stage()
        
        # Observe
        if stage == None:
            observation_from_handler = self._handler.makeObservation(self.observation_space.getPossibleObservations())
            observation = self.observation_space.observationAsArray(observation_from_handler)
        else:
            ob_space = UPObservationSpace(self._observation_names_stages[stage])
            observation_from_handler = self._handler.makeObservation(ob_space.getPossibleObservations())
            observation = ob_space.observationAsArray(observation_from_handler)
        if self._verbose:
            print(observation_from_handler)
        
        # Get reward
        reward = self.reward(observation_from_handler)
        
        # Advance time the rest of the way
        if self._use_emulator:
            self._handler.advanceTime(self._desired_action_interval/2.0)
        
        # Update any additional state
        self._prev_observation_from_handler = observation_from_handler    
        self._n_steps_taken += 1
        
        # Complete?
        if self._episode_length == None:
            done = False
        else:
            done = self._n_steps_taken >= self._episode_length
        
        # Additional info
        info = {'stage': self._stage}
        
        # Return
        return (observation, reward, done, info)
    
    def save_screenshot(self, filename):
        if self._use_emulator:
            print("WARNING: Attempted to take screenshot, which emulator can't do. Printing state instead.")
            print(self._handler.makeObservation(self.observation_space.getPossibleObservations()))
        else:
            self._handler.save_screenshot(filename)
    
    def _close(self):
        self._handler.quit()
        
    def _render(self, mode='human', close=False):
        return
        
    def _seed(self, seed=None):
        return []
    
    @property
    def action_space(self):
        return UPActionSpace(self._action_names_stages[self._initial_stage])

    @property
    def observation_space(self):
        return UPObservationSpace(self._observation_names_stages[self._initial_stage])
