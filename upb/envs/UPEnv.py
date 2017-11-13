from upb.game.UPGameHandler import *
from upb.emu.UPEmulator import *
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from datetime import datetime
import time
from collections import OrderedDict
import pickle

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
            'Investment Bankroll': [0, 1e6],
            'Stocks': [0, 1e6],
            'Investment Engine Level': [0, 10],
            'Investment Engine Upgrade Cost': [0, 1e4],
            'MegaClipper Cost': [500.0, 1e6],
            'Number of MegaClippers': [0, 1.0e2],
            'Riskiness': [1., 7.],
            'Number of Photonic Chips': [0, 10.],
            'Latest QOps': [0,1e4],
            'Photonic Chip 0 Level': [-1., 1.],
            'Yomi': [0, 1e5],
            'Tournament Cost': [0, 1e5],
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
            'RevTracker Activated': [0,1.],
            'Quantum Computing Activated': [0,1.],
            'Spectral Froth Annealment Activated': [0,1.],
            'MegaClippers Activated': [0,1.],
            'Strategic Modeling Activated': [0,1.]
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
        
    def observationAsOrderedDict(self, obs_array):
        obs_dict = OrderedDict()
        for i in range(self._nkeys):
            obs_dict[self._keys[i]] = obs_array[i]
            # Denormalise
            obs_dict[self._keys[i]] *= self._observation_ranges[self._keys[i]][1]
        return obs_dict
        
    def observationAsString(self, obs_array):
        obs = ""
        for i in range(self._nkeys):
            obs += "{}={:1.2g},".format(self._keys[i], obs_array[i]*self._observation_ranges[self._keys[i]][1])
        return obs
        
    def getObservationName(self, i):
        return self._keys[i]

class UPActionSpace(Discrete):
    def __init__(self, action_names):
        # Construct
        self._keys = action_names
        nkeys = len(self._keys)
        super(UPActionSpace, self).__init__(nkeys)
    
    # Converts action into a form that can be read by the game handler
    def actionAsString(self, action):
        return self._keys[action]
        
    def allActionsAsStrings(self):
        return self._keys.copy()

class UPEnv(Env):
    _observation_names_stages = []
    _action_names_stages = []
    _action_intervals_stages = []
    
    # Stage 0
    _action_intervals_stages.append(0.2)
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
    
    # Observations and actions common to stages 1-3
    _core_observation_set_1 = [
        'Unsold Inventory', 
        'Price per Clip', 
        'Public Demand', 
        'Available Funds', 
        'Autoclipper Cost', 
        'Number of Autoclippers',
        'Wire Inches',
        'Wire Cost',
        'Paperclips',
        'Marketing Level',
        'Marketing Cost',
        'Processors',
        'Memory',
        'Trust',
        'Next Trust',
        'Operations',
        'Creativity'
    ]    
    _core_action_set_1 = [
        'Make Paperclip', 
        'Lower Price', 
        'Raise Price', 
        'Buy Autoclipper',
        'Buy Wire',
        'Expand Marketing',
        'Add Processor',
        'Add Memory'
    ]
    
    # Stage 1
    _action_intervals_stages.append(0.2)
    _stage_1_projects = [
        'Improved AutoClippers',
        'Creativity',
        'Even Better AutoClippers',        
        'Improved Wire Extrusion',
        'New Slogan',
        'Limerick',
        'Lexical Processing',
        'Combinatory Harmonics',
        'The Hadwiger Problem'
    ]
    _stage_1_projects_obs = [proj+" Activated" for proj in _stage_1_projects]
    _stage_1_projects_ac = ["Activate "+proj for proj in _stage_1_projects]
    _observation_names_stages.append(_core_observation_set_1+_stage_1_projects_obs)
    _action_names_stages.append(_core_action_set_1+_stage_1_projects_ac)
    
    # Stage 2
    _action_intervals_stages.append(0.2)
    _stage_2_required_projects = [
        'Improved AutoClippers',
        'Creativity',
        'Even Better AutoClippers',
        'Improved Wire Extrusion',
        'New Slogan',
        'Limerick',
        'Lexical Processing'
    ]
    _stage_2_projects = [        
        'Optimized AutoClippers',                
        'Optimized Wire Extrusion',        
        'Catchy Jingle',
        'Combinatory Harmonics',
        'The Hadwiger Problem',
        'The Toth Sausage Conjecture',
        'Donkey Space'
    ]
    _stage_2_projects_obs = [proj+" Activated" for proj in _stage_2_projects]
    _stage_2_projects_ac = ["Activate "+proj for proj in _stage_2_projects]
    _observation_names_stages.append(_core_observation_set_1+_stage_2_projects_obs)
    _action_names_stages.append(_core_action_set_1+_stage_2_projects_ac)
    
    # Stage 3
    _action_intervals_stages.append(0.2)
    _stage_3_required_projects = [        
        'Optimized AutoClippers',                
        'Optimized Wire Extrusion',        
        'Catchy Jingle',
        'Combinatory Harmonics',
        'The Hadwiger Problem',
        'The Toth Sausage Conjecture',
    ]
    _stage_3_projects = [
        'Microlattice Shapecasting',
        'Hadwiger Clip Diagrams',
        'WireBuyer',
        'Hypno Harmonics',
        'Donkey Space'
    ]
    _stage_3_projects_obs = [proj+" Activated" for proj in _stage_3_projects]
    _stage_3_projects_ac = ["Activate "+proj for proj in _stage_3_projects]
    _observation_names_stages.append(_core_observation_set_1+_stage_3_projects_obs)
    _action_names_stages.append(_core_action_set_1+_stage_3_projects_ac)
    
    # Observations and actions common to stages 4+
    _core_observation_set_2 = [
        'Unsold Inventory', 
        'Price per Clip', 
        'Public Demand', 
        'Available Funds', 
        'Autoclipper Cost', 
        'Number of Autoclippers',
        'Paperclips',
        'Marketing Level',
        'Marketing Cost',
        'Processors',
        'Memory',
        'Trust',
        'Next Trust',
        'Operations',
        'Creativity',
        'Investment Bankroll',
        'Stocks',
        'Riskiness',
        'Number of Photonic Chips',
        'Photonic Chip 0 Level'      
    ]    
    _core_action_set_2 = [
        'Lower Price', 
        'Raise Price', 
        'Buy Autoclipper',
        'Expand Marketing',
        'Add Processor',
        'Add Memory',
        'Set Investment Low',
        'Set Investment Medium',
        'Set Investment High',
        'Withdraw',
        'Deposit',
        'Activate Photonic Chip',
        'Quantum Compute'
    ]
    
    # Stage 4
    _action_intervals_stages.append(2.5)
    _stage_4_required_projects = [
        'Microlattice Shapecasting',
        'Hadwiger Clip Diagrams',
        'WireBuyer',
        'Hypno Harmonics'
    ]
    _stage_4_projects = [
        'Algorithmic Trading',
        'Quantum Computing',
        'Donkey Space'
    ]
    _stage_4_projects_obs = [proj+" Activated" for proj in _stage_4_projects]
    _stage_4_projects_ac = ["Activate "+proj for proj in _stage_4_projects]
    _observation_names_stages.append(_core_observation_set_2+_stage_4_projects_obs)
    _action_names_stages.append(_core_action_set_2+_stage_4_projects_ac)    
    
    # Observations and actions common to stages 5+
    _core_observation_set_3 = _core_observation_set_2 + [
        'Latest QOps',
        'MegaClipper Cost',
        'Number of MegaClippers',
        'Investment Engine Level',
        'Investment Engine Upgrade Cost',
        'Yomi',
        'Tournament Cost'
    ]
    _core_action_set_3 = _core_action_set_2 + [
        'Buy MegaClipper',
        'Upgrade Investment Engine',
        'Run New Tournament'
    ]
    
    # Stage 5
    _action_intervals_stages.append(2.5)
    _stage_5_required_projects = [
        'Donkey Space'
    ]
    _stage_5_projects = [
        'Algorithmic Trading',
        'Quantum Computing',
        'Spectral Froth Annealment',
        'MegaClippers',
        'Strategic Modeling',   
    ]
    _stage_5_projects_obs = [proj+" Activated" for proj in _stage_5_projects]
    _stage_5_projects_ac = ["Activate "+proj for proj in _stage_5_projects]
    _observation_names_stages.append(_core_observation_set_3+_stage_5_projects_obs)
    _action_names_stages.append(_core_action_set_3+_stage_5_projects_ac)
       
    def __init__(self,
                 url,
                 initial_states_filename=None,
                 initial_stage=0,
                 final_stage=None,
                 resetter_agents=[],
                 use_emulator=False,
                 episode_length=None,
                 action_rate_speedup=1.0,
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
        self._action_rate_speedup = action_rate_speedup
        self._verbose = verbose
        
        # Initial states
        self._init_states = None
        if initial_states_filename != None:
            with open(initial_states_filename, 'rb') as f:
                self._init_states = pickle.load(f)            
        
        # Parameters for working up to a given stage
        if initial_stage != len(resetter_agents):
            raise Exception("Incorrect number of resetter agents ({}) for desired initial stage ({}). Should be equal.".format(len(resetter_agents), initial_stage))
        self._initial_stage = initial_stage
        self._final_stage = final_stage
        self._resetter_agents = resetter_agents
    
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
                    print("Advancing to stage 1.")
                stage_changed = True
                
        # Update rule for stage 1 -> 2   
        if self._stage == 1:
            required_projects_obs = [proj+" Activated" for proj in self._stage_2_required_projects]
            observation_from_handler = self._handler.makeObservation(required_projects_obs)
            all_projects_activated = True
            for name, obs in observation_from_handler.items():
                if obs != 1:
                    all_projects_activated = False
            if all_projects_activated:
                self._stage = 2
                if self._verbose:
                    print("Advancing to stage 2.")
                stage_changed = True
        
        # Update rule for stage 2 -> 3
        if self._stage == 2:
            required_projects_obs = [proj+" Activated" for proj in self._stage_3_required_projects]
            observation_from_handler = self._handler.makeObservation(required_projects_obs)
            all_projects_activated = True
            for name, obs in observation_from_handler.items():
                if obs != 1:
                    all_projects_activated = False
            if all_projects_activated:
                self._stage = 3
                if self._verbose:
                    print("Advancing to stage 3.")
                stage_changed = True
                
        # Update rule for stage 3 -> 4
        if self._stage == 3:
            required_projects_obs = [proj+" Activated" for proj in self._stage_4_required_projects]
            observation_from_handler = self._handler.makeObservation(required_projects_obs)
            all_projects_activated = True
            for name, obs in observation_from_handler.items():
                if obs != 1:
                    all_projects_activated = False
            if all_projects_activated:
                self._stage = 4
                if self._verbose:
                    print("Advancing to stage 4.")
                stage_changed = True
                
        # Update rule for stage 4 -> 5
        if self._stage == 4:
            required_projects_obs = [proj+" Activated" for proj in self._stage_5_required_projects]
            observation_from_handler = self._handler.makeObservation(required_projects_obs)
            all_projects_activated = True
            for name, obs in observation_from_handler.items():
                if obs != 1:
                    all_projects_activated = False
            if all_projects_activated:
                self._stage = 5
                if self._verbose:
                    print("Advancing to stage 5.")
                stage_changed = True 
        
        if stage_changed:
            self._observation_names = self._observation_names_stages[self._stage]
            self._action_names = self._action_names_stages[self._stage]
        
        self._desired_action_interval = self._action_intervals_stages[self._stage]/self._action_rate_speedup
        
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
        
        if target_stage == 1:
            max_game_time = 400
        elif target_stage == 2:
            max_game_time = 1000
        elif target_stage == 3:
            max_game_time = 1600
        elif target_stage == 4:
            max_game_time = 2200
        elif target_stage == 5:
            max_game_time = 2800
        elif target_stage == 6:
            raise NotImplementedError("No definition for stage 6+.")
        
        # Advance
        stochastic = True
        prev_stage = self._stage
        while self._stage < target_stage:
            # Act
            agent = agents[self._stage] 
            ac, vpred = agent.act(stochastic, ob)
            ob, rew, done, info = self._step(ac, self._stage)
            
            # Observe in new observation space if stage changed
            if prev_stage != self._stage:
                ob_space = UPObservationSpace(self._observation_names_stages[self._stage])
                observation_from_handler = self._handler.makeObservation(ob_space.getPossibleObservations())
                ob = ob_space.observationAsArray(observation_from_handler)
            
            # Restart if failed to get to next stage quickly enough
            #~ print(self._game_time)
            if self._game_time > max_game_time:
                print("WARNING: Timed out in stage {}. Resetting and trying fresh.".format(self._stage))
                self._n_steps_taken = 0
                self._prev_act_time = None
                self._handler.reset()
                self._stage = 0
                self._game_time = 0.0
                ob_space = UPObservationSpace(self._observation_names_stages[self._stage])
                observation_from_handler = self._handler.makeObservation(ob_space.getPossibleObservations())
                self._prev_observation_from_handler = observation_from_handler        
                ob = ob_space.observationAsArray(observation_from_handler)
            
            # Report
            if self._verbose and self._stage > prev_stage:          
                print("Advanced to stage {} after {} seconds.".format(self._stage, self._game_time))
            
            prev_stage = self._stage
        
        print("Completed initial stage advancement after {} seconds.".format(self._game_time))
    
    def _loadInitialState(self):
        if self._init_states != None:
            init_state = np.random.choice(self._init_states)
            self.loadStateFromString(init_state)
    
    def _reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        
        # Reset handler
        self._handler.reset()
        
        # Load an initial state to handler if available and possible
        self._loadInitialState()
        
        # Set correct initial stage
        self._stage = 0
        self._update_stage()      
        
        # Advance to initial stage
        self._game_time = 0.0
        if self._initial_stage != 0:
            self._advance_to_stage(self._initial_stage, self._resetter_agents)
        self._desired_action_interval = self._action_intervals_stages[self._initial_stage]/self._action_rate_speedup
        
        # Observe
        observation_from_handler = self._handler.makeObservation(self.observation_space.getPossibleObservations())
        observation = self.observation_space.observationAsArray(observation_from_handler)
        
        # Initial values
        self._prev_observation_from_handler = observation_from_handler
        self._prev_act_time = None
        self._n_steps_taken = 0
        
        # Return
        return observation
    
    def reward(self):
        ob_space = UPObservationSpace(self._observation_names_stages[self._stage])
        observation_from_handler = self._handler.makeObservation(ob_space.getPossibleObservations())
        if self._stage >= 0 and self._stage <= 3:
            reward = self.assetsAndCashReward(observation_from_handler)
            if self._stage < 3:
                reward *= 1.0
            elif self._stage == 3:
                reward *= 1e-4
        elif self._stage == 4:
            reward = self.assetsAndCashRewardStage4Plus(observation_from_handler)
            reward *= 1e-5
        elif self._stage == 5:
            reward = self.assetsAndCashRewardStage5Plus(observation_from_handler)
            reward *= 1e-6
        else:
            raise NotImplementedError("No definition for stage 6+.")
            
        return reward
            
    def assetsAndCashReward(self, observation_from_handler):
        return self.cashReward(observation_from_handler) + self.assetsReward(observation_from_handler)
    
    def assetsAndCashRewardStage4Plus(self, observation_from_handler):
        return self.cashReward(observation_from_handler) + self.assetsRewardStage4Plus(observation_from_handler)
        
    def assetsAndCashRewardStage5Plus(self, observation_from_handler):
        return self.cashReward(observation_from_handler) + self.assetsRewardStage5Plus(observation_from_handler)
    
    def _getWirePerSpool(self):
        wire_obs = [
            'Quantum Foam Annealment Activated',
            'Spectral Froth Annealment Activated',
            'Microlattice Shapecasting Activated',
            'Optimized Wire Extrusion Activated',
            'Improved Wire Extrusion Activated'
        ]
        obs = self._handler.makeObservation(wire_obs)    
        if obs['Quantum Foam Annealment Activated'] == 1:
            wire_per_spool = 173250
        elif obs['Spectral Froth Annealment Activated'] == 1:
            wire_per_spool = 15750
        elif obs['Microlattice Shapecasting Activated'] == 1:
            wire_per_spool = 5250
        elif obs['Optimized Wire Extrusion Activated'] == 1:
            wire_per_spool = 2625
        elif obs['Improved Wire Extrusion Activated'] == 1:
            wire_per_spool = 1500
        else:
            wire_per_spool = 1000
        return wire_per_spool
    
    def assetsReward(self, observation_from_handler):        
        dassets = 0.0
        
        wire_per_spool = self._getWirePerSpool()
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
        
    def assetsRewardStage4Plus(self, observation_from_handler):        
        dassets = 0.0
        
        dautoclippers = observation_from_handler['Number of Autoclippers'] - self._prev_observation_from_handler['Number of Autoclippers']
        autoclipper_cost = self._prev_observation_from_handler['Autoclipper Cost']        
        dassets += dautoclippers*autoclipper_cost 
        
        dmarketing = observation_from_handler['Marketing Level'] - self._prev_observation_from_handler['Marketing Level']
        marketing_cost = self._prev_observation_from_handler['Marketing Cost']
        dassets += dmarketing*marketing_cost
        
        try:
            prev_bankroll = self._prev_observation_from_handler['Investment Bankroll']
            prev_stocks = self._prev_observation_from_handler['Stocks']
        except KeyError:
            prev_bankroll = 0
            prev_stocks = 0
        
        dbankroll = observation_from_handler['Investment Bankroll'] - prev_bankroll
        dassets += dbankroll
        
        dstocks = observation_from_handler['Stocks'] - prev_stocks
        dassets += dstocks
        
        return dassets 
        
    def assetsRewardStage5Plus(self, observation_from_handler):
        dassets = self.assetsRewardStage4Plus(observation_from_handler)
        
        try:
            prev_megaclippers = self._prev_observation_from_handler['Number of MegaClippers']
            megaclipper_cost = self._prev_observation_from_handler['MegaClipper Cost']
        except KeyError:
            prev_megaclippers = 0
            megaclipper_cost = 0
        
        dmegaclippers = observation_from_handler['Number of MegaClippers'] - prev_megaclippers      
        dassets += dmegaclippers*megaclipper_cost
        
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
        # Timing control
        if self._use_emulator:
            self._game_time += self._desired_action_interval
        else:       
            if self._prev_act_time != None:
                dt = timeSeconds() - self._prev_act_time
                time_remaining = self._desired_action_interval - dt
                if time_remaining > 0:
                    time.sleep(time_remaining)
                    self._game_time += self._desired_action_interval
                else:
                    print("WARNING: Took {:1.2g} s for step, which is more than the desired {:1.2g} s.".format(dt, self._desired_action_interval))
                    self._game_time += dt
        
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
        reward = self.reward()
        
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
    
    def getStateAsString(self):
        if self._use_emulator:
            return self._handler.getStateAsString()
        else:
            raise NotImplementedError("No implementation for state fetching without emulator.")
            
    def loadStateFromString(self, stateString):
        if self._use_emulator:
            self._handler.loadStateFromString(stateString)
        else:
            raise NotImplementedError("No implementation for setting state without emulator.")
    
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
