"""Usage example: mpirun -np 4 python3 train.py
"""

import sys
import pickle

from mpi4py import MPI

import gym, logging

from baselines import logger
from baselines import bench
from baselines.ppo1 import pposgd_simple, mlp_policy
from baselines.common import tf_util

from UPEnv import *

# What to do
do_train = False
use_fresh_policy = False
do_observe = False

# Game handler
webdriver_name_training = 'PhantomJS'
webdriver_path_training = "/home/mikl/sfw/phantomjs-2.1.1-linux-x86_64/bin/phantomjs"
url_training = "file:///home/mikl/projects/upb/src/game/index2_train.html"
min_action_interval_training = None

webdriver_name_observation = 'Chrome'
webdriver_path_observation = None
url_observation = "file:///home/mikl/projects/upb/src/game/index2.html"
min_action_interval_observation = 0.2

# Training parameters
stage = 2
episode_length = 1024
timesteps_per_batch = episode_length
max_iters = 10
policy_filename = 'policy_stage{}.pickle'.format(stage)

# Stage 1
observation_names_stage1 = [
    'Unsold Inventory', 
    'Price per Clip', 
    'Public Demand', 
    'Available Funds',
    'Wire Inches',
    'Number of Autoclippers',
    'Wire Cost',
    'Autoclipper Cost'
    ]
action_names_stage1 = [
    'Make Paperclip', 
    'Lower Price', 
    'Raise Price']

# Stage 2
observation_names_stage2 = [
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
action_names_stage2 = [
    'Make Paperclip', 
    'Lower Price', 
    'Raise Price', 
    'Buy Autoclipper',
    'Buy Wire']

# Pick stage
if stage == 1:
    observation_names = observation_names_stage1
    action_names = action_names_stage1
elif stage == 2:
    observation_names = observation_names_stage2
    action_names = action_names_stage2

def train():
    # MPI setup
    rank = MPI.COMM_WORLD.Get_rank()
    sess = tf_util.single_threaded_session()
    sess.__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)
     
    # The training environment
    env = UPEnv(url_training, 
                observation_names, 
                action_names,
                episode_length=episode_length,
                min_action_interval=min_action_interval_training,
                webdriver_name=webdriver_name_training,
                webdriver_path=webdriver_path_training,
                headless=True                    
                )
    
    # Monitoring
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json"%rank))
    gym.logger.setLevel(logging.WARN)    
    
    # Learn
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, 
                                    ob_space=env.observation_space, 
                                    ac_space=env.action_space, 
                                    hid_size=32, 
                                    num_hid_layers=2)
    
    pposgd_simple.learn(env, policy_fn,
        max_iters=max_iters,
        timesteps_per_batch=timesteps_per_batch,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='constant'
    )

def main():
    train()

if __name__ == "__main__":
    main()
