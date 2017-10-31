"""Usage example: mpirun -np 4 python3 train.py
"""

import sys
import pickle

from mpi4py import MPI

import gym, logging

import tensorflow as tf

from baselines import logger
from baselines import bench
from baselines.ppo1 import pposgd_simple
from baselines.common import tf_util

from upb.game.UPGameHandler import LOCAL_GAME_URL_TRAIN
from upb.envs.UPEnv import *
from upb.util.UPUtil import *
from upb.agents.mlp import MLPAgent
import os

import matplotlib.pyplot as plt

# Load latest agent from file
do_load_latest_agent = True

# Game emulator
use_emulator = True
desired_action_interval_training = 0.2

# Game handler
webdriver_name_training = 'PhantomJS'
webdriver_path_training = "/home/mikl/sfw/phantomjs-2.1.1-linux-x86_64/bin/phantomjs"
#~ desired_action_interval_training = 0.067 # "Training" version of game for a sufficiently fast webdriver ticks three times faster
url_training = LOCAL_GAME_URL_TRAIN

# Resetting environment to an initial stage
initial_stage = 0
final_stage = 0 # Stage past which not to actually advance
resetter_agent_filenames = [os.path.join("agents","stage0.pickle")]

# Training parameters
episode_length = 2000
timesteps_per_batch = 4*episode_length
max_iters = 1000
schedule = 'linear'
iters_per_render = 10
iters_per_save = 10
data_dir = "data"
policy_filename_latest = os.path.join(data_dir,"policy_stage{}_latest.pickle".format(initial_stage))
policy_filename_latest_old = os.path.join(data_dir,"policy_stage{}_latest_old.pickle".format(initial_stage))
reward_history = []

# Set up data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def train():
    # MPI setup
    rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()
    sess = tf_util.single_threaded_session()
    sess.__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    
    # For resetting to a fixed stage
    resetter_agents = []
    for i in range(initial_stage):
        resetter_agent_filename = resetter_agent_filenames[i]
        ob_space = UPObservationSpace(UPEnv._observation_names_stages[i])
        ac_space = UPActionSpace(UPEnv._action_names_stages[i])
        agent_name = "resetter_agent_stage{}".format(i)
        agent = MLPAgent(name=agent_name, ob_space=ob_space, ac_space=ac_space, 
                         hid_size=32, num_hid_layers=2)
        agent.load_and_check(resetter_agent_filename)
        resetter_agents.append(agent)
    
    # The training environment
    env = UPEnv(url_training, 
                initial_stage=initial_stage,
                final_stage=final_stage,
                resetter_agents=resetter_agents,
                episode_length=episode_length,
                desired_action_interval=desired_action_interval_training,
                use_emulator=use_emulator,
                webdriver_name=webdriver_name_training,
                webdriver_path=webdriver_path_training,
                headless=True                    
                )
    
    # Callbacks to execute inside the trainer
    def save_callback(loc, glob):
        iters_so_far = loc['iters_so_far']
        if iters_so_far % iters_per_save == 0 and iters_so_far > 0:
            # Save policies
            policy_filename_iter = os.path.join(data_dir,"policy_stage{}_iter{}.pickle".format(initial_stage, iters_so_far))    
            pi = loc['pi']            
            pi.save_and_check_reload(policy_filename_iter)
            pi.save_and_check_reload(policy_filename_latest)
            
            policy_filename_iter_old = os.path.join(data_dir,"policy_stage{}_iter{}_old.pickle".format(initial_stage, iters_so_far))    
            oldpi = loc['oldpi']            
            oldpi.save_and_check_reload(policy_filename_iter_old)
            oldpi.save_and_check_reload(policy_filename_latest_old)
    
    def observe_callback(loc, glob):
        iters_so_far = loc['iters_so_far']
        if iters_so_far % iters_per_render == 0 and iters_so_far > 0:
            policy = loc['pi']
            env = loc['env'].env
            rollout(env, policy)
            env.save_screenshot("data/iter_{:05d}.png".format(iters_so_far))
            
            # Plot reward history
            fig = plt.figure()
            plt.plot(np.array(reward_history))
            plt.xlabel("Iterations")
            plt.ylabel("Reward")
            plt.savefig(os.path.join(data_dir,"reward_history.png"))
            plt.close(fig)
            
    def logging_callback(loc, glob):
        iters_so_far = loc['iters_so_far']
        if iters_so_far > 0:
            reward_history.append(np.mean(loc['rewbuffer']))
                
    def callback(loc, glob):
        if rank == 0:
            save_callback(loc, glob)
            logging_callback(loc,glob)
            observe_callback(loc, glob)           
    
    # Monitoring
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json"%rank))
    gym.logger.setLevel(logging.WARN)
    
    # Policy generating function
    def policy_fn(name, ob_space, ac_space):
        ob_space = UPObservationSpace(UPEnv._observation_names_stages[initial_stage])
        ac_space = UPActionSpace(UPEnv._action_names_stages[initial_stage])
        agent = MLPAgent(name=name, ob_space=ob_space, 
                         ac_space=ac_space, hid_size=32, num_hid_layers=2)
        
        if do_load_latest_agent:
            if name == "pi":
                agent.load_and_check(policy_filename_latest)
            elif name == "oldpi":
                agent.load_and_check(policy_filename_latest_old)
            else:
                print("WARNING: Don't know how to load {}.".format(name))
                
        return agent
    
    # Learn
    pposgd_simple.learn(env, policy_fn,
        max_iters=max_iters,
        timesteps_per_batch=timesteps_per_batch,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=8, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule=schedule,
        callback=callback
    )

def main():
    train()

if __name__ == "__main__":
    main()
