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

from upb.envs.UPEnv import *
from upb.util.UPUtil import *
from upb.agents.mlp import MLPAgent
import os

import matplotlib.pyplot as plt

# Resume learning from file
do_resume_learning = False

# Game handler
use_emulator = True
webdriver_name_training = 'PhantomJS'
webdriver_path_training = "/home/mikl/sfw/phantomjs-2.1.1-linux-x86_64/bin/phantomjs"
url_training = os.path.join(dirname(abspath(__file__)),"game","index2_train.html")
#~ desired_action_interval_training = 0.067
desired_action_interval_training = 0.2

# Training parameters
stage = 1
episode_length = 500
timesteps_per_batch = 2*episode_length
max_iters = 1000
schedule = 'linear'
iters_per_render = 100
iters_per_save = 100
data_dir = "data"
policy_filename_latest = os.path.join(data_dir,"policy_stage{}_latest.pickle".format(stage))
learning_state_filename_latest = os.path.join(data_dir,"learning_stage{}_latest.tf".format(stage))

reward_history = []

# Set up data directory
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Pick stage
if stage == 1:
    observation_names = up_observation_names_stage1
    action_names = up_action_names_stage1

def train():
    # MPI setup
    rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()
    sess = tf_util.single_threaded_session()
    sess.__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)
            
    # The training environment
    env = UPEnv(url_training, 
                observation_names, 
                action_names,
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
            # Save policy
            policy_filename_iter = os.path.join(data_dir,"policy_stage{}_iter{}.pickle".format(stage, iters_so_far))    
            pi = loc['pi']            
            pi.save_and_check_reload(policy_filename_iter)
            pi.save_and_check_reload(policy_filename_latest)
            
            # Save full learning state to make resuming easier
            learning_state_filename_iter = os.path.join(data_dir,"learning_stage{}_iter{}.tf".format(stage, iters_so_far))
            tf_util.save_state(learning_state_filename_iter)
            tf_util.save_state(learning_state_filename_latest)
    
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
            
    def resume_callback(loc, glob):
        if do_resume_learning:
            print("Resuming...")
            tf_util.load_state(learning_state_filename_latest)
            MPI.COMM_WORLD.Barrier()
            
    def logging_callback(loc, glob):
        iters_so_far = loc['iters_so_far']
        if iters_so_far > 0:
            reward_history.append(np.mean(loc['rewbuffer']))
                
    def callback(loc, glob):
        iters_so_far = loc['iters_so_far']
        if iters_so_far == 0:
            resume_callback(loc, glob)
        if rank == 0:
            save_callback(loc, glob)
            logging_callback(loc,glob)
            observe_callback(loc, glob)           
    
    # Monitoring
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json"%rank))
    gym.logger.setLevel(logging.WARN)
    
    # Policy generating function
    def policy_fn(name, ob_space, ac_space):
        return MLPAgent(name=name, 
                        ob_space=env.observation_space, 
                        ac_space=env.action_space, 
                        hid_size=32,
                        num_hid_layers=2)                
    
    # Learn
    pposgd_simple.learn(env, policy_fn,
        max_iters=max_iters,
        timesteps_per_batch=timesteps_per_batch,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=8, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule=schedule,
        callback=callback,
        resume_callback=None
    )

def main():
    train()

if __name__ == "__main__":
    main()
