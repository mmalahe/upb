"""Usage example: mpirun -np 4 python3 train.py
"""

# export CUDA_VISIBLE_DEVICES=""

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
from upb.agents.mlp import MLPAgent, load_mlp_agent_topology
import os

import matplotlib.pyplot as plt

import numpy as np

# Game emulator
use_emulator = True
action_rate_speedup = 1.0

# Game handler
webdriver_name_training = 'PhantomJS'
webdriver_path_training = "/home/mikl/sfw/phantomjs-2.1.1-linux-x86_64/bin/phantomjs"
#~ desired_action_interval_training = 0.067 # "Training" version of game for a sufficiently fast webdriver ticks three times faster
url_training = LOCAL_GAME_URL_TRAIN

# Resetting environment to an initial stage
initial_stage = 5
final_stage = 5 # Stage past which not to actually advance
resetter_agent_filenames = [os.path.join("agents","stage{}.pickle".format(i)) for i in range(initial_stage)]

# Training parameters
do_load_latest_agent = True
load_only_observation_scaling = False
update_obs_scaling = True
episode_length = 720
timesteps_per_batch = 64*episode_length
optim_batchsize = timesteps_per_batch
max_iters = 245
schedule = 'linear'
stochastic = True
clip_param = 0.175
vf_loss_coeff = 0.01
entcoeff = 0.00
optim_stepsize = 1e-3
optim_epochs = 256
gamma = 1.0
lam = 1.0

# Data management
iters_per_render = 10
iters_per_save = 1
iters_per_plot = 1
data_dir = "data"
init_states_dir = "inits"
policy_filename_latest = os.path.join(data_dir,"policy_stage{}_latest.pickle".format(initial_stage))
policy_filename_latest_old = os.path.join(data_dir,"policy_stage{}_latest_old.pickle".format(initial_stage))
initial_states_filename = os.path.join(init_states_dir, "stage{}.pickle".format(initial_stage))
#~ initial_states_filename = None
rewards_history = []
obs_means_history = []

def train():
    # MPI setup
    rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()
    
    sess = tf_util.single_threaded_session()
    sess.__enter__()
    if rank != 0:
        logger.set_level(logger.DISABLED)
        
    # Set up data directory
    if rank == 0:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    MPI.COMM_WORLD.Barrier()
    
    # For resetting to a fixed stage
    resetter_agents = load_resetter_agents(initial_stage, resetter_agent_filenames)
    
    # The training environment
    env = UPEnv(url_training,
                initial_states_filename=initial_states_filename,
                initial_stage=initial_stage,
                final_stage=final_stage,
                resetter_agents=resetter_agents,
                episode_length=episode_length,
                action_rate_speedup=action_rate_speedup,
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
            env = loc['env']
            rollout(env, policy)
            env.save_screenshot("data/iter_{:05d}.png".format(iters_so_far))
        
        if iters_so_far % iters_per_plot == 0 and iters_so_far > 0:
            # Plot reward history
            fig = plt.figure(figsize=(16,12))
            r_mean = [np.mean(r) for r in rewards_history]
            r_std = [np.std(r) for r in rewards_history]
            r_min = [np.min(r) for r in rewards_history]
            r_max = [np.max(r) for r in rewards_history]
            r_median = [np.median(r) for r in rewards_history] 
            plt.plot(np.array(r_min),'r-')
            plt.plot(np.array(r_median), 'k--')
            #~ plt.errorbar(np.arange(len(r_mean)), np.array(r_mean), yerr=np.array(r_std), fmt='k-')
            plt.plot(np.array(r_mean), 'k-')
            #~ plt.plot(np.array(r_max),'b-')
            plt.xlabel("Iterations")
            plt.ylabel("Reward")
            plt.savefig(os.path.join(data_dir,"reward_history.png"))
            plt.close(fig)
            print("Best min at iteration", np.argmax(r_min))
            print("Best mean at iteration", np.argmax(r_mean))
            print("Best median at iteration", np.argmax(r_median))
            print("Best max at iteration", np.argmax(r_max))
            plt.close()
            
            # Plot observation means            
            fig = plt.figure(figsize=(16,12))
            n_obs = len(obs_means_history[0])
            leg = []
            for i in range(n_obs):
                obs_mean_history = [obs_means[i] for obs_means in obs_means_history]
                plt.semilogy(obs_mean_history)
                obs_name = loc['env'].observation_space.getObservationName(i)
                leg.append(obs_name)
            plt.legend(leg, loc='upper left')
            plt.xlabel("Iterations")
            plt.ylabel("Observation running mean")
            plt.savefig(os.path.join(data_dir,"obs_mean_history.png"))
            
            # Close figure
            plt.close(fig)
            plt.close()
            
    def logging_callback(loc, glob):
        iters_so_far = loc['iters_so_far']
        if iters_so_far > 0:
            print(loc['rews'])
            print("Mean reward =", np.mean(loc['rews']))
            rewards_history.append(loc['rews'].copy())
            obs_means_history.append(loc['pi'].getObservationMeans())
                
    def callback(loc, glob):
        if rank == 0:
            save_callback(loc, glob)
            logging_callback(loc,glob)
            observe_callback(loc, glob)
    
    # Policy generating function
    def policy_fn(name, ob_space, ac_space):
        ob_space = UPObservationSpace(UPEnv._observation_names_stages[initial_stage])
        ac_space = UPActionSpace(UPEnv._action_names_stages[initial_stage])
        
        if do_load_latest_agent:
            hid_size, num_hid_layers = load_mlp_agent_topology(policy_filename_latest)
            agent = MLPAgent(name=name, ob_space=ob_space, 
                         ac_space=ac_space, hid_size=hid_size, num_hid_layers=num_hid_layers)
            if name == "pi":
                agent.load_and_check(policy_filename_latest, load_only_observation_scaling=load_only_observation_scaling)
            elif name == "oldpi":
                agent.load_and_check(policy_filename_latest_old, load_only_observation_scaling=load_only_observation_scaling)
            else:
                print("WARNING: Don't know how to load {}.".format(name))
        else:
            agent = MLPAgent(name=name, ob_space=ob_space, 
                         ac_space=ac_space, hid_size=96, num_hid_layers=2)
                
        return agent
    
    # Learn
    pposgd_simple.learn(env, policy_fn,
        max_iters=max_iters,
        timesteps_per_batch=timesteps_per_batch,
        clip_param=clip_param, entcoeff=entcoeff,
        vf_loss_coeff=vf_loss_coeff,
        optim_epochs=optim_epochs, optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
        gamma=gamma, lam=gamma,
        schedule=schedule,
        stochastic=stochastic,
        callback=callback,
        update_obs_scaling=update_obs_scaling
    )

def main():
    train()

if __name__ == "__main__":
    main()
