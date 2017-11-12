"""Usage example: mpirun -np 4 python3 rollout.py
"""

from upb.envs.UPEnv import *
from upb.util.UPUtil import *
from upb.agents.mlp import MLPAgent, load_mlp_agent
from baselines.common import tf_util
from mpi4py import MPI
import os
from upb.game.UPGameHandler import LOCAL_GAME_URL_STANDARD
import pickle

# Game handler
use_emulator = True
webdriver_name = 'Chrome'
webdriver_path = None
url = LOCAL_GAME_URL_STANDARD
action_rate_speedup = 1.0

# Files
agents_dir = "agents"
inits_dir = "inits"

# Environment parameters
initial_stage = 5
final_stage = 5
episode_length = 2

# Agents
resetter_agent_filenames = [os.path.join(agents_dir,"stage{}.pickle".format(i)) for i in range(initial_stage)]
agent_filename = os.path.join(agents_dir,"stage{}.pickle".format(initial_stage))

# Loading initial states
do_load_init_states = True
if do_load_init_states:
    init_states_loading_filename = os.path.join(inits_dir,"stage{}.pickle".format(initial_stage))
else:
    init_states_loading_filename = None

# Creating initial states
do_create_init_states = False
n_init_states = 128
init_states_creation_filename = os.path.join(inits_dir,"stage{}.pickle".format(initial_stage))
if do_create_init_states:
    assert episode_length == 0

def main():
     # MPI setup
    rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()
    sess = tf_util.single_threaded_session()
    sess.__enter__()
    
    # Set up data directories
    if rank == 0:
        for a_dir in [agents_dir, inits_dir]:
            if not os.path.exists(a_dir):
                os.makedirs(a_dir)
    MPI.COMM_WORLD.Barrier()
    
    # For resetting to a fixed stage
    resetter_agents = load_resetter_agents(initial_stage, resetter_agent_filenames)
    
    # The rollout environment
    env = UPEnv(url,
                initial_states_filename=init_states_loading_filename,
                initial_stage=initial_stage,
                final_stage=final_stage,
                resetter_agents=resetter_agents,
                use_emulator=use_emulator,
                episode_length=episode_length,
                action_rate_speedup=action_rate_speedup,
                webdriver_name=webdriver_name,
                webdriver_path=webdriver_path,
                headless=False                  
                )
                
    # The main agent
    agent = load_mlp_agent(agent_filename, "main", env.observation_space, env.action_space)
    
    # Callbacks
    def printActionProbs(agent, ob, ac, vpred, rew, done, info):
        print(agent.getActionProbabilities(ob))
    
    # Create a new set of initial conditions
    if do_create_init_states:
        create_states_batch(env, agent, n_init_states, init_states_creation_filename)
    # Normal Rollout
    else:
        rollout(env, agent, callback=printActionProbs)
        env.save_screenshot("rollout_final.png.")
    
if __name__ == "__main__":
    main()
