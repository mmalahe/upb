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
use_emulator = False
webdriver_name = 'Chrome'
webdriver_path = None
url = LOCAL_GAME_URL_STANDARD
action_rate_speedup = 1.0

# Files
agents_dir = "agents"
inits_dir = "inits"
render_dir = "render"

# Environment parameters
initial_stage = 0
final_stage = 6
episode_length = 5000

# Agents
agent_filenames = [os.path.join(agents_dir,"stage{}.pickle".format(i)) for i in range(final_stage+1)]

# Loading initial states
do_load_init_states = False
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
    
# Display
do_render_agent_decision = True
decision_base_filename = os.path.join(render_dir,"decision")
i_decision = 0

def main():
     # MPI setup
    rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()
    sess = tf_util.single_threaded_session()
    sess.__enter__()
    
    # Set up data directories
    if rank == 0:
        for a_dir in [agents_dir, inits_dir, render_dir]:
            if not os.path.exists(a_dir):
                os.makedirs(a_dir)
    MPI.COMM_WORLD.Barrier()
    
    # For resetting to a fixed stage
    resetter_agents = load_agents(initial_stage, agent_filenames, base_name="resetter_agent")
    
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
                
    # The main agents
    agents = load_agents(final_stage+1, agent_filenames, base_name="agent")
    
    # Callback
    def callback(iter_num, env, agent, ob, ac_avail, ac, vpred, rew, done, info):
        pass
    
    # Create a new set of initial conditions
    if do_create_init_states:
        create_states_batch(env, agents, n_init_states, init_states_creation_filename)
    # Normal Rollout
    else:
        rollout(env, agents, render_decision_basename=decision_base_filename, callback=callback)
        env.save_screenshot("final_state.png")
    
if __name__ == "__main__":
    main()
