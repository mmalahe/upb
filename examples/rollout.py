"""Usage example: mpirun -np 4 python3 rollout.py
"""

from upb.envs.UPEnv import *
from upb.util.UPUtil import *
from upb.agents.mlp import MLPAgent, load_mlp_agent_topology
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
episode_length = 0

# Agents
resetter_agent_filenames = [os.path.join(agents_dir,"stage{}.pickle".format(i)) for i in range(initial_stage)]
policy_filename = os.path.join(agents_dir,"stage{}.pickle".format(initial_stage))

# Creating initial states
create_init_states = True
n_init_states = 2
init_states_filename = os.path.join(inits_dir,"stage{}.pickle".format(initial_stage))

def observe():
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
    resetter_agents = []
    for i in range(initial_stage):
        resetter_agent_filename = resetter_agent_filenames[i]
        ob_space = UPObservationSpace(UPEnv._observation_names_stages[i])
        ac_space = UPActionSpace(UPEnv._action_names_stages[i])
        agent_name = "resetter_agent_stage{}".format(i)
        hid_size, num_hid_layers = load_mlp_agent_topology(resetter_agent_filename)
        agent = MLPAgent(name=agent_name, ob_space=ob_space, 
                     ac_space=ac_space, hid_size=hid_size, num_hid_layers=num_hid_layers)
        agent.load_and_check(resetter_agent_filename)
        resetter_agents.append(agent)
    
    # The observation environment
    env = UPEnv(url,
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
                
    # The agent
    ob_space = UPObservationSpace(UPEnv._observation_names_stages[initial_stage])
    ac_space = UPActionSpace(UPEnv._action_names_stages[initial_stage])
    hid_size, num_hid_layers = load_mlp_agent_topology(policy_filename)
    agent = MLPAgent(name="pi", ob_space=ob_space, 
                 ac_space=ac_space, hid_size=hid_size, num_hid_layers=num_hid_layers)
    agent.load_and_check(policy_filename)
    
    # Create a new set of initial conditions
    if create_init_states:
        init_states = []
        for i in range(n_init_states):
            print("Creating init state {}.".format(i))
            rollout(env, agent)
            init_states.append(env.getStateAsString())
        with open(init_states_filename, 'wb') as f:
            pickle.dump(init_states, f)
    # Normal Rollout
    else:
        rollout(env, agent)
        env.save_screenshot("rollout_final.png")   
    
def main():
    observe()

if __name__ == "__main__":
    main()
