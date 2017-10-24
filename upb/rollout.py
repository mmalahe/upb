"""Usage example: mpirun -np 4 python3 rollout.py
"""

from UPEnv import *
from UPUtil import *
from policies import *
from mpi4py import MPI

# Game handler
use_emulator = True
webdriver_name_observation = 'Chrome'
webdriver_path_observation = None
url_observation = "file:///home/mikl/projects/upb/src/game/index2.html"
desired_action_interval_observation = 0.2

# Environment parameters
stage = 1
episode_length = 50000

# Policy
policy_filename = 'data/policy_stage{}_latest.pickle'.format(stage)
#~ policy_filename = 'data/policy_stage{}_iter130.pickle'.format(stage)

# Pick stage
if stage == 1:
    observation_names = up_observation_names_stage1
    action_names = up_action_names_stage1
    
def observe():
     # MPI setup
    rank = MPI.COMM_WORLD.Get_rank()
    comm_size = MPI.COMM_WORLD.Get_size()
    sess = tf_util.single_threaded_session()
    sess.__enter__()
    
    # The observation environment
    env = UPEnv(url_observation, 
                observation_names, 
                action_names,
                use_emulator=use_emulator,
                episode_length=episode_length,
                desired_action_interval=desired_action_interval_observation,
                webdriver_name=webdriver_name_observation,
                webdriver_path=webdriver_path_observation,
                headless=False                  
                )
                
    # The policy
    policy = MlpPolicySaveable(name='pi', 
                               ob_space=env.observation_space, 
                               ac_space=env.action_space, 
                               hid_size=32, 
                               num_hid_layers=2)
    policy.load_and_check(policy_filename)
    
    # Rollout
    rollout(env, policy)
    policy.check_is_same_as(policy_filename)
    env.save_screenshot("rollout_final.png")      
    
def main():
    observe()

if __name__ == "__main__":
    main()
