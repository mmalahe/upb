from UPEnv import *
from rllab.algos.npo import NPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import run_experiment_lite
import pickle

# What to do
do_train = True
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
use_fresh_policy = False
stage = 2
path_length = 1000
batch_size = 4*path_length
n_snapshots = 2
n_itr_per_snapshot = 1
policy_filename = 'latest_policy_stage{}.pickle'.format(stage)

# Stage 1
observation_names_stage1 = [
    'Unsold Inventory', 
    'Price per Clip', 
    'Public Demand', 
    'Available Funds']
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

def train(*_):   
    # The training environment
    env = normalize(UPEnv(url_training, 
                          observation_names, 
                          action_names,
                          min_action_interval=min_action_interval_training,
                          webdriver_name=webdriver_name_training,
                          webdriver_path=webdriver_path_training,
                          headless=True                          
                          ))

    # Initial policy
    if use_fresh_policy:
        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32,32)
        )
    else:
        with open(policy_filename,'rb') as f:
            policy = pickle.load(f)
    
    # Baseline
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    # Training loop
    for i_snapshot in range(n_snapshots):     
        print("SNAPSHOT NUMBER {}".format(i_snapshot)) 
        algo = NPO(
            env=env,
            policy=policy,
            baseline=baseline,
            whole_paths=True,
            batch_size=batch_size,
            max_path_length=path_length,
            n_itr=n_itr_per_snapshot,
            discount=1.00,
            step_size=0.01,
        )

        # Do the training
        algo.train()
        
        # Get back the policy
        policy = algo.policy
    
        # Save policy
        with open(policy_filename,'wb') as f:
            pickle.dump(policy, f)

if do_train:
    train()

if do_observe:
    with open(policy_filename,'rb') as f:
        final_policy = pickle.load(f)
    env = normalize(UPEnv(url_observation, 
                          observation_names, 
                          action_names,
                          min_action_interval=min_action_interval_observation,
                          webdriver_name=webdriver_name_observation,
                          webdriver_path=webdriver_path_observation,
                          headless=False,
                          verbose=True                        
                          ))
    observation = env.reset()
    for i in range(path_length):
        action, _ = final_policy.get_action(observation)
        observation, reward, done, info = env.step(action)
    input("Press Enter to continue...")
