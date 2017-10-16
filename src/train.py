from UPEnv import *
from rllab.algos.npo import NPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.instrument import run_experiment_lite
import pickle

stage = 2
path_length = 2000
batch_size = 1*path_length
n_snapshots = 150
n_itr_per_snapshot = 1
policy_filename = 'latest_policy_{}.pickle'.format(stage)
#~ selenium_executor = 'http://127.0.0.1:4445/wd/hub'
selenium_executor = None

observation_names_stage1 = ['Unsold Inventory', 'Price per Clip', 'Public Demand', 'Available Funds']
action_names_stage1 = ['Make Paperclip', 'Lower Price', 'Raise Price']

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

if stage == 1:
    observation_names = observation_names_stage1
    action_names = action_names_stage1
elif stage == 2:
    observation_names = observation_names_stage2
    action_names = action_names_stage2

def run_task(*_):   
    # The training environment
    env = normalize(UPEnv("file:///home/mikl/projects/upb/src/game/index2_train.html", observation_names, action_names, selenium_executor=selenium_executor))

    # Fresh policy
    #~ policy = CategoricalMLPPolicy(
        #~ env_spec=env.spec,
        #~ hidden_sizes=(32,32)
    #~ )
    
    # Load policy
    with open(policy_filename,'rb') as f:
        policy = pickle.load(f)
    
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    for i_snapshot in range(n_snapshots):     
        print("Snapshot number "+str(i_snapshot))  
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

# Train
#~ run_task()

# Parallel Train
#~ run_experiment_lite(
    #~ run_task,
    #~ # Number of parallel workers for sampling
    #~ n_parallel=2,
    #~ # Only keep the snapshot parameters for the last iteration
    #~ snapshot_mode="last",
    #~ # Specifies the seed for the experiment. If this is not provided, a random seed
    #~ # will be used
    #~ seed=1,
    #~ # plot=True,
#~ )

# Observe final policy
with open(policy_filename,'rb') as f:
    final_policy = pickle.load(f)
env = normalize(UPEnv("file:///home/mikl/projects/upb/game/index2.html", observation_names, action_names, headless=False, verbose=True))
observation = env.reset()
for i in range(path_length):
    action, _ = final_policy.get_action(observation)
    observation, reward, done, info = env.step(action)

input("Press Enter to continue...")
