from UPEnv import *
from rllab.algos.npo import NPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import pickle

observation_names = ['Unsold Inventory', 'Price per Clip', 'Public Demand', 'Available Funds']
action_names = ['Make Paperclip', 'Lower Price', 'Raise Price']
path_length = 100
policy_filename = 'latest_policy.pickle'

def run_task(*_):
    # The training environment
    env = normalize(UPEnv("file:///home/mikl/projects/upb/src/index2.html", observation_names, action_names))

    # Solver
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(10,)
    )
    
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    
    algo = NPO(
        env=env,
        policy=policy,
        baseline=baseline,
        whole_paths=True,
        batch_size=path_length,
        max_path_length=path_length,
        n_itr=100,
        discount=0.99,
        step_size=0.01,
    )

    # Do the training
    algo.train()
    
    # Get back the policy
    final_policy = algo.policy
    
    # Save policy
    with open(policy_filename,'wb') as f:
        pickle.dump(final_policy, f)

# Train
run_task()

# Observe final policy
with open(policy_filename,'rb') as f:
    final_policy = pickle.load(f)
env = normalize(UPEnv("file:///home/mikl/projects/upb/src/index2.html", observation_names, action_names, verbose=True))
observation = env.reset()
for i in range(path_length):
    action, _ = final_policy.get_action(observation)
    observation, reward, done, info = env.step(action)

input("Press Enter to continue...")
