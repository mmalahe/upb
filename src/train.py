from UPEnv import *
from rllab.algos.trpo import TRPO
from rllab.algos.vpg import VPG
from rllab.algos.npo import NPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.misc.instrument import run_experiment_lite

# The training environment
env = normalize(UPEnv("file:///home/mikl/projects/upb/src/index2.html"))

# Solver
policy = CategoricalMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(32, 32)
)

#~ policy = GaussianMLPPolicy(
    #~ env_spec=env.spec,
    #~ hidden_sizes=(32, 32)
#~ )

baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    whole_paths=True,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)

# Do the training
algo.train()
