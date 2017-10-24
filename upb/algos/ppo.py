import keras

class ProximalPolicyOptimization(object):
    def __init__(self, env, policy_init):
        self._env = env
        self._policy = policy_init
        
    def learn(self):
        # Compile policy
        self._policy.compile()
        
        # Optimize
        
        
