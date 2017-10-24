import keras

class ProximalPolicyOptimization(object):
    """An implementation of "Proximal Policy Optimization Algorithms"
    
    This implementation is built around policies that are represented
    as keras models.
    """
    def __init__(self, env, policy_init_old, policy_init):
        self._env = env
        self._policy_old = policy_init_old
        self._policy = policy_init
        
    def learn(self, max_iters=1024,
                    timesteps_per_batch=1024,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=8, optim_stepsize=1e-3, optim_batchsize=64,
                    gamma=0.99, lam=0.95):
        """ Learns according to the objective
        
        .. math::        
            L^{\mathrm{CLIP}}
        
        :param clip_param: the clipping parameter :math:'\epsilon'
        :type clip_param: float
        :returns:  None -- Nothing
        :raises:
        """
        
        # Begin loop
        
        # Run old policy
        
        # Compute advantage estimates
        
        # Optimize
        
        # Set current policy to old policy
        
    def get_policy(self):
        return self._policy
