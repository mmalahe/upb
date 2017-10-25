import keras
import keras.backend as K

class ProximalPolicyOptimization(object):
    """An implementation of "Proximal Policy Optimization Algorithms"
    
    This implementation is built around policies that are represented
    as keras models.
    """
    def __init__(self, env, policy_init_old, policy_init):
        r""" Constructor
        
        :param policy_init_old: initial :math:`\pi_{\theta_{\mathrm{old}}}`
        :param policy_init: initial :math:`\pi_\theta`    
        """
        self._env = env
        self._pi_old = policy_init_old
        self._pi = policy_init
        self._oblength = self._policy._oblength
        self._aclength = self._policy._aclength
    
    def _create_training_function(self, clip_param):
        r"""Construct the loss function.
        
        Based on the loss
        
        .. math::
            L^{\mathrm{CLIP}} (\theta) = \hat{\mathbb{E}}_t 
            \left[ 
            \min\left(r_t(\theta) \hat{A}_t, \mathrm{clip} (r_t(\theta),
            1-\epsilon,1+\epsilon) \hat{A}_t \right)           
            \right],
        
        where
        
        .. math::
           r_t (\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)},
            
        and "clip" simply bounds :math:`r_t(\theta)` between
        :math:`1-\epsilon` and :math:`1+\epsilon`.
        
        :param clip_param: the clipping parameter :math:`\epsilon`
        :type clip_param: float        
        """
        # Placeholders for batches of actions
        # This will hold a_t for the length of the optimization batch.
        ac_prob_batch = K.placeholder(shape=(None, self._aclength), name="ac_prob")
        ac_prob_old_batch = K.placeholder(shape=(None, self._aclength), name="ac_prob_old")
        ac_onehot = K.placeholder(shape=(None, self._aclength), name="ac_onehot")
        
        # Placeholder for a batch of advantages
        # This will hold \hat{A}_t for the length of the optimization batch.
        adv_batch = K.placeholder(shape=(None,), name="adv")
        
        # Placeholder for a batch of returns
        ret_batch = K.placeholder(shape=(None,), name="ret")
        
        # \pi_\theta(a_t|s_t)
        pi_a_given_s_batch = K.sum(ac_prob_batch*ac_onehot, axis=1)
        
        # \pi_{\theta_{\mathrm{old}}}(a_t|s_t)
        pi_old_a_given_s_batch = K.sum(ac_prob_old_batch*ac_onehot, axis=1)
        
        # r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
        r_t_batch = K.exp(K.log(pi_a_given_s_batch) - K.log(pi_old_a_given_s_batch))
        
        # The argument inside the loss L^{\mathrm{CPI}}, which is r_t \hat{A}_t
        l_cpi_arg_batch = r_t_batch*adv_batch
        
        # The clipped argument inside L^{\mathrm{CPI}}
        l_cpi_arg_clipped_batch = K.clip(r_t_batch, 1.0-clip_param, 1.0+clip_param)*adv_batch
        
        # The loss L^{\mathrm{CLIP}}
        l_clip = K.mean(K.minimum(l_cpi_arg_batch, l_cpi_arg_clipped_batch))
    def learn(self, max_iters=1024,
                    timesteps_per_batch=1024,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=8, optim_stepsize=1e-3, optim_batchsize=64,
                    gamma=0.99, lam=0.95):
        r""" Learns the policy.
        
        :param clip_param: the clipping parameter :math:`\epsilon`
        :type clip_param: float
        :returns:  None -- Nothing
        :raises:
        """
        
        # Set up the loss function
        self._create_training_function(clip_param)
        
        # Begin loop
        
        # Run old policy
        
        # Compute advantage estimates
        
        # Optimize
        
        # Set current policy to old policy
        
    def get_policy(self):
        return self._policy
