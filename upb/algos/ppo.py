import keras
import keras.backend as K

class ProximalPolicyOptimization(object):
    """An implementation of "Proximal Policy Optimization Algorithms"
    
    This implementation is built around policies that are represented
    as keras models.
    """
    def __init__(self, env, agent_init_old, agent_init):
        r""" Constructor
        
        :param agent_init_old: initial agent with policy :math:`\pi_{\theta_{\mathrm{old}}}`
        :param agent_init: initial agent with :math:`\pi_\theta`    
        """
        self._env = env
        self._ag_old = policy_init_old
        self._ag = policy_init
        self._oblength = self._policy._oblength
        self._aclength = self._policy._aclength
    
    def _create_training_function(self, clip_param):
        r"""Construct the loss function.
        
        Based on the objective
        
        .. math::
            L^{\mathrm{CLIP+VF+S}} (\theta) = \hat{\mathbb{E}}_t 
            \left[
                L^{\mathrm{CLIP}}_t -c_1 L^{\mathrm{VF}}_t
                + c_2 S[\pi_\theta] (s_t)
            \right],
        where
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
        ac_onehot_batch = K.placeholder(shape=(None, self._aclength), name="ac_onehot")
        
        # Placeholder for a batch of advantages
        # This will hold \hat{A}_t for the length of the optimization batch.
        adv_batch = K.placeholder(shape=(None,), name="adv")
        
        # Placeholder for a batch of returns
        ret_batch = K.placeholder(shape=(None,), name="ret")
        
        # Placeholders for value predictions and targets
        vpred_batch = K.placeholder(shape=(None,), name="vpred")
        vtarg_batch = K.placeholder(shape=(None,), name="vtarg")
        
        # \pi_\theta(a_t|s_t)
        pi_a_given_s_batch = K.sum(ac_prob_batch*ac_onehot_batch, axis=1)
        
        # \pi_{\theta_{\mathrm{old}}}(a_t|s_t)
        pi_old_a_given_s_batch = K.sum(ac_prob_old_batch*ac_onehot_batch, axis=1)
        
        # r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
        r_t_batch = K.exp(K.log(pi_a_given_s_batch) - K.log(pi_old_a_given_s_batch))
        
        # L^{\mathrm{CPI}}_t = r_t \hat{A}_t
        l_cpi_t_batch = r_t_batch*adv_batch
        
        # L^{\mathrm{CPI}}_t with clipped reward
        l_cpi_t_clipped_batch = K.clip(r_t_batch, 1.0-clip_param, 1.0+clip_param)*adv_batch
        
        # The objective L^{\mathrm{CLIP}}
        l_clip = K.mean(K.minimum(l_cpi_t_batch, l_cpi_t_clipped_batch))
        
        # Value function error term L^{\mathrm{VF}}
        l_vf = K.sum(K.square(vpred_batch-vtarg_batch))
        
        # Entropy bonus S[\pi_{\theta}]        
        pi_entropy = -K.sum(pi_a_given_s_batch*K.log(pi_a_given_s_batch))
        
        # The final loss L^{\mathrm{CLIP+VF+S}}
        c_1 = 1.0
        c_2 = 0.01
        l_clip_vf_s = l_clip - c_1*l_vf - c_2*pi_entropy        
        
        # Set up optimizer
        # @todo: make sure signs are correct
        adam = keras.optimizers.Adam()
        updates = adam.get_updates(params=[self._ag._policy.trainable_weights,
                                           self._ag._value_fn.trainable_weights
                                           ]
                                   constraints=[],
                                   loss=l_clip_vf_s)
        
        # Set up training function
        self.training_fn = backend.function(inputs=[ac_prob_batch,
                                                    ac_prob_old_batch,
                                                    ac_onehot_batch,
                                                    adv_batch,
                                                    ret_batch,
                                                    vpred_batch,
                                                    vtarg_batch
                                                    ]
                                            outputs=[],
                                            updates=updates)
    
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
        
        # Run old policy for a batch
        
        # INPUTS TO TRAINING FUNCTION #
        ###############################
        
        # Action probabilities for current policy: fetch
        #~ ac_prob_batch = K.placeholder(shape=(None, self._aclength), name="ac_prob")
        
        # Action probabilities for old policy: fetch
        #~ ac_prob_old_batch = K.placeholder(shape=(None, self._aclength), name="ac_prob_old")
        
        # Actions taken by current policy: fetch
        #~ ac_onehot = K.placeholder(shape=(None, self._aclength), name="ac_onehot")
        
        # Advantages: calculate with Mnih estimate
        #~ adv_batch = K.placeholder(shape=(None,), name="adv")
        
        # Returns: fetch
        #~ ret_batch = K.placeholder(shape=(None,), name="ret")
        
        # Value predictions: fetch
        #~ vpred_batch = K.placeholder(shape=(None,), name="vpred")
        
        # Value targets: Estimate with TD-Lambda
        #~ vtarg_batch = K.placeholder(shape=(None,), name="vtarg")
        
        # TRAIN #
        #########
        
        #~ self.training_fn(...)
        
        # UPDATE #
        ##########
        
        # Set old = new
        
    def get_policy(self):
        return self._policy
