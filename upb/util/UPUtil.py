from upb.agents.mlp import load_mlp_agent
from upb.envs.UPEnv import UPEnv, UPObservationSpace, UPActionSpace
import pickle

def rollout(env, agent, callback=None):
    ob_prev = env.reset()
    ac_avail = env.getAvailableActions()
    done = False
    stochastic = True
    iter_num = 0
    while not done:                
        ac, vpred = agent.act(stochastic, ob_prev, ac_avail)
        if iter_num > 0:
            if callback != None:
                callback(iter_num, env, agent, ob_prev, ac_avail, ac, vpred, rew, done, info)
        ob, rew, done, info = env.step(ac)        
        ac_avail = info['Available Actions']        
        ob_prev = ob        
        iter_num += 1
    
def load_resetter_agents(initial_stage, resetter_agent_filenames):
    resetter_agents = []
    for i in range(initial_stage):
        resetter_agent_filename = resetter_agent_filenames[i]
        agent_name = "resetter_agent_stage{}".format(i)
        ob_space = UPObservationSpace(UPEnv._observation_names_stages[i])
        ac_space = UPActionSpace(UPEnv._action_names_stages[i])        
        agent = load_mlp_agent(resetter_agent_filename, agent_name, ob_space, ac_space)
        resetter_agents.append(agent)
    return resetter_agents
    
def create_states_batch(env, agent, n_init_states, filename):
    init_states = []
    for i in range(n_init_states):
        print("Creating init state {}.".format(i))
        rollout(env, agent)
        env.save_screenshot("")
        init_states.append(env.getStateAsString())
    with open(filename, 'wb') as f:
        pickle.dump(init_states, f)
