from upb.agents.mlp import load_mlp_agent
from upb.envs.UPEnv import UPEnv, UPObservationSpace, UPActionSpace
from upb.util.visualise import DecisionRenderer
import pickle

def rollout(env, agents, render_decision_basename=None, callback=None):
    ob_prev = env.reset()
    ac_avail = env.getAvailableActions()
    done = False
    stochastic = True
    iter_num = 0
    if render_decision_basename != None:
        decision_renderer = DecisionRenderer()
    
    while not done:
        agent = agents[env.stage]
        ac, vpred = agent.act(stochastic, ob_prev, ac_avail)
        if iter_num > 0:
            if render_decision_basename != None:
                decision_filename = render_decision_basename+str(iter_num).zfill(5)+".png"
                decision_renderer.render(env, agent, ob, ac_avail, ac, vpred, rew, decision_filename)
            if callback != None:
                callback(iter_num, env, agent, ob_prev, ac_avail, ac, vpred, rew, done, info)
        ob, rew, done, info = env._step(ac, stage=env.stage)        
        ac_avail = info['Available Actions']        
        ob_prev = ob        
        iter_num += 1

def load_agents(initial_stage, agent_filenames, base_name="agent"):
    agents = []
    for i in range(initial_stage):
        agent_filename = agent_filenames[i]
        agent_name = base_name+"_{}".format(i)
        ob_space = UPObservationSpace(UPEnv._observation_names_stages[i])
        ac_space = UPActionSpace(UPEnv._action_names_stages[i])        
        agent = load_mlp_agent(agent_filename, agent_name, ob_space, ac_space)
        agents.append(agent)
    return agents
    
def create_states_batch(env, agent, n_init_states, filename):
    init_states = []
    for i in range(n_init_states):
        print("Creating init state {}.".format(i))
        rollout(env, agent)
        env.save_screenshot("")
        init_states.append(env.getStateAsString())
    with open(filename, 'wb') as f:
        pickle.dump(init_states, f)
