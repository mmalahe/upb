from upb.agents.mlp import load_mlp_agent
from upb.envs.UPEnv import UPEnv, UPObservationSpace, UPActionSpace
from upb.util.visualise import DecisionRenderer
import pickle
import numpy as np

def rollout(env, agents, render_decision_basename=None, callback=None):
    stochastic = True    
    
    if render_decision_basename != None:
        decision_renderer = DecisionRenderer(ob_plot_type='free_screenshot')
    
    ob = env.reset()
    ac_avail = env.getAvailableActions()
    done = False
    iter_num = 0
    stage_old = env.stage
    while not done:
        stage = env.stage
        
        # Observe again if stage changed
        if stage > stage_old:
            observation_from_handler, ob = env.observe(stage)
            ac_avail = np.array(env.getAvailableActions(stage))
        agent = agents[stage]
        
        # Select action
        ac, vpred = agent.act(stochastic, ob, ac_avail)
        
        # Render
        if iter_num > 0:
            if render_decision_basename != None:
                decision_filename = render_decision_basename+str(iter_num).zfill(5)+".png"
                decision_renderer.render(env, agent, ob, ac_avail, ac, vpred, rew, decision_filename)
            if callback != None:
                callback(iter_num, env, agent, ob, ac_avail, ac, vpred, rew, done, info)
        
        # Observe
        ob, rew, done, info = env._step(ac, stage=env.stage)        
        ac_avail = info['Available Actions']
        
        # Update            
        iter_num += 1
        stage_old = stage

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
