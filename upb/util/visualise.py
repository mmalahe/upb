from collections import OrderedDict
import matplotlib.pyplot as plt

def render_agent_decision(env, agent, ob, ac, vpred, rew, filename):    
    # Get observations
    obs_dict = env.observation_space.observationAsOrderedDict(ob)
    n_obs = len(obs_dict)
    
    # Get actions
    action_names = env.action_space.allActionsAsStrings()
    n_actions = len(action_names)
    ac_probs = agent.getActionProbabilities(ob)
    ac_prob_dict = OrderedDict()
    for i in range(n_actions):
        ac_prob_dict[action_names[i]] = ac_probs[i]
    selected_action = action_names[ac]
    
    # Plot action probabilities
    labels = action_names
    sizes = ac_probs
    explode = [0 for i in range(len(ac_prob_dict))]
    explode[ac] = 0.1
    
    # Exclude labels for probabilities that are too low, except for if they're chosen
    for i in range(len(ac_probs)):
        ac_prob = ac_probs[i]
        if i != ac and ac_prob < 0.05:
            labels[i] = ''
    
    # Plot
    fig1, ax1 = plt.subplots()
    patches, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='',
            shadow=False, startangle=90)
    #~ texts[ac].set_fontsize(20)
    texts[ac].set_weight('bold')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
