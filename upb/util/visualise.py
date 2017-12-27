from collections import OrderedDict
import matplotlib.pyplot as plt

def render_agent_decision(env, agent, ob, ac_avail, ac, vpred, rew, filename):    
    # Get observations
    obs_dict = env.observation_space.observationAsOrderedDict(ob)
    n_obs = len(obs_dict)
    
    # Get actions
    action_names = env.action_space.allActionsAsStrings()
    n_actions = len(action_names)
    ac_probs = agent.getActionProbabilities(ob, ac_avail)
    ac_prob_dict = OrderedDict()
    for i in range(n_actions):
        ac_prob_dict[action_names[i]] = ac_probs[i]
    selected_action = action_names[ac]
    
    # Set up plots
    fig, ax_list = plt.subplots(ncols=2, figsize=(16,12))
    
    # Observations
    ob_ax = ax_list[0]
    row_labels = list(obs_dict.keys())
    col_labels = ["Value"]
    table_text = [["{:1.3g}".format(value)] for value in obs_dict.values()]
    ob_ax.axis('tight')
    ob_ax.axis('off')
    the_table = ob_ax.table(cellText=table_text,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center',
                      colWidths=[0.2]
                      )  
    
    
    # Actions
    labels = action_names
    sizes = ac_probs
    explode = [0 for i in range(len(ac_prob_dict))]
    explode[ac] = 0.1
    
    # Exclude labels for probabilities that are too low, except for if they're chosen
    for i in range(len(ac_probs)):
        ac_prob = ac_probs[i]
        if i != ac and ac_prob < 0.05:
            labels[i] = ''
    
    # Plot actions
    ac_ax = ax_list[1]
    patches, texts, autotexts = ac_ax.pie(sizes, explode=explode, labels=labels, autopct='',
            shadow=False, startangle=90)
    texts[ac].set_weight('bold')
    ac_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Make rooom for pie chart labels
    #~ plt.subplots_adjust(wspace=0.5, right=0.8)
    
    # Save figure and close
    plt.savefig(filename, bbox_inches='tight')
    #~ plt.savefig(filename)
    plt.close()
