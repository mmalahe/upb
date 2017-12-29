from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mp
from matplotlib import gridspec
import os

class DecisionRenderer:
    def __init__(self, ob_plot_type='embedded_screenshot', ac_plot_type='bar'):
        self._ob_plot_type = ob_plot_type
        self._ac_plot_type = ac_plot_type
        self._fig = None
        self._obs_prev = None
        self._acs_prev = None
        self._iter = 1
    
    def render(self, env, agent, ob, ac_avail, ac, vpred, rew, filename):
        # Get observations
        obs_dict = env.getCurrentObservationSpace().observationAsOrderedDict(ob)
        n_obs = len(obs_dict)
        
        # Get actions
        action_names = env.getCurrentActionSpace().allActionsAsStrings()
        n_actions = len(action_names)
        ac_probs = agent.getActionProbabilities(ob, ac_avail)
        selected_action = action_names[ac]
        
        # Make a fresh plot if observations or actions have changed
        new_plot = False
        if env.observation_space.getPossibleObservations() != self._obs_prev:
            self._obs_prev = env.observation_space.getPossibleObservations()
            new_plot = True
        if action_names != self._acs_prev:
            self._acs_prev = action_names
            new_plot = True
        
        # Set up plots
        if new_plot:
            if self._fig != None:
                plt.close(self._fig)
            dpi = 96
            self._ax_list = []
            if self._ob_plot_type == 'free_screenshot':
                base_xdim = 640
                base_ydim = 1080
                self._fig = plt.figure(figsize=(1.47*base_xdim/dpi,1.22*base_ydim/dpi), dpi=dpi)
                gs = gridspec.GridSpec(1, 1)
                self._ax_list.append(plt.subplot(gs[0]))
            else:
                base_xdim = 1920
                base_ydim = 1080
                self._fig = plt.figure(figsize=(1.47*base_xdim/dpi,1.22*base_ydim/dpi), dpi=dpi)
                gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])            
                self._ax_list.append(plt.subplot(gs[0]))
                self._ax_list.append(plt.subplot(gs[1]))
                plt.subplots_adjust(wspace=0.0)
        
        # Observations
        if self._ob_plot_type != 'free_screenshot':
            ob_ax = self._ax_list[0]
        
        if self._ob_plot_type == 'embedded_screenshot':
            env.save_screenshot("tmp.png")
            im = mpimg.imread("tmp.png")
            if new_plot:
                self._ob_handle = ob_ax.imshow(im, interpolation='lanczos')
                ob_ax.set_xticks([])
                ob_ax.set_xticklabels([])
                ob_ax.set_yticks([])
                ob_ax.set_yticklabels([])
            else:
                self._ob_handle.set_data(im)
                plt.draw()
            
        elif self._ob_plot_type == 'table':
            table_text = [["{:1.3g}".format(value)] for value in obs_dict.values()]
            if new_plot:
                row_labels = list(obs_dict.keys())
                col_labels = ["Value"]
                
                ob_ax.axis('tight')
                ob_ax.axis('off')
                self._ob_handle = ob_ax.table(cellText=table_text,
                                  rowLabels=row_labels,
                                  colLabels=col_labels,
                                  loc='center',
                                  colWidths=[0.2]
                                  )
            else:             
                for i_row in range(len(table_text)):
                    for i_col in range(len(table_text[0])):                    
                        self._ob_handle._cells[(i_row+1, i_col)]._text.set_text(table_text[i_row][i_col])
        
        elif self._ob_plot_type == 'free_screenshot':
            basedir = os.path.dirname(filename)
            screenshot_name = os.path.join(basedir, "free_screenshot{}.png".format(self._iter))
            env.save_screenshot(screenshot_name)    
        
        else:
            raise Exception("Don't know what to do with observation plot type {}.".format(self._ob_plot_type))
                    
        # Plot actions
        labels = action_names
        sizes = ac_probs
        if self._ob_plot_type != 'free_screenshot':
            ac_ax = self._ax_list[1]
        else:
            ac_ax = self._ax_list[0]
        
        if self._ac_plot_type == 'pie':
            # Chosen action slice "explodes" out of plot
            explode = [0 for i in range(len(ac_probs))]
            explode[ac] = 0.1
            
            # Exclude labels for probabilities that are too low, except for if they're chosen
            for i in range(len(ac_probs)):
                ac_prob = ac_probs[i]
                if i != ac and ac_prob < 0.05:
                    labels[i] = ''
            
            # Pie chart 
            patches, texts, autotexts = ac_ax.pie(sizes, explode=explode, labels=labels, autopct='',
                    shadow=False, startangle=90)
            texts[ac].set_weight('bold')
            ac_ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
        elif self._ac_plot_type == 'bar':
            # Bar graph
            y_pos = range(len(action_names))
            if new_plot:
                colors = ['black' for i in range(len(ac_probs))]
                colors[ac] = 'red'
                self._ac_handle = ac_ax.barh(y_pos, sizes, align='center', color=colors)
                ac_ax.set_xlabel('Probability')
                ac_ax.set_xlim(0.0,1.0)
                ac_ax.set_yticks(y_pos)
                ac_ax.set_yticklabels(labels)
                ac_ax.invert_yaxis()
            else:
                for i in range(len(ac_probs)):
                    self._ac_handle[i].set_width(ac_probs[i])
                    self._ac_handle[i].set_height
                    self._ac_handle[i].set_color('black')
                self._ac_handle[ac].set_color('red')       
            
        else:
            raise Exception("Don't know what to do with action plot type {}.".format(self._ac_plot_type))
                    
        # Save figure
        self._fig.savefig(filename, bbox_inches='tight', dpi='figure')
        self._iter += 1                 
