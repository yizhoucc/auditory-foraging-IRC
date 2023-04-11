import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.colors import ListedColormap

# class PlotHelper():
#     def __init__(self, num_steps, no_attention_modes, correct_lick_choice_idxs, wrong_lick_choice_idxs, attention_choice_idxs, obs_certainity_possible, num_states, figsize=(15, 3),bbox_to_anchor=(1.5, 1.05)):
#         self.num_steps = num_steps
#         self.no_attention_modes = no_attention_modes
#         self.correct_lick_choice_idxs = correct_lick_choice_idxs
#         self.wrong_lick_choice_idxs = wrong_lick_choice_idxs
#         self.attention_choice_idxs = attention_choice_idxs
#         self.obs_certainity_possible = obs_certainity_possible
#         self.num_states = num_states
#         self.figsize = figsize
#         self.bbox_to_anchor = bbox_to_anchor

#     def make_episode_plot(self, plot_variable, label, feature_color_list = [], for_belief = False, action_color_list = ['blue','blueviolet','indigo','magenta','darkcyan','cyan']):
#         fig_w, fig_h = self.figsize
#         aspect = self.num_steps*fig_h/fig_w*1.5
#         fig, ax = plt.subplots(figsize=self.figsize) 
#         attention_choice_lists = []
#         attention_choice_text = []
#         for attention_choice in range(self.no_attention_modes):
#             attention_choice_lists.append(ax.scatter(np.arange(self.num_steps+1)[self.attention_choice_idxs[attention_choice]], .3 * np.ones(sum(self.attention_choice_idxs[attention_choice])), color=action_color_list[attention_choice], marker='o', s=50))
#             attention_choice_text.append(f'Obs. conf. {self.obs_certainity_possible[attention_choice]}')
        
#         correct_lick_choice = ax.scatter(np.arange(self.num_steps+1)[self.correct_lick_choice_idxs], -0.3 * np.ones(sum(self.correct_lick_choice_idxs)), color='darkorange', marker='^', s=50)
#         wrong_lick_choice = ax.scatter(np.arange(self.num_steps+1)[self.wrong_lick_choice_idxs], -0.3 * np.ones(sum(self.wrong_lick_choice_idxs)), color='darkorange', marker='v', s=50)
#         feature_colors = ListedColormap(feature_color_list)
#         if for_belief:
#             h = ax.imshow(plot_variable.T, aspect=aspect, extent=[-0.5, self.num_steps+0.5, -0.5, 0.5], vmin=0, vmax=1, origin='lower', cmap='gist_gray',)
#             cbar = plt.colorbar(h, label=label)
#             cbar.set_ticks(np.arange(2))
#             ax.set_yticks(1/self.num_states * np.arange(self.num_states) - 0.5 + 0.5 * 1/self.num_states)
#             ax.set_yticklabels([f'{i}' for i in range(self.num_states)])
#         else:
#             h = ax.imshow(plot_variable.T, aspect=aspect, extent=[-0.5, self.num_steps+0.5, -0.5, 0.5], vmin=np.min(plot_variable) - 0.5, vmax=np.max(plot_variable) + 0.5, origin='lower', cmap=feature_colors)
#             cbar = plt.colorbar(h, label=label)
#             cbar.set_ticks(np.arange(np.min(plot_variable), np.max(plot_variable)+1))
#             ax.set_yticks([])
#             ax.set_ylabel('')
#         action_legend_list = [correct_lick_choice,wrong_lick_choice] + attention_choice_lists
#         ax.legend(action_legend_list, ['correct_lick_choice','wrong_lick_choice']+attention_choice_text, bbox_to_anchor=self.bbox_to_anchor, fontsize=12)
#         ax.set_xlim([-0.5, self.num_steps+0.5])
#         ax.set_xticks([0, self.num_steps])
#         ax.set_xlabel('Time')
#         return fig
    
#     #for episodic
#     def make_hist_plots(self, offset_actions, observations, states, no_signal_nodes, no_attention_modes):
#         flatten_observations = observations.flatten()
#         flatten_states = states.flatten()
#         fig_w, fig_h = self.figsize
#         fig, axs = plt.subplots(3,2,figsize=(fig_w, 6*fig_h))
#         axs[0,0].hist(offset_actions[np.flatnonzero(flatten_observations == 3)]%no_attention_modes, bins = np.arange(-.5,no_attention_modes,1))
#         axs[0, 0].set(xticks = range(no_attention_modes), xlabel = 'attention levels', ylabel = 'count', title = 'attention during ITI')
#         axs[0,1].hist((offset_actions[np.flatnonzero(flatten_observations == 3)]>= no_attention_modes).astype(int), bins = np.arange(-.5,2,1))
#         axs[0, 1].set(xticks = range(2), xlabel = 'lick choice', ylabel = 'count', title = 'lick during ITI')
#         axs[1, 0].hist(offset_actions[np.flatnonzero(flatten_states == 0)]%no_attention_modes, bins = np.arange(-.5,no_attention_modes,1))
#         axs[1, 0].set(xticks = range(no_attention_modes), xlabel = 'attention levels', ylabel = 'count', title = 'attention during noise')
#         axs[1, 1].hist((offset_actions[np.flatnonzero(flatten_states == 0)]>= no_attention_modes).astype(int), bins = np.arange(-.5,2,1))
#         axs[1, 1].set(xticks = range(2), xlabel = 'lick choice', ylabel = 'count', title = 'lick during noise')
#         axs[2, 0].hist(offset_actions[np.flatnonzero((flatten_states >= 1) & (flatten_states <= no_signal_nodes))]%no_attention_modes, bins = np.arange(-.5,no_attention_modes,1))
#         axs[2, 0].set(xticks = range(no_attention_modes), xlabel = 'attention levels', ylabel = 'count', title = 'attention during signal')
#         axs[2, 1].hist((offset_actions[np.flatnonzero((flatten_states >= 1) & (flatten_states <= no_signal_nodes))]>= no_attention_modes).astype(int), bins = np.arange(-.5,2,1))
#         axs[2, 1].set(xticks = range(2), xlabel = 'lick choice', ylabel = 'count', title = 'lick during signal')
#         return fig

# #for episodic
# def assign_state_class(true_state, no_signal_nodes, no_penalty_nodes):
#     if true_state == 0:
#         state_class = true_state
#     elif true_state <= no_signal_nodes:
#         state_class = 1
#     elif true_state <= no_signal_nodes + no_penalty_nodes:
#         state_class = 2
#     else:
#         state_class = 3
#     return state_class

# def plot_AF_episode(episode, env):        
#     obs_certainity_possible = env.obs_certainity_possible
#     dict_action_possible = env.dict_action_possible #{key: (lick_choice,attention_choice)}
#     no_attention_modes = env.no_attention_modes

#     #for episodic
#     no_signal_nodes = env.no_signal_nodes
#     no_penalty_nodes = env.no_penalty_nodes
#     no_ITI_nodes = env.no_ITI_nodes

#     licking_actions = [action_key for action_key in dict_action_possible if dict_action_possible[action_key][0] == 1]
#     attention_action_keys = []
#     for attention_choice in range(no_attention_modes):
#         attention_action_keys.append([action_key for action_key in dict_action_possible if dict_action_possible[action_key][1] == attention_choice])
#     num_steps = episode['num_steps']
#     states = episode['states']
#     observations = episode['observations']
#     actions = episode['actions']
#     rewards = episode['rewards']
#     probs = episode['q_probs']
#     num_states = states.max()+1

#     #Modeling assumption - Based on current observation, you choose whether to lick at the current state, and whether to attend at next state. 
#     offset_actions = np.insert(actions, -1, 0, axis=0) #offset actions to match time steps, as it looks like the action at last time step is not taken.
#     offset_rewards = np.insert(rewards, -1, 0, axis=0) #offset actions to match time steps, as it looks like the rewards at last time step is not computed.
#     correct_lick_choice_idxs = list(map(lambda action_choice: True if action_choice in licking_actions else False, offset_actions))&(offset_rewards>=0)
#     wrong_lick_choice_idxs = list(map(lambda action_choice: True if action_choice in licking_actions else False, offset_actions))&(offset_rewards<0)
#     attention_choice_idxs = []
#     for attention_choice in range(no_attention_modes):
#         attention_choice_idxs.append(list(map(lambda action_choice: True if action_choice in attention_action_keys[attention_choice] else False, offset_actions)))
#     env_plotter = PlotHelper(num_steps, no_attention_modes, correct_lick_choice_idxs, wrong_lick_choice_idxs, attention_choice_idxs, obs_certainity_possible, num_states)    
#     figs = []
    
#     #for episodic
#     # fig = env_plotter.make_episode_plot(plot_variable = states, label = 'True Sate', feature_color_list = ['khaki','green','limegreen','palegreen','lime','red','maroon'])
    
#     state_classes = np.array([[assign_state_class(true_state, no_signal_nodes, no_penalty_nodes)] for true_state in states.flatten()])
#     fig = env_plotter.make_episode_plot(plot_variable = state_classes, label = 'Sate class', feature_color_list = ['khaki','green','red','maroon'])
#     figs.append(fig)
#     fig = env_plotter.make_episode_plot(plot_variable = observations, label = 'Observation', feature_color_list = ['black','white','red','maroon'])
#     figs.append(fig)
#     fig = env_plotter.make_episode_plot(plot_variable = probs, label = 'Belief', for_belief = True)    
#     figs.append(fig)
#     fig = env_plotter.make_hist_plots(offset_actions, observations, states, no_signal_nodes, no_attention_modes)
#     figs.append(fig)
#     return figs



def gaussian_like_belief(belief_length, mu_length = 10, sigma_length = 10):
    mu_vector = np.linspace(0, belief_length-1, mu_length)
    sigma_vector = np.linspace(0.000001, (belief_length-1)/2, sigma_length)
    belief_matrix = np.zeros((mu_length, belief_length, sigma_length))
    for mu_ind in range(mu_length):
        for sigma_ind in range(sigma_length):
            for belief_ind in range(belief_length):
                belief_matrix[mu_ind, belief_ind, sigma_ind] = 1/np.sqrt(2 * np.pi * sigma_vector[sigma_ind]**2) * np.exp(-.5 * ((belief_ind-mu_vector[mu_ind])/sigma_vector[sigma_ind])**2)
            belief_matrix[mu_ind, :, sigma_ind] /= np.sum(belief_matrix[mu_ind, :, sigma_ind])
    return mu_vector, sigma_vector, belief_matrix