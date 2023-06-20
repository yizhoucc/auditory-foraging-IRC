import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import copy

class PlotHelper():
    def __init__(self, num_steps, no_attention_modes, correct_lick_choice_idxs, wrong_lick_choice_idxs, attention_choice_idxs, obs_certainity_possible, num_states, figsize=(15, 3),bbox_to_anchor=(1.5, 1.05)):
        self.num_steps = num_steps
        self.no_attention_modes = no_attention_modes
        self.correct_lick_choice_idxs = correct_lick_choice_idxs
        self.wrong_lick_choice_idxs = wrong_lick_choice_idxs
        self.attention_choice_idxs = attention_choice_idxs
        self.obs_certainity_possible = obs_certainity_possible
        self.num_states = num_states
        self.figsize = figsize
        self.bbox_to_anchor = bbox_to_anchor

    def make_episode_plot(self, plot_variable, label, feature_color_list = [], for_belief = False, attention_color_list = ['blue','blueviolet','indigo','magenta','darkcyan','cyan']):
        fig_w, fig_h = self.figsize
        aspect = self.num_steps*fig_h/fig_w*1.5
        fig, ax = plt.subplots(figsize=self.figsize) 
        attention_choice_lists = []
        attention_choice_text = []
        for attention_choice in range(self.no_attention_modes):
            attention_choice_lists.append(ax.scatter(np.arange(self.num_steps+1)[self.attention_choice_idxs[attention_choice]], .3 * np.ones(sum(self.attention_choice_idxs[attention_choice])), color=attention_color_list[attention_choice], marker='o', s=50))
            attention_choice_text.append(f'Obs. conf. {self.obs_certainity_possible[attention_choice]}')
        
        correct_lick_choice = ax.scatter(np.arange(self.num_steps+1)[self.correct_lick_choice_idxs], -0.3 * np.ones(sum(self.correct_lick_choice_idxs)), color='darkorange', marker='^', s=50)
        wrong_lick_choice = ax.scatter(np.arange(self.num_steps+1)[self.wrong_lick_choice_idxs], -0.3 * np.ones(sum(self.wrong_lick_choice_idxs)), color='darkorange', marker='v', s=50)
        feature_colors = ListedColormap(feature_color_list)
        if for_belief:
            h = ax.imshow(plot_variable.T, aspect=aspect, extent=[-0.5, self.num_steps+0.5, -0.5, 0.5], vmin=0, vmax=1, origin='lower', cmap='gist_gray',)
            cbar = plt.colorbar(h, label=label)
            cbar.set_ticks(np.arange(2))
            ax.set_yticks(1/self.num_states * np.arange(self.num_states) - 0.5 + 0.5 * 1/self.num_states)
            ax.set_yticklabels([f'{i}' for i in range(self.num_states)])
        else:
            h = ax.imshow(plot_variable.T, aspect=aspect, extent=[-0.5, self.num_steps+0.5, -0.5, 0.5], vmin=np.min(plot_variable) - 0.5, vmax=np.max(plot_variable) + 0.5, origin='lower', cmap=feature_colors)

            print('problem somwhere here')
            print(plot_variable)
            print(np.where(plot_variable == 2))


            cbar = plt.colorbar(h, label=label)
            cbar.set_ticks(np.arange(np.min(plot_variable), np.max(plot_variable)+1))
            ax.set_yticks([])
            ax.set_ylabel('')
        action_legend_list = [correct_lick_choice,wrong_lick_choice] + attention_choice_lists
        ax.legend(action_legend_list, ['correct_lick_choice','wrong_lick_choice']+attention_choice_text, bbox_to_anchor=self.bbox_to_anchor, fontsize=12)
        ax.set_xlim([-0.5, self.num_steps+0.5])
        ax.set_xticks([0, self.num_steps])
        ax.set_xlabel('Time')
        return fig
    
    #for episodic
    def make_hist_plots(self, offset_actions, observations, states, no_signal_nodes, no_attention_modes):
        flatten_observations = observations.flatten()
        flatten_states = states.flatten()
        fig_w, fig_h = self.figsize
        fig, axs = plt.subplots(3,2,figsize=(fig_w, 6*fig_h))
        axs[0,0].hist(offset_actions[np.flatnonzero(flatten_observations == 3)]%no_attention_modes, bins = np.arange(-.5,no_attention_modes,1))
        axs[0, 0].set(xticks = range(no_attention_modes), xlabel = 'attention levels', ylabel = 'count', title = 'attention during ITI')
        axs[0,1].hist((offset_actions[np.flatnonzero(flatten_observations == 3)]>= no_attention_modes).astype(int), bins = np.arange(-.5,2,1))
        axs[0, 1].set(xticks = range(2), xlabel = 'lick choice', ylabel = 'count', title = 'lick during ITI')
        axs[1, 0].hist(offset_actions[np.flatnonzero(flatten_states == 0)]%no_attention_modes, bins = np.arange(-.5,no_attention_modes,1))
        axs[1, 0].set(xticks = range(no_attention_modes), xlabel = 'attention levels', ylabel = 'count', title = 'attention during noise')
        axs[1, 1].hist((offset_actions[np.flatnonzero(flatten_states == 0)]>= no_attention_modes).astype(int), bins = np.arange(-.5,2,1))
        axs[1, 1].set(xticks = range(2), xlabel = 'lick choice', ylabel = 'count', title = 'lick during noise')
        axs[2, 0].hist(offset_actions[np.flatnonzero((flatten_states >= 1) & (flatten_states <= no_signal_nodes))]%no_attention_modes, bins = np.arange(-.5,no_attention_modes,1))
        axs[2, 0].set(xticks = range(no_attention_modes), xlabel = 'attention levels', ylabel = 'count', title = 'attention during signal')
        axs[2, 1].hist((offset_actions[np.flatnonzero((flatten_states >= 1) & (flatten_states <= no_signal_nodes))]>= no_attention_modes).astype(int), bins = np.arange(-.5,2,1))
        axs[2, 1].set(xticks = range(2), xlabel = 'lick choice', ylabel = 'count', title = 'lick during signal')
        return fig
    
    def policy_for_gaussian_beliefs(self, agent, no_nodes, dict_action_possible, licking_action_keys, attention_action_keys, no_signal_nodes, mu_max = None, sigma_max_factor = 6, mu_length = 20, sigma_length = 10):
        no_signal_and_noise_nodes = no_signal_nodes + 1
        mu_max = no_signal_and_noise_nodes-1 if mu_max == None else mu_max
        sigma_max = mu_max/sigma_max_factor
        mu_vector = np.linspace(0, mu_max, mu_length)
        sigma_vector = np.linspace(0.1, sigma_max, sigma_length)
        belief_matrix = np.zeros((mu_length, sigma_length, no_nodes))
        action_prob_matrix = np.zeros((mu_length, sigma_length, len(dict_action_possible)))
        lick_prob_matrix = np.zeros((mu_length, sigma_length))
        attention_prob_matrix = np.zeros((mu_length, sigma_length, len(attention_action_keys)))
        for mu_ind in range(mu_length):
            for sigma_ind in range(sigma_length):
                for belief_ind in range(0, no_signal_nodes+1):
                    belief_matrix[mu_ind, sigma_ind, belief_ind] = 1/np.sqrt(2 * np.pi * sigma_vector[sigma_ind]**2) * np.exp(-.5 * ((belief_ind-mu_vector[mu_ind])/sigma_vector[sigma_ind])**2)
                belief_matrix[mu_ind, sigma_ind, :] /= np.sum(belief_matrix[mu_ind, sigma_ind, :])
                action_prob_matrix[mu_ind, sigma_ind, :] = agent.agent_action_distribution(np.array([belief_matrix[mu_ind, sigma_ind, :]]))[0]
        for lick_key in licking_action_keys:
            lick_prob_matrix += action_prob_matrix[:, :, lick_key]
        for attention_choice in range(len(attention_action_keys)):
            for attention_keys in attention_action_keys[attention_choice]:
                attention_prob_matrix[:,:,attention_choice] += action_prob_matrix[:, :, attention_keys]
        
        fig_w, fig_h = self.figsize

        no_xticks = int(no_nodes/20)
        no_yticks = int(mu_length/5)
        no_rows_in_subplot = 10
        xticks = np.around(np.linspace(0, no_signal_and_noise_nodes-1, no_xticks)).astype(int)
        yticks = np.around(np.linspace(0, len(mu_vector)-1, no_yticks)).astype(int)
        fig1, axs = plt.subplots(no_rows_in_subplot,figsize=(1.5*fig_w, 15*fig_h))
        for sigma_ind in range(len(sigma_vector)):
            h = axs[sigma_ind].imshow(belief_matrix[:, sigma_ind, :no_signal_and_noise_nodes])
            cbar = plt.colorbar(h, label= 'prob.')
            cbar.set_ticks(np.round([np.min(belief_matrix[:, sigma_ind, :no_signal_and_noise_nodes]), np.max(belief_matrix[:, sigma_ind, :no_signal_and_noise_nodes])],4))
            axs[sigma_ind].set(xticks = xticks, xticklabels = np.around(np.arange(no_nodes)[xticks],1), xlabel = 'node index')
            axs[sigma_ind].set(yticks = yticks, yticklabels = np.around(mu_vector[yticks],1), ylabel = 'mean (node)')
            axs[sigma_ind].set(title = f's.t.d of {round(sigma_vector[sigma_ind],2)}')


        no_xticks = int(sigma_length/2)
        no_yticks = int(mu_length/2)
        no_rows_in_subplot = 2
        no_cols_in_subplot = 3
        xticks = np.around(np.linspace(0, len(sigma_vector)-1, no_xticks)).astype(int)
        yticks = np.around(np.linspace(0, len(mu_vector)-1, no_yticks)).astype(int)
        plt.figure()
        fig2 = plt.imshow(lick_prob_matrix)
        plt.colorbar(label= 'prob.', ticks = np.round([np.min(lick_prob_matrix), np.max(lick_prob_matrix)],2))
        plt.xlabel('s.t.d (node)')
        plt.ylabel('mean (node)')
        plt.xticks(ticks = xticks, labels = np.around(sigma_vector[xticks],1))
        plt.yticks(ticks = yticks, labels = np.around(mu_vector[yticks],1))
        plt.title(f'prob. of licking')

        
        fig3, axs = plt.subplots(no_rows_in_subplot,no_cols_in_subplot,figsize=(1.2*fig_w, 4.75*fig_h))
        for attention_choice in range(len(attention_action_keys)):
            row_ind = attention_choice//no_cols_in_subplot
            col_ind = attention_choice%no_cols_in_subplot
            h = axs[row_ind, col_ind].imshow(attention_prob_matrix[:,:,attention_choice])
            cbar = plt.colorbar(h, label= 'prob.', shrink = .7)
            cbar.set_ticks(np.round([np.min(attention_prob_matrix[:,:,attention_choice]), np.max(attention_prob_matrix[:,:,attention_choice])],2))
            axs[row_ind, col_ind].set(xticks = xticks, xticklabels = np.around(sigma_vector[xticks],1), xlabel = 's.t.d (node)')
            axs[row_ind, col_ind].set(yticks = yticks, yticklabels = np.around(mu_vector[yticks],1), ylabel = 'mean (node)')
            axs[row_ind, col_ind].set(title = f'prob. of attention {attention_choice}')
        
        return fig1, fig2, fig3

    def policy_for_all_signal_noise_durations(self, agent, env, states, probs, licking_action_keys, attention_action_keys, no_signal_nodes, no_additional_episodes = 0, attention_color_list = ['blue','blueviolet','indigo','magenta','darkcyan','cyan']):
        no_signal_and_noise_nodes = no_signal_nodes + 1
        episodes_states = states
        episodes_beliefs = probs
        for _ in range(no_additional_episodes):
            episode = agent.run_one_episode(env=env, num_steps=10000, q_states = [[i] for i in range(env.no_nodes)])
            episodes_states = np.concatenate((episodes_states,episode['states']))
            episodes_beliefs = np.concatenate((episodes_beliefs,episode['q_probs']))
        chosen_time = [i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes]
        chosen_beliefs = episodes_beliefs[chosen_time]
        sorted_indices = np.argsort(np.sum(chosen_beliefs[:,1:no_signal_and_noise_nodes],1))
        sorted_signal_prob = np.sum(chosen_beliefs[:,1:no_signal_and_noise_nodes],1)[sorted_indices]
        sorted_beliefs = chosen_beliefs[sorted_indices]

        action_prob_matrix = np.array([agent.agent_action_distribution(np.array([sorted_beliefs[index, :]]))[0] for index in range(len(sorted_beliefs))])
        lick_prob_matrix = np.zeros(len(sorted_beliefs))
        attention_prob_matrix = np.zeros((len(sorted_beliefs), len(attention_action_keys)))
        for lick_key in licking_action_keys:
            lick_prob_matrix += action_prob_matrix[:,lick_key]
        for attention_choice in range(len(attention_action_keys)):
            for attention_keys in attention_action_keys[attention_choice]:
                attention_prob_matrix[:,attention_choice] += action_prob_matrix[:, attention_keys]
        # plt.figure(figsize = self.figsize)
        # fig1 = plt.imshow(sorted_beliefs[:,0:no_signal_and_noise_nodes].T)
        # plt.colorbar(label= 'prob.', ticks = np.round([np.min(sorted_beliefs[:,:no_signal_and_noise_nodes]), np.max(sorted_beliefs[:,:no_signal_and_noise_nodes])],2), shrink = .5)
        # plt.xlabel('belief index')
        # plt.ylabel('node index')
        # plt.title(f'Sampled beliefs')
        # plt.figure(figsize = self.figsize)
        # fig2 = plt.plot(sorted_signal_prob, 'o-')
        # plt.xlabel('belief index')
        # plt.ylabel('signal probability')
        plt.figure()
        fig3 = plt.plot(sorted_signal_prob, lick_prob_matrix, 'o-', color = 'r')
        plt.xlabel('signal belief')
        plt.title(f'prob. of licking')
        for attention_choice in range(len(attention_action_keys)):
            plt.plot(sorted_signal_prob, attention_prob_matrix[:,attention_choice], 'o-', color = attention_color_list[attention_choice])
        plt.legend(['lick']+[f'attention {attention_choice}' for attention_choice in range(len(attention_action_keys))], bbox_to_anchor=(1.5, 1.05), fontsize=12)
        plt.xlabel('signal prob.')
        plt.ylabel('prob.')
        plt.title('')
        plt.figure(figsize = self.figsize)
        # fig4 = plt.plot(lick_prob_matrix, 'o-', color = 'r')
        # plt.xlabel('belief index')
        # plt.title(f'prob. of licking')
        # for attention_choice in range(len(attention_action_keys)):
        #     plt.plot(attention_prob_matrix[:,attention_choice], 'o-', color = attention_color_list[attention_choice])
        # plt.legend(['lick']+[f'attention {attention_choice}' for attention_choice in range(len(attention_action_keys))], bbox_to_anchor=(1.5, 1.05), fontsize=12)
        # plt.xlabel('belief index')
        # plt.ylabel('prob.')
        # plt.title('')
        # return fig1, fig2, fig3, fig4
        return fig3

    def count_success_streak(self, agent, env, states, no_signal_nodes, no_additional_episodes = 10):
        no_signal_and_noise_nodes = no_signal_nodes + 1
        episodes_states = states
        success_streak_list = [len([i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes and episodes_states[i-1] >= no_signal_and_noise_nodes])]
        for _ in range(no_additional_episodes):
            episode = agent.run_one_episode(env=env, num_steps=10000, q_states = [[i] for i in range(env.no_nodes)])
            episodes_states = episode['states']
            episodes_beliefs = episode['q_probs']
            success_streak_list.append(len([i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes and episodes_states[i-1] >= no_signal_and_noise_nodes]))
            average_success_streak = sum(success_streak_list)/(no_additional_episodes+1)
        return average_success_streak, success_streak_list
    
    def policy_for_signal_noise_durations(self, agent, states, observations, probs, licking_action_keys, attention_action_keys, no_signal_nodes, nodes_from_zero, time_steps_before_lick, attention_color_list = ['blue','blueviolet','indigo','magenta','darkcyan','cyan']):
        no_signal_and_noise_nodes = no_signal_nodes + 1
        episodes_states = states
        episodes_beliefs = probs
        chosen_start_time = [i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes and episodes_states[i-1] >= no_signal_and_noise_nodes]
        chosen_end_time = [i for i in range(len(episodes_states)) if episodes_states[i] < no_signal_and_noise_nodes and episodes_states[i+1] >= no_signal_and_noise_nodes]
        no_acquisitions = len(chosen_start_time)
        if no_acquisitions == 1:
            return None
        fig_w, fig_h = self.figsize
        fig, axs = plt.subplots(no_acquisitions, 2, figsize=(1.5*fig_w, 15*fig_h))
        for acquisition_no in range(no_acquisitions):
            start_time = chosen_start_time[acquisition_no]
            end_time = chosen_end_time[acquisition_no]
            chosen_beliefs = episodes_beliefs[start_time:end_time+1][-time_steps_before_lick:]
            
            # chosen_observations = observations[start_time:end_time+1][-time_steps_before_lick:]
            # obs_favoring_noise = [1 if val == 0 else 0 for val in chosen_observations]
            
            action_prob_matrix = np.array([agent.agent_action_distribution(np.array([chosen_beliefs[index, :]]))[0] for index in range(len(chosen_beliefs))])
            lick_prob_matrix = np.zeros(len(chosen_beliefs))
            attention_prob_matrix = np.zeros((len(chosen_beliefs), len(attention_action_keys)))
            for lick_key in licking_action_keys:
                lick_prob_matrix += action_prob_matrix[:,lick_key]
            for attention_choice in range(len(attention_action_keys)):
                for attention_keys in attention_action_keys[attention_choice]:
                    attention_prob_matrix[:,attention_choice] += action_prob_matrix[:, attention_keys]
            h = axs[acquisition_no, 0].imshow(chosen_beliefs[:,0:nodes_from_zero].T)
            plt.colorbar(h, label= 'prob.', shrink = .7)
            axs[acquisition_no, 0].set(xticks = np.arange(time_steps_before_lick), xticklabels = [str(-i) for i in range(time_steps_before_lick-1,-1,-1)], xlabel = 'time')
            axs[acquisition_no, 0].set(yticks = np.arange(nodes_from_zero), ylabel = 'node index')
            axs[acquisition_no, 0].set(title = f'Beliefs before licking')
            
            axs[acquisition_no, 1].plot(lick_prob_matrix, 'v-', color = 'r')
            for attention_choice in range(len(attention_action_keys)):
                axs[acquisition_no, 1].plot(attention_prob_matrix[:,attention_choice], 'o-', color = attention_color_list[attention_choice])
            axs[acquisition_no, 1].plot(np.sum(chosen_beliefs[:,1:no_signal_and_noise_nodes],1), '*-')
            # axs[acquisition_no, 1].stem(obs_favoring_noise, '--')
            axs[acquisition_no, 1].set(xticks = np.arange(time_steps_before_lick), xticklabels = [str(-i) for i in range(time_steps_before_lick-1,-1,-1)], xlabel = 'time')
            axs[acquisition_no, 1].set(yticks = np.arange(0,1,.1), ylabel = 'prob.')
            axs[acquisition_no, 1].set(title = f'Policy before licking')
            axs[acquisition_no, 1].legend(['lick']+[f'attention {attention_choice}' for attention_choice in range(len(attention_action_keys))]+['signal prob.'], bbox_to_anchor=(1.5, 1.05), fontsize=12)
        return fig
    

#for episodic
def assign_state_class(true_state, no_signal_nodes, no_penalty_nodes):
    if true_state == 0:
        state_class = true_state
    elif true_state <= no_signal_nodes:
        state_class = 1
    elif true_state <= no_signal_nodes + no_penalty_nodes:
        state_class = 2
    else:
        state_class = 3
    return state_class


def plot_AF_episode(episode, env, agent, nodes_from_zero = 20, time_steps_before_lick = 10):        
    obs_certainity_possible = env.obs_certainity_possible
    dict_action_possible = env.dict_action_possible #{key: (lick_choice,attention_choice)}
    no_attention_modes = env.no_attention_modes

    #for episodic
    no_signal_nodes = env.no_signal_nodes
    no_penalty_nodes = env.no_penalty_nodes
    no_ITI_nodes = env.no_ITI_nodes
    no_nodes = env.no_nodes

    licking_action_keys = [action_key for action_key in dict_action_possible if dict_action_possible[action_key][0] == 1]
    attention_action_keys = []
    for attention_choice in range(no_attention_modes):
        attention_action_keys.append([action_key for action_key in dict_action_possible if dict_action_possible[action_key][1] == attention_choice])
    
    num_steps = episode['num_steps']
    states = episode['states']
    observations = episode['observations']
    actions = episode['actions']
    rewards = episode['rewards']
    probs = episode['q_probs']
    num_states = states.max()+1

    attention_color_list = ['blue','blueviolet','indigo','magenta','darkcyan','cyan']

    #Modeling assumption - Based on current observation, you choose whether to lick at the current state, and whether to attend at next state. 
    # offset_actions = np.insert(actions, -1, 0, axis=0) #offset actions to match time steps, as it looks like the action at last time step is not taken.
    # offset_rewards = np.insert(rewards, -1, 0, axis=0) #offset actions to match time steps, as it looks like the rewards at last time step is not computed.
    offset_actions = np.append(actions, 0) #offset actions to match time steps, as it looks like the action at last time step is not taken.
    offset_rewards = np.append(rewards, 0) #offset actions to match time steps, as it looks like the rewards at last time step is not computed.
    correct_lick_choice_idxs = list(map(lambda action_choice: True if action_choice in licking_action_keys else False, offset_actions))&(offset_rewards>=0)
    wrong_lick_choice_idxs = list(map(lambda action_choice: True if action_choice in licking_action_keys else False, offset_actions))&(offset_rewards<0)
    attention_choice_idxs = []
    for attention_choice in range(no_attention_modes):
        attention_choice_idxs.append(list(map(lambda action_choice: True if action_choice in attention_action_keys[attention_choice] else False, offset_actions)))
    env_plotter = PlotHelper(num_steps, no_attention_modes, correct_lick_choice_idxs, wrong_lick_choice_idxs, attention_choice_idxs, obs_certainity_possible, num_states)    
    figs = []
    
    #for episodic
    ## fig = env_plotter.make_episode_plot(plot_variable = states, label = 'True Sate', feature_color_list = ['khaki','green','limegreen','palegreen','lime','red','maroon'])
    state_classes = np.array([[assign_state_class(true_state, no_signal_nodes, no_penalty_nodes)] for true_state in states.flatten()])
    fig = env_plotter.make_episode_plot(plot_variable = state_classes, label = 'State class', feature_color_list = ['khaki','green','red','maroon'], attention_color_list = attention_color_list)
    figs.append(fig)
    fig = env_plotter.make_episode_plot(plot_variable = observations, label = 'Observation', feature_color_list = ['black','white','red','maroon'], attention_color_list = attention_color_list)
    figs.append(fig)
    fig = env_plotter.make_episode_plot(plot_variable = probs, label = 'Belief', for_belief = True, attention_color_list = attention_color_list)    
    figs.append(fig)

    # fig = env_plotter.policy_for_all_signal_noise_durations(agent, env, states, probs, licking_action_keys, attention_action_keys, no_signal_nodes)
    # figs.append(fig)
    
    # fig = env_plotter.policy_for_signal_noise_durations(agent, states, observations, probs, licking_action_keys, attention_action_keys, no_signal_nodes, nodes_from_zero = nodes_from_zero, time_steps_before_lick = time_steps_before_lick)
    # figs.append(fig)

    # average_success_streak, success_streak_list = env_plotter.count_success_streak(agent, env, states, no_signal_nodes, no_additional_episodes = 20)
    # print(f'average success streak is {average_success_streak}')
    # plt.figure()
    # plt.stem(1 + np.arange(0,len(success_streak_list)),success_streak_list)
    # plt.xlabel('episode no.')
    # plt.ylabel('success streak')
    # plt.show()

    # fig1, fig2, fig3 = env_plotter.policy_for_gaussian_beliefs(agent, no_nodes, dict_action_possible, licking_action_keys, attention_action_keys, no_signal_nodes, mu_max = 20, mu_length = 20)
    # figs.append(fig1)
    # figs.append(fig2)
    # figs.append(fig3)
    
    # fig = env_plotter.make_hist_plots(offset_actions, observations, states, no_signal_nodes, no_attention_modes)
    # figs.append(fig)

    # fig1, fig2, fig3, fig4 = env_plotter.policy_for_all_signal_noise_durations(agent, env, states, probs, licking_action_keys, attention_action_keys, no_signal_nodes)
    # figs.append(fig1)
    # figs.append(fig2)
    # figs.append(fig3)
    # figs.append(fig4)
    return figs

def particle_filter(agent, env, lick_actions, state_list, no_particles = 10, sampling_freq = 1):

        # code from agent.py in irc package
        _to_restore_train = agent.algo.policy.training # policy will be set to evaluation mode temporarily
        agent.algo.policy.set_training_mode(False)
        
        observation_matrix = env.find_observation_matrix()
        
        env.state = state_list[0]
        observation = env.observe_step(0) # attention choice shouldn't matter, assuming we start with ITI.
        belief = env.init_belief(observation)
        belief_list = [[belief] for _ in range(no_particles)]
        observation = observation[0]
        particle_observation_prob = 1 # assuming we start with a fully observable state (ITI).
        observation_list = [[[observation]] for _ in range(no_particles)]        
        particles_distribution = 1/no_particles * np.ones(no_particles)
        particles_likelihoods = np.ones(no_particles)
        action_list = [[] for _ in range(no_particles)]

        sampling_count_tracker = {} #key are the time indices where additional sampling was done, and values represent number of times. 
        
        for time in range(len(lick_actions)):
            if time != len(lick_actions) - 1: env.state = state_list[time + 1]
            

            # print(time)

            step_particle = True
            sampling_count = 0

            while step_particle:
                
                instant_likelihood = []

                sampling_count += 1
                
                for particle in range(no_particles):
                    action, _ = agent.algo.predict(belief_list[particle][-1]) # might have to have this in tensor
                    action_list[particle].append(action.item())
                    instant_action_probs = agent.agent_action_distribution(np.array([belief_list[particle][-1]]))[0]
                    if lick_actions[time] == 1:
                        if action >= env.no_attention_modes:
                            particle_action_prob = instant_action_probs[action]/sum(instant_action_probs[env.no_attention_modes:])
                        else:
                            particle_action_prob = 0
                    elif lick_actions[time] == 0:
                        if action >= env.no_attention_modes:
                            particle_action_prob = 0
                        else: 
                            particle_action_prob = instant_action_probs[action]/sum(instant_action_probs[:env.no_attention_modes])
                    else:
                        raise Exception("Lick actions can only be 0 or 1.")
                    instant_likelihood.append(particle_observation_prob * particle_action_prob)
                    _, attention_choice = env.dict_action_possible[int(action)]
                    
                    if time != len(lick_actions) - 1: 
                        observation = env.observe_step(attention_choice)[0] # check if [0] is required, depending on observer_step in new env code. Also note how observe_step comes after env.step.
                        observation_list[particle].append([observation])
                        particle_observation_prob = observation_matrix[observation, env.state, attention_choice]
                        next_belief = env.update_belief(belief_list[particle][-1], action, observation)
                        belief_list[particle].append(next_belief)
                        if next_belief is None and particle_action_prob != 0:
                            raise Exception('Error: Liklihood should have been zero when wrong belief update happens!')
                
                if sum(instant_likelihood) == 0:
                    # print(f'Need to sample again for time {time}')
                    for particle_ind in range(len(belief_list)):
                        action_list[particle_ind] = action_list[particle_ind][:-1]
                        observation_list[particle_ind] = observation_list[particle_ind][:-1]
                        belief_list[particle_ind] = belief_list[particle_ind][:-1]
                else:
                    step_particle = False
            
            if sampling_count > 1:
                sampling_count_tracker[time] = sampling_count
            
            particles_likelihoods = np.multiply(particles_likelihoods, np.array(instant_likelihood))
            particles_distribution = np.multiply(particles_distribution, np.array(instant_likelihood))
            particles_distribution = particles_distribution/np.sum(particles_distribution)
            
            if time%sampling_freq == 0:
                temp_observation_list = [[] for _ in range(no_particles)]
                temp_belief_list = [[] for _ in range(no_particles)]
                temp_action_list = [[] for _ in range(no_particles)]
                temp_particles_likelihoods = [[] for _ in range(no_particles)]
                for particle in range(no_particles):
                    chosen_particle = np.random.choice(no_particles, p = particles_distribution)
                    temp_observation_list[particle] = copy.deepcopy(observation_list[chosen_particle])
                    temp_belief_list[particle] = copy.deepcopy(belief_list[chosen_particle])
                    temp_action_list[particle] = copy.deepcopy(action_list[chosen_particle])
                    temp_particles_likelihoods[particle] = particles_likelihoods[chosen_particle]
                observation_list = copy.deepcopy(temp_observation_list)
                belief_list = copy.deepcopy(temp_belief_list)
                action_list = copy.deepcopy(temp_action_list)
                particles_likelihoods = np.array(temp_particles_likelihoods)
                particles_distribution = 1/no_particles * np.ones(no_particles)
        
        # code from agent.py in irc package
        agent.algo.policy.set_training_mode(_to_restore_train) 
        
        particle_filter_output = {}
        generated_episodes = []
        sorted_indices = np.argsort(-1 * particles_likelihoods)
        episode_states = np.array([[state] for state in state_list])
        for ind in sorted_indices:
            temp_dict = {}
            temp_dict['states'] = episode_states
            temp_dict['actions'] = np.array(action_list[ind])
            temp_dict['observations'] = np.array(observation_list[ind])
            temp_dict['q_probs'] = np.array(belief_list[ind])
            temp_dict['num_steps'] =  len(temp_dict['actions'])
            generated_episodes.append(temp_dict)
        particle_filter_output['generated_episodes'] = generated_episodes
        particle_filter_output['particles_likelihoods'] = particles_likelihoods[sorted_indices]
        particle_filter_output['sampling count tracker'] = sampling_count_tracker
        
        return particle_filter_output