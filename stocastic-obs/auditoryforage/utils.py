import matplotlib.pyplot as plt
import numpy as np
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

def plot_auditory_foraging_v1_episode(episode, num_shades=None, num_states=None, figsize=(15, 3),bbox_to_anchor=(1.5, 1.05)):

    # action 0 impllies no lick and no attention
    # action 1 impllies no lick and yes attention
    # action 2 impllies yes lick and no attention
    # action 3 impllies yes lick and yes attention

    num_steps = episode['num_steps']
    states = episode['states']
    observations = episode['observations']
    actions = episode['actions']
    rewards = episode['rewards']
    #Lokesh changed temporarily
    # assert not np.any(episode['q_states']-np.array([(1,)]))
    probs = episode['q_probs']
    beliefs = episode['beliefs']
    
    if num_shades is None:
        num_shades = observations.max()
    if num_states is None:
        num_states = states.max()+1

    #Modeling assumption - Based on current observation, you choose whether to lick at the current state, and whether to attend at next state. 
    offset_actions = np.insert(actions, -1, 0, axis=0) #offset actions to match time steps, as it looks like the action at last time step is not taken.
    offset_rewards = np.insert(rewards, -1, 0, axis=0) #offset actions to match time steps, as it looks like the rewards at last time step is not computed.
    correct_lick_choice_idxs = ((offset_actions==2)|(offset_actions==3))&(offset_rewards>=0)
    wrong_lick_choice_idxs = ((offset_actions==2)|(offset_actions==3))&(offset_rewards<0)
    attention_choice_idxs = ((offset_actions==1)|(offset_actions==3))

    fig_w, fig_h = figsize
    aspect = num_steps*fig_h/fig_w*1.5
    figs = []


    fig, ax = plt.subplots(figsize=figsize)
    attention_choice = ax.scatter(np.arange(num_steps+1)[attention_choice_idxs], np.zeros(sum(attention_choice_idxs)), color='blue', marker='o', s=50)
    correct_lick_choice = ax.scatter(np.arange(num_steps+1)[correct_lick_choice_idxs], np.zeros(sum(correct_lick_choice_idxs)), color='magenta', marker='^', s=50)
    wrong_lick_choice = ax.scatter(np.arange(num_steps+1)[wrong_lick_choice_idxs], np.zeros(sum(wrong_lick_choice_idxs)), color='magenta', marker='v', s=50)

    cmap_manual =   ListedColormap(['yellow','green','limegreen','palegreen','lime','red','maroon'])
    h = ax.imshow(
        states.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=np.min(states) - 0.5, vmax=np.max(states) + 0.5, origin='lower', cmap=cmap_manual,
    )


    cbar = plt.colorbar(h, label='True Sate')
    cbar.set_ticks(np.arange(np.min(states), np.max(states)+1))
    ax.legend([correct_lick_choice,wrong_lick_choice,attention_choice], ['correct_lick_choice','wrong_lick_choice','attention_choice'], bbox_to_anchor=bbox_to_anchor, fontsize=12)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks([])
    ax.set_xlabel('Time')
    ax.set_ylabel('')
    figs.append(fig)
    
    

    fig, ax = plt.subplots(figsize=figsize)
    attention_choice = ax.scatter(np.arange(num_steps+1)[attention_choice_idxs], np.zeros(sum(attention_choice_idxs)), color='blue', marker='o', s=50)
    correct_lick_choice = ax.scatter(np.arange(num_steps+1)[correct_lick_choice_idxs], np.zeros(sum(correct_lick_choice_idxs)), color='magenta', marker='^', s=50)
    wrong_lick_choice = ax.scatter(np.arange(num_steps+1)[wrong_lick_choice_idxs], np.zeros(sum(wrong_lick_choice_idxs)), color='magenta', marker='v', s=50)
    cmap_manual =   ListedColormap(['black','grey','white','red','maroon'])
    h = ax.imshow(
        observations.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=np.min(observations) - 0.5, vmax=np.max(observations) + 0.5, origin='lower', cmap=cmap_manual,
    )


    cbar = plt.colorbar(h, label='Observation')
    cbar.set_ticks(np.arange(np.min(observations), np.max(observations)+1))
    ax.legend([correct_lick_choice,wrong_lick_choice,attention_choice], ['correct_lick_choice','wrong_lick_choice','attention_choice'], bbox_to_anchor=bbox_to_anchor, fontsize=12)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks([])
    ax.set_xlabel('Time')
    ax.set_ylabel('')
    figs.append(fig)

    
    fig, ax = plt.subplots(figsize=figsize)
    attention_choice = ax.scatter(np.arange(num_steps+1)[attention_choice_idxs], np.zeros(sum(attention_choice_idxs)), color='blue', marker='o', s=50)
    correct_lick_choice = ax.scatter(np.arange(num_steps+1)[correct_lick_choice_idxs], np.zeros(sum(correct_lick_choice_idxs)), color='magenta', marker='^', s=50)
    wrong_lick_choice = ax.scatter(np.arange(num_steps+1)[wrong_lick_choice_idxs], np.zeros(sum(wrong_lick_choice_idxs)), color='magenta', marker='v', s=50)
    h = ax.imshow(
        probs.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=1, origin='lower', cmap='gist_gray',
    )

    cbar = plt.colorbar(h, label='Belief')
    cbar.set_ticks([0, 1])
    ax.legend([correct_lick_choice,wrong_lick_choice,attention_choice], ['correct_lick_choice','wrong_lick_choice','attention_choice'], bbox_to_anchor=bbox_to_anchor, fontsize=12)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks(1/num_states * np.arange(num_states) - 0.5 + 0.5 * 1/num_states)
    ax.set_yticklabels([f'{i}' for i in range(num_states)])
    
    ax.set_xlabel('Time')
    ax.set_ylabel('States')
    figs.append(fig)
    return figs