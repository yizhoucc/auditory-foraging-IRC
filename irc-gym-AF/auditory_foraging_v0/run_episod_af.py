import sys
sys.path.append('/home/lokesh/Documents/Projects/irc-gym-AF/irc-gym-main/')

import matplotlib.pyplot as plt
import numpy as np

from irc import BeliefAgentFamily
# from irc.examples import FoodBoxEnv # to change to your custom environment
# bafam = BeliefAgentFamily(FoodBoxEnv)
from AF_env import AuditoryForaging # to change to your custom environment # to change to your custom environment
bafam = BeliefAgentFamily(AuditoryForaging)

env_param = (0.05, 0.8, 10.)
seed = 0
agent, _ = bafam.optimal_agent(env_param, seed=seed)


print("Running imaginary episode using the internal model...")
episode = agent.run_one_episode()


def plot_single_box_episode(episode, num_shades=None, figsize=(10, 1.5)):
    num_steps = episode['num_steps']
    states = episode['states']
    obss = episode['obss']
    actions = episode['actions']
    rewards = episode['rewards']
    assert not np.any(episode['q_states']-np.array([(1,)]))
    probs = episode['q_probs']
    if num_shades is None:
        num_shades = obss.max()

    fig_w, fig_h = figsize
    aspect = num_steps*fig_h/fig_w*1.5
    figs = []

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        states.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
    )
    cbar = plt.colorbar(h, label='Has food')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['F', 'T'])
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks([])
    ax.set_xlabel('Time')
    figs.append(fig)

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        obss.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=num_shades, origin='lower', cmap='coolwarm',
    )
    cbar = plt.colorbar(h, label='Color cue')
    cbar.set_ticks([0, num_shades])
    idxs = (actions==1)&(rewards>0)
    h_true = ax.scatter(np.arange(num_steps)[idxs], np.zeros(sum(idxs)), color='magenta', marker='o', s=50)
    idxs = (actions==1)&(rewards<0)
    h_false = ax.scatter(np.arange(num_steps)[idxs], np.zeros(sum(idxs)), color='salmon', marker='x', s=50)
    ax.legend([h_true, h_false], ['Has food', 'No food'], loc='upper left', fontsize=12)
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks([])
    ax.set_xlabel('Time')
    figs.append(fig)

    fig, ax = plt.subplots(figsize=figsize)
    h = ax.imshow(
        probs.T, aspect=aspect, extent=[-0.5, num_steps+0.5, -0.5, 0.5],
        vmin=0, vmax=1, origin='lower', cmap='coolwarm',
    )
    cbar = plt.colorbar(h, label='Belief')
    cbar.set_ticks([0, 1])
    ax.set_xlim([-0.5, num_steps+0.5])
    ax.set_xticks([0, num_steps])
    ax.set_yticks([])
    ax.set_xlabel('Time')
    figs.append(fig)
    return figs


figs = plot_single_box_episode(episode)