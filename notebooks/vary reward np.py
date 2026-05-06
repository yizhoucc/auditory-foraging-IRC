#!/usr/bin/env python3
"""Auto-generated training script from notebook. Run from repo root."""
import sys, os, warnings
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) != "" else os.getcwd()
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.append(os.path.join(REPO, "irc_gym"))
os.makedirs("ycstore", exist_ok=True)

import numpy as np
import torch
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import seaborn as sns
import pandas as pd
import pickle
import multiprocess
from collections import OrderedDict, defaultdict, Counter
from stable_baselines3 import PPO

from auditoryforage.AF_env import (
    AuditoryForaging, AuditoryForagingReward2, AuditoryForagingReward,
    AFnorate, AF2p, AF2pp, AuditoryForagingEnergy, AFVaryAttention
)
from irc_gym.irc.model import FuncBeliefModel
from ult import *

print("=== Starting: notebooks/vary reward np.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AuditoryForagingReward2 as AFR2


# --- Cell 1 ---
start_rgb = (0.0, 1, 0.7) # low reward
end_rgb = (0.0, 0.2, 0.2) # high reward
cmap = LinearSegmentedColormap.from_list('custom_cmap', [start_rgb, end_rgb])


# --- Cell 2 ---
modelname = 'varyreward npv2'
# training hyper params
epoch_size = 33333
node=10
n_epoch = 22
n_seed = 1
attcost=-1.6
facost=-50
p1p2 = np.array([.6, .7])
vmin, vmax = 10, 5000
food_reward_list = np.linspace(vmin, vmax, 5)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])
env_param = [0, None, attcost, .25, facost, 0, 0]
seed=0
thismodel = f'seed_{seed}_{modelname}'

for seed in range(n_seed):
    # set seed
    np.random.seed(seed)
    task = AFR2(spec={'agent':
                    {'lick_cost': env_param[0],
                    'food_reward': env_param[1],
                    'attention_cost_coeff': env_param[2],
                    'attention_cost_temp': env_param[3],
                    'penalty_cost': env_param[4],
                    'iti_cost': env_param[5],
                    'time_in_game_reward': env_param[6]},
                    'experiment': {'no_signal_nodes': node}})
    task.food_reward_list = food_reward_list
    taskbelief = FuncBeliefModel(env=task, rng=1)
    # set p1 p2
    task.obs_certainity_possible = p1p2
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs/{modelname}",)
    # train
    for i in range(n_epoch):
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=thismodel,
                    reset_num_timesteps=False,
                    )
        model.save(f'ycstore/{thismodel}_epoch_{i}')
        print(f'ycstore/{thismodel}_epoch_{i}')

        # eval
        # collect trial data
        no_episodes=3333
        def eval_wrapper(a):
            episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                    num_steps=100000, deterministic=True)
            return episode

        with multiprocess.Pool(processes=8) as pool:
            all_episode_data = pool.map(eval_wrapper, range(no_episodes))

        
        all_reward_res={} # k: v = foodreward idx: this food reward res
        for r in range(len(food_reward_list)):
            attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=[],[],[],[],[],[]
            all_reward_res[r]=attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len


        for episode in all_episode_data[:]:
            attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=all_reward_res[episode['trial_food_reward_idx']]
            att_time_list.append(
                np.where(np.array([elt[1] for elt in episode['actions']]) == 1)[0])
        
            if episode['actions'][-1][0] == 1:
                lick_time_list.append(len(episode['states']))
            else:
                lick_time_list.append(None)
            trial_len.append(len(episode['actions']))


        # side by side bar, seperated
        use_reward=np.arange(len(food_reward_list))
        reward_labels = []
        reward_data = []

        for food_reward_idx, v in all_reward_res.items():
            if food_reward_idx in use_reward:
                attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len = v
                att_gap2previous = [previous_item_gap(a) for a in att_time_list]
                a = np.concatenate(att_gap2previous)

                # Count occurrences of values within bins
                counts, bins = np.histogram(a, bins=np.linspace(2, 25, 25))

                # Normalize counts
                counts_normalized = counts / np.sum(counts)

                reward_labels.append(
                    f'food reward={food_reward_list[food_reward_idx]:.0f}')
                reward_data.append(counts_normalized)

        # Create side-by-side bar chart
        fig, axs = plt.subplots(len(use_reward), 1, figsize=(4, 4), sharex=True)
        width = 1  # Width of each bar
        x = np.arange(len(bins) - 1)

        for i in range(len(reward_labels)):
            axs[i].bar(x + i * width, reward_data[i],
                    width=width, label=reward_labels[i], color=cmap(np.linspace(0, 1, len(use_reward)))[i])
            axs[i].spines['left'].set_visible(False)
            axs[i].set_yticks([])

        axs[i].set_xlim(0, None)
        axs[i].set_xticks([0, 10, 20])
        axs[0].set_title('Gap to Previous Attention')
        # axs[i].set_ylabel('Attention Counts (normalized)')
        plt.suptitle('Attention Counts (normalized)', rotation='vertical', x=-0.1, y=0.5, va='center'                 )

        axs[i].set_xlabel('Time Since Last Attention')
        pass


        # autocorr
        from ult import empirical_autocorrelation_new
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        # use_reward = [2, 3, 4]
        # auto correlation
        for i, food_reward_idx in enumerate(use_reward):
            att_seq_list = []
            for episode in all_episode_data:
                if episode['trial_food_reward_idx'] == food_reward_idx:
                    att_seq_list.append([float(elt[1])
                                        for elt in episode['actions']])
            # Sorting based on increasing trial length
            sorted_list_of_lists = sorted(att_seq_list, key=len)
            # Choosing what range of trial lengths to use. Could modify to be more consistent.
            sequences = sorted_list_of_lists[-555:]
            # autocorr
            autocorr, total_counts = empirical_autocorrelation_new(
                sequences, min_no_samples=0)
            xs = np.arange(0, len(autocorr), 1)
            xs = xs-len(xs)//2
            ax1.plot(xs, autocorr,
                    label=f'food reward={food_reward_list[food_reward_idx]:.0f}', color=cmap(np.linspace(0, 1, len(use_reward)))[i])
            # trial len dist
            sns.kdeplot([len(a) for a in att_seq_list],
                        fill=False, bw_adjust=0.5, ax=ax2, color=cmap(np.linspace(0, 1, len(use_reward)))[i])

            ax1.set_xlabel(r'$\tau$ [s]')
            ax1.set_title('auto correlation')
            ax1.set_xlim(-50, 50)
            ax1.set_ylim(-0.5, 1)
            ax1.set_xticks([-50, 0, 50])
            # ax1.set_yscale('log')
            ax1.set_yticks([1])
            ax1.spines['left'].set_position('zero')
            ax1.spines['bottom'].set_position('zero')
            ax1.tick_params(axis='x', pad=50) 

            ax2.set_xlabel('trial length')
            ax2.set_ylabel('probablity')
            ax2.set_xlim(0, 80)
            ax2.set_xticks([0, 40, 80])
            ax2.set_yticks([])
            ax2.set_title('trial length distribution')

        color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax2)
        ticks = [0, 0.5, 1]
        tick_labels = ['Low food reward', '', 'High food reward']
        color_bar.set_ticks(ticks)
        color_bar.set_ticklabels(tick_labels)


        fig.suptitle('varying reward, non peroid')
        plt.tight_layout()
        # quicksave('autocorr', modelname)
        pass


    notify(f'training {thismodel} finished')


print("=== Done: notebooks/vary reward np.ipynb ===")
