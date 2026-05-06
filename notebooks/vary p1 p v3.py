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

print("=== Starting: notebooks/vary p1 p v3.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))

# --- Cell 1 ---
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AuditoryForagingReward2 as AFR2
import warnings
warnings.filterwarnings("ignore")

# --- Cell 2 ---
start_rgb = (0.0, 1, 0.7) # low reward
end_rgb = (0.0, 0.2, 0.2) # high reward
cmap = LinearSegmentedColormap.from_list('custom_cmap', [start_rgb, end_rgb])

# --- Cell 3 ---
modelname = 'varyp1pv3'
plist = [np.array([.5, .7]),np.array([.51,.7]),np.array([.52, .7]),np.array([.53, .7])]

epoch_size = 33333
n_epoch = 22
n_seed = 1
seed = 0
no_episodes = 3333  # for eval plot during training
plot_no_episodes = 333
vmin, vmax = 10, 10000
food_reward_list = np.linspace(vmin, vmax, 5)
attcost = -1.6
facost = -50
env_param = [0, None, attcost, .25, facost, 0, 0]
np.random.seed(seed)

modeli=0
for iepoch in range(0,n_epoch):

    
    for p1p2 in plist:

        thismodel = f'seed{seed}_{modelname}_m{modeli}_ep{iepoch}'
        task = AFR2(spec={'agent': {'lick_cost': env_param[0],
                                    'food_reward': env_param[1],
                                    'attention_cost_coeff': env_param[2],
                                    'attention_cost_temp': env_param[3],
                                    'penalty_cost': env_param[4],
                                    'iti_cost': env_param[5],
                                    'time_in_game_reward': env_param[6]}})
        task.food_reward_list = food_reward_list
        taskbelief = FuncBeliefModel(env=task, rng=1)
        task.obs_certainity_possible = p1p2

        # train
        if iepoch == 0:
            model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs/{modelname}",)
        else:
            previous_model = f'seed{seed}_{modelname}_m{modeli}_ep{iepoch-1}'
            model = PPO.load(previous_model, env=taskbelief)
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=f'{p1p2}',
                    reset_num_timesteps=False,
                    )
        model.save(thismodel)
        print(f'epoch{iepoch},p {p1p2}: {thismodel} saved')

    if iepoch < 5:
        continue  # skip early plots

    use_reward = [0,1,2, 3, 4]
    for _,food_reward_idx in enumerate(use_reward):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        for i, p1p2 in enumerate(plist):
            thismodel = f'seed{seed}_{modelname}_m{modeli}_ep{iepoch}'
            task = AFR2(spec={'agent': {'lick_cost': env_param[0],
                                    'food_reward': env_param[1],
                                    'attention_cost_coeff': env_param[2],
                                    'attention_cost_temp': env_param[3],
                                    'penalty_cost': env_param[4],
                                    'iti_cost': env_param[5],
                                    'time_in_game_reward': env_param[6]}})
            task.food_reward_list = food_reward_list
            taskbelief = FuncBeliefModel(env=task, rng=1)
            task.obs_certainity_possible = p1p2
            model = PPO.load(thismodel)

            def eval_wrapper(a):
                episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                        num_steps=100000, deterministic=True)
                return episode

            with multiprocess.Pool(processes=8) as pool:
                all_episode_data = pool.map(eval_wrapper, range(no_episodes))

            all_reward_res = {}  # k: v = foodreward idx: this food reward res
            for r in range(len(food_reward_list)):
                attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len = [], [], [], [], [], []
                all_reward_res[r] = attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len

            for episode in all_episode_data:
                attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len = all_reward_res[
                    episode['trial_food_reward_idx']]
                att_time_list.append(
                    np.where(np.array([elt[1] for elt in episode['actions']]) == 1)[0])
                if episode['actions'][-1][0] == 1:
                    lick_time_list.append(len(episode['states']))
                else:
                    lick_time_list.append(None)
                trial_len.append(len(episode['actions']))

            att_seq_list = []
            for episode in all_episode_data:
                if episode['trial_food_reward_idx'] == food_reward_idx and np.all(episode['p1p2']== p1p2):
                    att_seq_list.append([float(elt[1])
                                        for elt in episode['actions']])
            # Sorting based on increasing trial length
            sorted_list_of_lists = sorted(att_seq_list, key=len)
            # Choosing what range of trial lengths to use. Could modify to be more consistent.
            sequences = sorted_list_of_lists[:]
            # autocorr
            autocorr, total_counts = empirical_autocorrelation(
                sequences, min_no_samples=0)
            xs = np.arange(0, len(autocorr), 1)
            xs = xs-len(xs)//2
            ax1.plot(xs, autocorr,label=f'p:{p1p2}', color=cmap(np.linspace(0, 1, len(plist)))[i])
            # trial len dist
            sns.kdeplot([len(a) for a in att_seq_list],
                        fill=False, bw_adjust=0.5, ax=ax2, color=cmap(np.linspace(0, 1, len(plist)))[i])

            ax1.set_xlabel(r'$\tau$ [s]')
            ax1.set_title('auto correlation (log)')
            ax1.set_xlim(-50, 50)
            ax1.set_ylim(0, None)
            ax1.set_xticks([-50, 0, 50])
            ax1.set_yscale('log')
            ax1.spines['left'].set_position('zero')
            ax1.set_yticks([])

            ax2.set_xlabel('trial length')
            ax2.set_ylabel('probablity')
            ax2.set_xlim(0, 80)
            ax2.set_xticks([0, 40, 80])
            ax2.set_yticks([])
            ax2.set_title('trial length distribution')

        color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax2)
        ticks = [0, 0.5, 1]
        tick_labels = [f'Low p1={plist[0][0]}', '', f'High p1={plist[-1][0]}']
        color_bar.set_ticks(ticks)
        color_bar.set_ticklabels(tick_labels)

        plt.suptitle(f'varying p1, p2={p1p2[1]}, food reward={food_reward_list[food_reward_idx]:.0f}')
        plt.tight_layout()
        quicksave('autocorr', modelname)
        pass


print("=== Done: notebooks/vary p1 p v3.ipynb ===")
