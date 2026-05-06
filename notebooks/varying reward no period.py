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

print("=== Starting: notebooks/varying reward no period.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AuditoryForagingReward2 as AFR2


# --- Cell 1 ---
modelname = 'varyreward67noperoid'
epoch_size = 33333
n_epoch = 55
n_seed = 1
seed = 0
no_episodes = 3333  # for eval plot during training
plot_no_episodes = 333
vmin, vmax = 10, 10000
food_reward_list = np.linspace(vmin, vmax, 9)
attcost = -1.6
facost = -50
p1p2 = np.array([.6, .7])
env_param = [0, None, attcost, .25, facost, 0, 0]
np.random.seed(seed)


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
model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
            clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs/{modelname}")

# train
for iepoch in range(0,n_epoch):
    thismodel = f'seed{seed}_{modelname}_{iepoch}'
    model.learn(total_timesteps=epoch_size,
                tb_log_name=thismodel,
                reset_num_timesteps=False,)
    model.save(f'ycstore/{thismodel}')
    print('epoch', iepoch)

    # eval
    def eval_wrapper(a):
        episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                num_steps=100000, deterministic=True)
        return episode

    with multiprocess.Pool(processes=8) as pool:
        all_episode_data = pool.map(eval_wrapper, range(no_episodes))


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # auto correlation 
    for food_reward_idx in range(len(food_reward_list)):
        att_seq_list = []
        for episode in all_episode_data:
            if episode['trial_food_reward_idx'] == food_reward_idx:
                att_seq_list.append([float(elt[1])
                                    for elt in episode['actions']])
        # Sorting based on increasing trial length
        sorted_list_of_lists = sorted(att_seq_list, key=len)
        # Choosing what range of trial lengths to use. Could modify to be more consistent.
        sequences = sorted_list_of_lists[-plot_no_episodes:]
        # autocorr
        autocorr, total_counts = empirical_autocorrelation(
            sequences, min_no_samples=1111)
        xs = np.arange(0, len(autocorr), 1)
        xs = xs-len(xs)//2
        ax1.plot(xs, autocorr,
                    label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
        # trial len dist
        sns.kdeplot([len(a) for a in att_seq_list],
                    fill=False, bw_adjust=0.5, ax=ax2)

        ax1.set_xlabel(r'$\tau$ [s]')
        ax1.set_title('auto correlation')
        # ax1.set_xlim(-50,50)
        # ax1.set_ylim(0, .03)
        ax1.set_xticks([-50, 0, 50])
        ax1.set_yticklabels([])
        centerax(ax1)
        quickleg(ax1)
        ax2.set_xlabel('trial length')
        ax2.set_ylabel('probablity')
        ax2.set_xlim(0, 80)
        ax2.set_title('trial length distribution')

    plt.tight_layout()
    quicksave('autocorr', modelname)
    pass

notify(f'training {thismodel} finished')


# --- Cell 2 ---
# eval plot
iepoch=24

modelname = 'varyreward67noperoid'
epoch_size = 33333
n_epoch = 55
n_seed = 1
seed = 0
no_episodes = 33333# for eval plot during training
plot_no_episodes = 33333
vmin, vmax = 10, 10000
food_reward_list = np.linspace(vmin, vmax, 5)
attcost = -1.6
facost = -50
p1p2 = np.array([.6, .7])
env_param = [0, None, attcost, .25, facost, 0, 0]
np.random.seed(seed)


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

thismodel = f'seed{seed}_{modelname}_{iepoch}'
model=PPO.load(f'ycstore/{thismodel}')
print('epoch', iepoch,f'model', thismodel)

# eval
def eval_wrapper(a):
    episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                            num_steps=100000, deterministic=True)
    return episode

with multiprocess.Pool(processes=8) as pool:
    all_episode_data = pool.map(eval_wrapper, range(no_episodes))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# auto correlation 
for food_reward_idx in range(len(food_reward_list)):
    att_seq_list = []
    for episode in all_episode_data:
        if episode['trial_food_reward_idx'] == food_reward_idx:
            att_seq_list.append([float(elt[1])
                                for elt in episode['actions']])
    # Sorting based on increasing trial length
    sorted_list_of_lists = sorted(att_seq_list, key=len)
    # Choosing what range of trial lengths to use. Could modify to be more consistent.
    sequences = sorted_list_of_lists[-plot_no_episodes:]
    # autocorr
    autocorr, total_counts = empirical_autocorrelation(
        sequences, min_no_samples=0)
    xs = np.arange(0, len(autocorr), 1)
    xs = xs-len(xs)//2
    ax1.plot(xs, autocorr,
                label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
    # trial len dist
    sns.kdeplot([len(a) for a in att_seq_list],
                fill=False, bw_adjust=0.5, ax=ax2)

    ax1.set_xlabel(r'$\tau$ [s]')
    ax1.set_title('auto correlation')
    ax1.set_xlim(-50,50)
    # ax1.set_ylim(0, .03)
    ax1.set_xticks([-50, 0, 50])
    ax1.set_yticklabels([])
    ax1.set_yscale('log')
    # centerax(ax1)
    ax1.spines['left'].set_position('zero')
    quickleg(ax1)
    ax2.set_xlabel('trial length')
    ax2.set_ylabel('probablity')
    ax2.set_xlim(0, 80)
    ax2.set_title('trial length distribution')

fig.suptitle('varying reward, non peroid')
plt.tight_layout()
quicksave('autocorr', modelname)
pass



print("=== Done: notebooks/varying reward no period.ipynb ===")
