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

print("=== Starting: notebooks/vary p1 v5.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AuditoryForagingReward2 as AFR2


# --- Cell 1 ---
# train all agent at the same time (not family)
modelname = 'vary p1 v5'
plist = [np.array([.5, .8]),np.array([.55, .8]),np.array([.6, .8]),np.array([.65, .8])]


epoch_size = 33333
n_epoch = 66
n_seed = 1
seed = 0
no_episodes = 6666  # for eval plot during training
plot_no_episodes = 555
vmin, vmax = 10, 10000
food_reward_list = np.linspace(vmin, vmax, 5)
attcost = -1.6
facost = -50
node=25
env_param = [0, None, attcost, .25, facost, 0, 0]
np.random.seed(seed)
minlen, maxlen=50,100

# --- Cell 2 ---
for iepoch in range(4, n_epoch):
    # training ----------------
    for ip, p1p2 in enumerate(plist):
        thismodel = f'seed{seed}_{modelname}_p{ip}_ep{iepoch}'
    
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
        task.obs_certainity_possible = p1p2

        # train
        if iepoch == 0:
            model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                        clip_range=0.1, ent_coef=0.01)
        else:
            previous_model = f'seed{seed}_{modelname}_p{ip}_ep{iepoch-1}'
            model = PPO.load(f'ycstore/{previous_model}', env=taskbelief)
        model.learn(total_timesteps=epoch_size,)
        model.save(f'ycstore/{thismodel}')
        print(f'epoch{iepoch}, p{p1p2}: yctore/{thismodel} saved')
        model = PPO.load(f'ycstore/{thismodel}', env=taskbelief) # test load


    # eval collection --------------------
    res={}
    for food_reward_idx in range(len(food_reward_list)):
        res[food_reward_idx]={}

    for ip, p1p2 in enumerate(plist):
        thismodel = f'seed{seed}_{modelname}_p{ip}_ep{iepoch}'
        task = AFR2(spec={'agent': 
                                            {'lick_cost': env_param[0],
                                            'food_reward': env_param[1],
                                            'attention_cost_coeff': env_param[2],
                                            'attention_cost_temp': env_param[3],
                                            'penalty_cost': env_param[4],
                                            'iti_cost': env_param[5],
                                            'time_in_game_reward': env_param[6]},
                                    'experiment':{                              'no_signal_nodes':node}})
        task.food_reward_list = food_reward_list
        taskbelief = FuncBeliefModel(env=task, rng=1)
        task.obs_certainity_possible = p1p2
        model = PPO.load(f'ycstore/{thismodel}', env=taskbelief)
        
        def eval_wrapper(a):
            episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                    num_steps=100000, deterministic=True)
            return episode

        with multiprocess.Pool(processes=8) as pool:
            all_episode_data = pool.map(eval_wrapper, range(no_episodes))
        
        for episode in all_episode_data:
            if ip in res[episode['trial_food_reward_idx']]:
                res[episode['trial_food_reward_idx']][ip].append(episode)
            else:
                res[episode['trial_food_reward_idx']][ip]=[episode]


    # eval ploting --------------------
    for food_reward_idx in range(len(food_reward_list)):     

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
        rasterfig, axs = plt.subplots(len(plist), 1, figsize=(
                3, len(plist)*2,), sharex=True, sharey=True)
        for ip, p1p2 in enumerate(plist):
            all_episode_data=res[food_reward_idx][ip]

            # att times -------------
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

            # raster fig
            attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len = all_reward_res[
                food_reward_idx]
            trial_len = np.array(trial_len)
            valid_ind = np.where((minlen < trial_len) & (trial_len < maxlen))[0]
            grid = np.zeros((len(valid_ind), max(trial_len)+2))
            print(food_reward_idx,ip, grid.shape)
            sortind = np.argsort([trial_len[i] for i in valid_ind])

            # att
            plot_data = [att_time_list[i] for i in valid_ind]
            for i, arr in enumerate(plot_data):
                grid[i, arr] = -1
            # lick
            plot_data = [lick_time_list[i] for i in valid_ind]
            for i, arr in enumerate(plot_data):
                if arr:
                    grid[i, arr] = 1

            colors = [attn_color, 'white', lick_color]
            stops = [0.0, 0.5, 1.0]
            attnlickcmap = LinearSegmentedColormap.from_list(
                'custom_cmap', list(zip(stops, colors)))

            c = axs[ip].imshow(grid, cmap=attnlickcmap, vmin=-1, vmax=1,
                                aspect='auto', interpolation='none')

            axs[ip].set_xticks([0, 50, 100])
            axs[ip].set_xlim(0, 100)
            axs[ip].set_ylim(0,300)
            axs[ip].set_yticks([])
            axs[ip].spines['left'].set_visible(False)

            # auto correlation -------------
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
            autocorr, total_counts = empirical_autocorrelation_new(
                sequences, min_no_samples=1111)
            xs = np.arange(0, len(autocorr), 1)
            xs = xs-len(xs)//2
            ax1.plot(xs, autocorr,

                        label=f'p:{p1p2}', color=cmap(np.linspace(0, 1, len(plist)))[ip])
            # trial len dist
            sns.kdeplot([len(a) for a in att_seq_list],
                        fill=False, bw_adjust=0.5, ax=ax2, color=cmap(np.linspace(0, 1, len(plist)))[ip])


        axs[ip].set_xlabel('time in trial')
        # lickdot = Patch(color=lick_color, label='lick')
        # attdot = Patch(color=attn_color, label='attention')
        # extra_legend = plt.legend(
        #     handles=[lickdot, attdot], bbox_to_anchor=(0, 1), frameon=True)
        # ax.add_artist(extra_legend)
        rasterfig.show()


        ax1.set_xlabel(r'$\tau$ [s]')
        ax1.set_title('auto correlation')
        ax1.set_xlim(-50, 50)
        ax1.set_ylim(None, 1)
        ax1.set_xticks([-50, 0, 50])
        # ax1.set_yscale('log')
        ax1.set_yticks([1])
        ax1.spines['left'].set_position('zero')
        ax1.spines['bottom'].set_position('zero')
        ax1.tick_params(axis='x', pad=50) 

        ax2.set_xlabel('trial length')
        ax2.set_ylabel('probablity')
        ax2.set_yticks([])
        ax2.set_xlim(0, 80)
        ax2.set_title('trial length distribution')

        color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax2)
        ticks = [0, 0.5, 1]
        tick_labels = [f'Low p={plist[0][0]}', '', f'High p={plist[-1][0]}']
        color_bar.set_ticks(ticks)
        color_bar.set_ticklabels(tick_labels)


        fig.suptitle(f'vary p1, p2={p1p2[1]} food \nreward={food_reward_list[food_reward_idx]:.0f}')
        fig.tight_layout()
        pass
    

print("=== Done: notebooks/vary p1 v5.ipynb ===")
