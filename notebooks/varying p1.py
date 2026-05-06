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

print("=== Starting: notebooks/varying p1.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AuditoryForagingReward2 as AFR2


# --- Cell 1 ---
# modelname = 'varyp1_v0'
# p1list=[np.array([.5, .9]),np.array([.6, .9]),np.array([.7, .9]),np.array([.8, .9])]

modelname = 'varyp1_v3'
p1list=[np.array([.5, .7]),np.array([.501, .7]),np.array([.505, .7])]



no_episodes=3333
plot_no_episodes=3333
epoch_size = 33333
n_epoch = 22
n_seed = 1
modeli = 0
vmin, vmax = 10, 10000
food_reward_list = np.linspace(vmin, vmax, 5)


print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])


for attcost in [-1.6]:
    for facost in [-50,]:
        for p1p2 in p1list:
            env_param = [0, None, attcost, .25, facost, 0, 0]
            seed=0
            thismodel = f'seed_{seed}_{modelname}_{modeli}'
            print(thismodel)
            for seed in range(n_seed):
                # set seed
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

                    for episode in all_episode_data:
                        attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=all_reward_res[episode['trial_food_reward_idx']]
                        att_time_list.append(
                            np.where(np.array([elt[1] for elt in episode['actions']]) == 1)[0])
                        if episode['actions'][-1][0] == 1:
                            lick_time_list.append(len(episode['states']))
                        else:
                            lick_time_list.append(None)
                        trial_len.append(len(episode['actions']))

                    # auto correlation    
                    for food_reward_idx in range(len(food_reward_list)):
                        att_seq_list = []
                        for episode in all_episode_data:
                            if episode['trial_food_reward_idx']==food_reward_idx:
                                att_seq_list.append([float(elt[1]) for elt in episode['actions']])
                        # Sorting based on increasing trial length
                        sorted_list_of_lists = sorted(att_seq_list, key=len)
                        sequences = sorted_list_of_lists[-333:-1] # Choosing what range of trial lengths to use. Could modify to be more consistent.
                        normalized_sequences = []
                        max_length = 0
                        for sequence in sequences:
                            normalized_seq = np.array(sequence)
                            if not np.isnan(normalized_seq.any()):
                                normalized_sequences.append(normalized_seq)
                                max_length = max(max_length, len(normalized_seq))    
                            
                        padded_sequences = []
                        for seq in normalized_sequences:
                            padded_seq = np.pad(seq, (0, max_length - len(seq)), mode='constant')
                            padded_sequences.append(padded_seq)

                        autocorrelation_sum = np.zeros(2 * max_length - 1)
                        for seq in padded_sequences:
                            autocorrelation_sum += np.nan_to_num(compute_autocorrelation(seq), nan=0)

                        average_autocorrelation = autocorrelation_sum / len(padded_sequences)
                        xs=np.arange(0,len(average_autocorrelation),1)
                        xs=xs-len(xs)//2
                        plt.plot(xs,average_autocorrelation,label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
                        plt.xlabel('tau')
                        plt.ylabel('autocorrelation')
                        plt.title(f'p1p2={p1p2} epoch:{i}')
                    plt.xlim(-50,50)
                    quicksave('autocorr', modelname)
                    pass


                    # # heatmap trial indicator (att and lick)
                    # for food_reward_idx in (all_reward_res.keys()):
                    #     attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=all_reward_res[food_reward_idx]
                    #     sortind=np.argsort(trial_len[:plot_no_episodes])

                    #     grid = np.zeros((plot_no_episodes, max(trial_len)+2))

                    #     # att
                    #     plot_data =att_time_list[:plot_no_episodes]
                    #     for i, arr in enumerate(plot_data):
                    #         grid[i, arr] = -1

                    #     # lick
                    #     plot_data =lick_time_list[:plot_no_episodes]
                    #     for i, arr in enumerate(plot_data):
                    #         if arr:
                    #             grid[i, arr] = 1
                        
                    #     plt.figure()
                    #     c=plt.imshow(grid[sortind], cmap='bwr', vmin=-1, vmax=1,
                    #                     aspect='auto', interpolation='none')
                    #     # plt.colorbar(c)
                    #     plt.xlabel('time in trial')
                    #     plt.ylabel('episode no.')
                    #     plt.title(f'attn, food reward: {food_reward_list[food_reward_idx]}')

                    #     # legend
                    #     # Create Patch objects for the custom legend
                    #     red_patch = Patch(color='red', label='lick')
                    #     blue_patch = Patch(color='blue', label='attention')

                    #     # Create a new legend
                    #     extra_legend = plt.legend(handles=[red_patch, blue_patch], loc='upper right', frameon=True)
                    #     frame = extra_legend.get_frame()
                        # frame.set_color('lightgrey')

                        # # Add the new legend to the existing plot
                        # plt.gca().add_artist(extra_legend)
                        # quicksave(f'ep indicator {food_reward_idx}', modelname)
                        # pass

                notify(f'training {thismodel} finished')
            modeli += 1

# --- Cell 2 ---
# modelname = 'varyp1_v0'
# p1list=[np.array([.5, .9]),np.array([.6, .9]),np.array([.7, .9]),np.array([.8, .9])]

# modelname = 'varyp1_v3'
# p1list=[np.array([.5, .7]),np.array([.501, .7]),np.array([.505, .7])]

epochtouse=11

no_episodes=3333
plot_no_episodes=3333
epoch_size = 33333
n_epoch = 20
n_seed = 1


vmin, vmax = 10, 5000
food_reward_list = np.linspace(vmin, vmax, 5)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

seed=0
np.random.seed(seed)

attcost=-1.6
facost=-50
env_param = [0, None, attcost, .25, facost, 0, 0]


for i in range(epochtouse,epochtouse+1):
    for food_reward_idx in [2,3,4]: # one plot
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

        for modeli, p1p2 in enumerate(p1list):
            
            # load model (for each p1p2)
            thismodel = f'seed_{seed}_{modelname}_{modeli}'
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
            model = PPO.load(f'ycstore/{thismodel}_epoch_{i}')
            print(f'ycstore/{thismodel}_epoch_{i}')

            # collect data
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

            for episode in all_episode_data:
                attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=all_reward_res[episode['trial_food_reward_idx']]
                att_time_list.append(
                    np.where(np.array([elt[1] for elt in episode['actions']]) == 1)[0])
                if episode['actions'][-1][0] == 1:
                    lick_time_list.append(len(episode['states']))
                else:
                    lick_time_list.append(None)
                trial_len.append(len(episode['actions']))


            # auto correlation   (plot for each p1p2)
            att_seq_list = []
            for episode in all_episode_data:
                if np.all(episode['p1p2']==p1p2):
                    att_seq_list.append([float(elt[1]) for elt in episode['actions']])
            # Sorting based on increasing trial length
            sorted_list_of_lists = sorted(att_seq_list, key=len)
            sequences = sorted_list_of_lists[-333:-1] # Choosing what range of trial lengths to use. Could modify to be more consistent.
            normalized_sequences = []
            max_length = 0
            for sequence in sequences:
                normalized_seq = np.array(sequence)
                if not np.isnan(normalized_seq.any()):
                    normalized_sequences.append(normalized_seq)
                    max_length = max(max_length, len(normalized_seq))    
                
            padded_sequences = []
            for seq in normalized_sequences:
                padded_seq = np.pad(seq, (0, max_length - len(seq)), mode='constant')
                padded_sequences.append(padded_seq)

            autocorrelation_sum = np.zeros(2 * max_length - 1)
            for seq in padded_sequences:
                autocorrelation_sum += np.nan_to_num(compute_autocorrelation(seq), nan=0)


            average_autocorrelation = autocorrelation_sum / len(padded_sequences)
            xs=np.arange(0,len(average_autocorrelation),1)
            xs=xs-len(xs)//2
            ax1.plot(xs,average_autocorrelation,label=f'p1p2={p1p2}')
            ax1.set_xlabel('tau')
            ax1.set_ylabel('autocorrelation')
            ax1.set_xlim(-50,50)
            # ax1.set_title(f'food reward={food_reward_list[food_reward_idx]:.0f} epoch:{i}')
            ax1.set_title(f'food reward={food_reward_list[food_reward_idx]:.0f}, auto correlation')
            ax1.legend()

            # ax2.hist([len(a) for a in att_seq_list])
            sns.kdeplot([len(a) for a in att_seq_list], fill=False, bw_adjust=0.5)
            ax2.set_xlabel('trial length')
            ax2.set_ylabel('probablity')
            ax2.set_xlim(0,80)
            ax2.set_title('trial length distribution')
            # ax2.set_title(f'food reward={food_reward_list[food_reward_idx]:.0f} epoch:{i}')
        
        ax1.legend(bbox_to_anchor=(1,0))
        plt.tight_layout()
        quicksave('autocorr', modelname)
        pass





# --- Cell 3 ---

# try new autocorr

def empirical_autocorrelation(list_of_seqs, min_no_samples):
    # input is a list of lists, each inner list being a sequence
    '''
    list_of_seqs: list of lists
    min_no_samples: min 
    '''

    def first_index_above_value(array, value):
        indices = np.where(array >= value)
        if len(indices[0]) == 0:
            return None
        return indices[0][0]

    def pad_lists_with_zeros(lists):
        max_length = max(len(inner_list) for inner_list in lists) 
        padded_lists = []
        for inner_list in lists:
            current_length = len(inner_list)
            total_padding = max_length - current_length
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            padded_list = [0] * left_padding + inner_list + [0] * right_padding
            padded_lists.append(padded_list)
        return padded_lists

    def generate_autocorr_count(n):
        no_samples_one_direction = [n - i for i in range(n)]
        no_samples_both_directions = list(reversed(no_samples_one_direction[1:])) + no_samples_one_direction
        return no_samples_both_directions 

    def convert_to_float(list_of_lists):
        return [[float(elt) for elt in sublist] for sublist in list_of_lists]
    
    list_of_seqs = convert_to_float(list_of_seqs)
    list_of_counts = []
    list_of_unorm_corrs = []
    for seq in list_of_seqs:
        unorm_corr = np.correlate(seq, seq, mode='full')
        list_of_unorm_corrs.append(list(unorm_corr))
        list_of_counts.append(generate_autocorr_count(len(seq)))
    total_counts = np.sum(np.array(pad_lists_with_zeros(list_of_counts)), 0)
    autocorr = np.sum(np.array(pad_lists_with_zeros(list_of_unorm_corrs)), 0)/total_counts

    ind = first_index_above_value(total_counts, min_no_samples)
    if ind != 0:
        return autocorr[ind:-ind], total_counts[ind:-ind]
    
    return autocorr, total_counts

plt.plot(empirical_autocorrelation(att_time_list, 0)[1])
plt.plot(empirical_autocorrelation(att_time_list, 0)[0])

# --- Cell 4 ---

for episode in all_episode_data:
    attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=all_reward_res[episode['trial_food_reward_idx']]
    att_time_list.append(
        np.where(np.array([elt[1] for elt in episode['actions']]) == 1)[0])
    if episode['actions'][-1][0] == 1:
        lick_time_list.append(len(episode['states']))
    else:
        lick_time_list.append(None)
    trial_len.append(len(episode['actions']))


# auto correlation 
fig,ax1=plt.subplots(1,1)
att_seq_list = []
for episode in all_episode_data:
    if np.all(episode['p1p2']==p1p2):
        att_seq_list.append([float(elt[1]) for elt in episode['actions']])
        
# Sorting based on increasing trial length
sorted_list_of_lists = sorted(att_seq_list, key=len)
sequences = sorted_list_of_lists[-333:-1] # Choosing what range of trial lengths to use. Could modify to be more consistent.
normalized_sequences = []
max_length = 0
for sequence in sequences:
    normalized_seq = np.array(sequence)
    if not np.isnan(normalized_seq.any()):
        normalized_sequences.append(normalized_seq)
        max_length = max(max_length, len(normalized_seq))    
    
padded_sequences = []
for seq in normalized_sequences:
    padded_seq = np.pad(seq, (0, max_length - len(seq)), mode='constant')
    padded_sequences.append(padded_seq)

autocorrelation_sum = np.zeros(2 * max_length - 1)
for seq in padded_sequences:
    autocorrelation_sum += np.nan_to_num(compute_autocorrelation(seq), nan=0)


average_autocorrelation = autocorrelation_sum / len(padded_sequences)
xs=np.arange(0,len(average_autocorrelation),1)
xs=xs-len(xs)//2
ax1.plot(xs,average_autocorrelation,label=f'oringal')
ax1.plot(empirical_autocorrelation(att_time_list, 0)[1], label= 'autocorr')
ax1.plot(empirical_autocorrelation(att_time_list, 0)[0], label= 'count')
ax1.set_xlabel('tau')
ax1.set_ylabel('autocorrelation')
ax1.set_xlim(-50,50)
# ax1.set_title(f'food reward={food_reward_list[food_reward_idx]:.0f} epoch:{i}')
ax1.set_title(f'food reward={food_reward_list[food_reward_idx]:.0f}, auto correlation')
ax1.legend()
pass


# --- Cell 5 ---
att_time_list
# figure 7
# for food_reward_idx, v in all_reward_res.items():
#     attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=v
#     att_size_previous=[previous_block_size(a) for a in att_time_list]
#     a=np.concatenate(att_size_previous)
#     plt.hist(a, bins=np.linspace(2,15,15), alpha=0.3, density=True,label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
#     # sns.kdeplot(a, fill=False, bw_adjust=0.5)
# plt.ylim(0,1)
# plt.title('previous attention block size')
# plt.ylabel('attention counts, normalized to prob')
# plt.xlabel('previous attention block size')
# plt.legend()
# pass

for food_reward_idx, v in all_reward_res.items():
    if food_reward_idx in [0,1,2]:
        attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=v
        att_gap2previous=[previous_item_gap(a) for a in att_time_list]
        a=np.concatenate(att_gap2previous)
        plt.hist(a, bins=np.linspace(2,15,15), alpha=0.3, density=True,label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
        # sns.kdeplot(a, fill=False, bw_adjust=0.5)
# plt.ylim(0,1)
plt.title('gap to previous attention')
plt.ylabel('attention counts, normalized to prob')
plt.xlabel('time since last attention')
plt.legend()
pass

for food_reward_idx, v in all_reward_res.items():
    # if food_reward_idx in [3,4]:
        attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=v
        att_gap2previous=[previous_item_gap(a) for a in att_time_list]
        a=np.concatenate(att_gap2previous)
        a=a[a>0]
        # plt.hist(a, bins=np.linspace(2,15,15), alpha=0.3, density=False,label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
        sns.kdeplot(a, fill=False,clip=(0, np.max(a)), bw_adjust=3,label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
# plt.ylim(0,1)
plt.title('gap to previous attention')
plt.ylabel('attention counts, normalized to prob')
plt.xlabel('time since last attention')
plt.legend()
quicksave('gap full', modelname)
pass

for food_reward_idx, v in all_reward_res.items():
    if food_reward_idx in [0,2,4]:
        attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=v
        att_gap2previous=[previous_item_gap(a) for a in att_time_list]
        a=np.concatenate(att_gap2previous)
        a=a[a>0]
        # plt.hist(a, bins=np.linspace(2,15,15), alpha=0.3, density=False,label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
        sns.kdeplot(a, fill=False, clip=(0, np.max(a)),bw_adjust=3,label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
plt.xlim(0,15)
plt.title('gap to previous attention')
plt.ylabel('attention counts, normalized to prob')
plt.xlabel('time since last attention')
plt.legend()
quicksave('gap', modelname)
pass

# --- Cell 6 ---

# Initialize lists for each reward
reward_labels = []
reward_data = []

for food_reward_idx, v in all_reward_res.items():
    if food_reward_idx in [0, 1, 2]:
        attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len = v
        att_gap2previous = [previous_item_gap(a) for a in att_time_list]
        a = np.concatenate(att_gap2previous)
        
        # Count occurrences of values within bins
        counts, bins = np.histogram(a, bins=np.linspace(2, 25, 25))
        
        # Normalize counts
        counts_normalized = counts / np.sum(counts)
        
        reward_labels.append(f'food reward={food_reward_list[food_reward_idx]:.0f}')
        reward_data.append(counts_normalized)

# Create side-by-side bar chart
fig, ax = plt.subplots()
width = 0.2  # Width of each bar
x = np.arange(len(bins) - 1)

for i in range(len(reward_labels)):
    ax.bar(x + i * width, reward_data[i], width=width, label=reward_labels[i])

# Add labels and legend
ax.set_title('Gap to Previous Attention')
ax.set_ylabel('Attention Counts (normalized)')
ax.set_xlabel('Time Since Last Attention')

# Reduce number of xticks
num_xticks = 6
plt.xticks(np.arange(0, len(bins), step=int(len(bins) / num_xticks)), [f'{int(bin)}-{int(bin+1)}' for bin in bins[::int(len(bins) / num_xticks)]])
    
ax.legend()

pass


# --- Cell 7 ---
notify()

print("=== Done: notebooks/varying p1.ipynb ===")
