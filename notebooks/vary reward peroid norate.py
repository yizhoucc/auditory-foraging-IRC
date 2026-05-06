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

print("=== Starting: notebooks/vary reward peroid norate.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AuditoryForagingReward2 as AFR2,AFnorate


# --- Cell 1 ---
start_rgb = (0.0, 1, 0.7) # low reward
end_rgb = (0.0, 0.2, 0.2) # high reward
cmap = LinearSegmentedColormap.from_list('custom_cmap', [start_rgb, end_rgb])


# --- Cell 2 ---
modelname = 'vary reward norate v3'
# training hyper params
epoch_size = 33333

n_epoch = 22
n_seed = 1
attcost=-1.6
facost=-50
p1p2 = np.array([.5, .7])
vmin, vmax = 10, 100
food_reward_list = np.linspace(vmin, vmax, 5)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])
env_param = [0, None, attcost, .25, facost, 0, 0]
seed=0


for seed in range(n_seed):
    # set seed
    np.random.seed(seed)
    task = AFnorate(spec={'agent': {'lick_cost': env_param[0],
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

    # train
    for iepoch in range(n_epoch):
        thismodel = f'seed{seed}_{modelname}_ep{iepoch}'
        if iepoch == 0:
            model =  PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs/{modelname}",)
        else:
            previous_model =f'seed{seed}_{modelname}_ep{iepoch-1}'
            model = PPO.load(f'ycstore/{previous_model}', env=taskbelief)

        model.learn(total_timesteps=epoch_size,
                    tb_log_name=thismodel,
                    reset_num_timesteps=False,
                    )
        model.save(f'ycstore/{thismodel}')
        print(f'epoch{iepoch}: yctore/{thismodel} saved')

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

        # process trial data
        def process_one_episode_wrapper(episode):
            return process_one_episode(episode, task)
        with multiprocess.Pool(processes=8) as pool:
            results = pool.map(process_one_episode_wrapper, all_episode_data)

        # process the mp results
        hit_count_list = [[0] for _ in range(len(food_reward_list))]
        miss_count_list = [[0] for _ in range(len(food_reward_list))]
        false_alarm_list = [[0] for _ in range(len(food_reward_list))]

        noise_time_before_lick_list = [[0] for _ in range(len(food_reward_list))]
        signal_time_before_lick_list = [[0] for _ in range(len(food_reward_list))]
        total_noise_time_list = [[0] for _ in range(len(food_reward_list))]
        total_signal_time_list = [[0] for _ in range(len(food_reward_list))]

        # dist naming space: list of list of num. big list of each reward cond. small list of each trial's summary stats, which is a single number.
        total_reward_dist= [[] for _ in range(len(food_reward_list))]
        rt_hit_dist=[[] for _ in range(len(food_reward_list))]
        rt_fa_dist=[[] for _ in range(len(food_reward_list))]
        total_att_time_dist = [[] for _ in range(len(food_reward_list))]
        for res in results:
            (food_reward_idx,
            hit_count,
            miss_count,
            false_alarm_count,

            noise_time_before_lick,
            signal_time_before_lick,
            total_noise_time,
            total_signal_time,

            total_reward,

            attention_time_points_across_episodes,
            episode_length_across_episodes,
            hit_reaction_time_across_episodes,
            fa_reaction_time_across_episodes
            ) = res

            hit_count_list[food_reward_idx][0] += (hit_count)
            miss_count_list[food_reward_idx][0] += (miss_count)
            false_alarm_list[food_reward_idx][0] += (false_alarm_count)

            noise_time_before_lick_list[food_reward_idx][0] += (noise_time_before_lick)
            signal_time_before_lick_list[food_reward_idx][0] += (
                signal_time_before_lick)
            total_noise_time_list[food_reward_idx][0] += (total_noise_time)
            total_signal_time_list[food_reward_idx][0] += (total_signal_time)

            total_reward_dist[food_reward_idx].append(total_reward)
            rt_hit_dist[food_reward_idx].append(signal_time_before_lick)
            rt_fa_dist[food_reward_idx].append(noise_time_before_lick)
            total_att_time_dist[food_reward_idx].append(len(attention_time_points_across_episodes[0]))

        # side by side bar, seperated
        use_reward=[2,3,4]
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        use_reward = [2, 3, 4]
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
            sequences = sorted_list_of_lists[:]
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
            # ax1.set_ylim(None, 1)
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
        quicksave('autocorr', modelname)
        pass

        minlen, maxlen = 50, 100
        use_reward = list(range(len(food_reward_list)))
        fig, axs = plt.subplots(len(use_reward), 1, figsize=(
            3, len(use_reward)*2,), sharex=True, sharey=True)

        for axi, food_reward_idx in enumerate(use_reward):
            attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len = all_reward_res[
                food_reward_idx]
            trial_len = np.array(trial_len)
            valid_ind = np.where((minlen < trial_len) & (trial_len < maxlen))[0]
            grid = np.zeros((len(valid_ind), max(trial_len)+2))
            sortind = np.argsort([trial_len[i] for i in valid_ind])
            # valid_ind = valid_ind[sortind]

            # att
            plot_data = [att_time_list[i] for i in valid_ind]
            for i, arr in enumerate(plot_data):
                grid[i, arr] = -1

            # lick
            plot_data = [lick_time_list[i] for i in valid_ind]
            for i, arr in enumerate(plot_data):
                if arr:
                    grid[i, arr] = 1
            if np.sum(grid) == 0:
                print(food_reward_idx, 'no attn')
                continue

            colors = [attn_color, 'white', lick_color]
            stops = [0.0, 0.5, 1.0]
            attnlickcmap = LinearSegmentedColormap.from_list(
                'custom_cmap', list(zip(stops, colors)))

            c = axs[axi].imshow(grid, cmap=attnlickcmap, vmin=-1, vmax=1,
                                aspect='auto', interpolation='none')

            plt.xticks([0, 50, 100])
            plt.xlim(0, 100)
            plt.ylim(0,300)
            plt.yticks([])
            ax = axs[axi]
            ax.spines['left'].set_visible(False)

        # reaction time
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        data = []
        for food_reward_idx, v in enumerate(rt_hit_dist):
            v = np.array(v)
            v = v[v > 1]  # ignore the 0 reaction time in the plots
            if len(v) == 0:
                continue
            data.append((food_reward_idx, v))


        for i, (food_reward_idx, curve) in enumerate(data):
            sns.kdeplot(curve, label=f'reward:{food_reward_list[food_reward_idx]:.0f}', clip=(1, np.max(v)), bw_adjust=1,color=cmap(np.linspace(0, 1, len(data)))[i])

            # plt.plot(curve, color=cmap(np.linspace(0, 1, len(data)))[i])
        color_bar=plt.colorbar(plt.cm.ScalarMappable(cmap=cmap),ax=ax)
        ticks = [0, 0.5, 1]
        tick_labels = ['Low food reward', '', 'High food reward']
        color_bar.set_ticks(ticks)
        color_bar.set_ticklabels(tick_labels)

        ax.set_ylabel('probability')
        ax.set_xlabel('reaction time')
        centerax(ax)
        ax.set_xticks([0, 10, 20])
        ax.set_yticklabels([])
        # quickleg(ax, bbox_to_anchor=(-0.5, 0))
        ax.set_title('vary reward \n reaction time distribution')


    notify(f'training {thismodel} finished')


print("=== Done: notebooks/vary reward peroid norate.ipynb ===")
