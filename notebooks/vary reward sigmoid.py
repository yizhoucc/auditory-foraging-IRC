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

print("=== Starting: notebooks/vary reward sigmoid.ipynb ===")

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
modelname = 'varyrewardsig'
# training hyper params
epoch_size = 33333
n_epoch = 22
n_seed = 1
modeli = 0
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
        no_episodes=3333
        def eval_wrapper(a):
            episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                    num_steps=100000, deterministic=True)
            return episode

        with multiprocess.Pool(processes=8) as pool:
            all_episode_data = pool.map(eval_wrapper, range(no_episodes))

        # process trial data
        def process_one_episode_wrapper(episode):
            return process_one_episode(episode, task)
        with multiprocess.Pool(processes=8) as pool:
            results = pool.map(process_one_episode_wrapper, all_episode_data)

        all_reward_res = {}  # k: v = foodreward idx: this food reward res
        for r in range(len(food_reward_list)):
            attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len, attention_entropy_list, lick_entropy_list = [], [], [], [], [], [], [], []
            all_reward_res[r] = attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len, attention_entropy_list, lick_entropy_list


        for episode in all_episode_data:
            attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len, attention_entropy_list, lick_entropy_list = all_reward_res[
                episode['trial_food_reward_idx']]
            att_time_list.append(
                np.where(np.array([elt[1] for elt in episode['actions']]) == 1)[0])

            if episode['actions'][-1][0] == 1:
                lick_time_list.append(len(episode['states']))
            else:
                lick_time_list.append(None)
            trial_len.append(len(episode['actions']))

            attention_prob, lick_prob, noise_prob, time = get_att_lick_probs_new(
                episode, model)
            attention_prob_list.append(attention_prob)
            lick_prob_list.append(lick_prob)
            noise_prob_list.append(noise_prob)
            attention_entropy_list.append(compute_bernoulli_entropy(attention_prob))
            lick_entropy_list.append(compute_bernoulli_entropy(lick_prob))
            

        for title in [ 'att prob Vs noise prob Vs time', 'lick prob Vs noise prob Vs time', 'att entropy Vs noise prob Vs time','lick entropy Vs noise prob Vs time']:


            plt.figure()
            for food_reward_idx in range(len(food_reward_list)):
                attention_prob_list, lick_prob_list, noise_prob_list, lick_time_list, att_time_list, trial_len, attention_entropy_list, lick_entropy_list = all_reward_res[
                    food_reward_idx]

                if title == 'att prob Vs noise prob Vs time':
                    data = (attention_prob_list, noise_prob_list)
                elif title == 'lick prob Vs noise prob Vs time':
                    data = (lick_prob_list, noise_prob_list)
                elif title == 'att entropy Vs noise prob Vs time':
                    data = (attention_entropy_list, noise_prob_list)
                elif title == 'lick entropy Vs noise prob Vs time':
                    data = (lick_entropy_list, noise_prob_list)
                else:
                    print('Error: Change title!')

                if title in ['att prob Vs noise prob Vs time', 'lick prob Vs noise prob Vs time', 'att entropy Vs noise prob Vs time', 'lick entropy Vs noise prob Vs time']:
                    no_ticks = 3
                    no_prob_bins = 10
                    min_display_time = 0
                    max_display_time = 50
                    prob_bin_vals = np.linspace(0,1,11)
                    display_tick_inds = np.linspace(
                        0, len(prob_bin_vals) - 1, no_ticks).astype(int)

                    (metric_list, noise_prob_list) = data
                    max_time = max([len(metric_val) for metric_val in metric_list])

                    prob_dict = {}
                    for episode_ind in range(len(noise_prob_list)):
                        for time_ind in range(len(noise_prob_list[episode_ind])):
                            prob_dict.setdefault((int(np.floor(noise_prob_list[episode_ind][time_ind] * no_prob_bins)), time_ind), [
                            ]).append(metric_list[episode_ind][time_ind])

                    avg_metric_matrix = np.full((no_prob_bins + 1, max_time), np.nan)
                    sem_metric_matrix = np.full((no_prob_bins + 1, max_time), np.nan)
                    count_matrix = np.full((no_prob_bins + 1, max_time), 0.0)
                    for (noise_prob_bin, time_ind), metric_list in prob_dict.items():
                        avg_metric_matrix[noise_prob_bin, time_ind] = np.mean(metric_list)
                        sem_metric_matrix[noise_prob_bin, time_ind] = np.std(
                            metric_list)/np.sqrt(len(metric_list))
                        count_matrix[noise_prob_bin, time_ind] = len(metric_list)

                    mean_list = np.nanmean(avg_metric_matrix, 1)
                    sample_size_list = np.sum(count_matrix, 1)
                    sem_list = np.nanstd(avg_metric_matrix, 1)/np.sqrt(sample_size_list)

                    # plt.errorbar([ind for ind in range(len(mean_list))], mean_list[::-1], yerr= sem_list[::-1], fmt='o', markersize=6, capsize=4)
                    y=mean_list[::-1]
                    valid_indices = np.where(~np.isnan(y))[0]
                    plt.plot(y[valid_indices], 'o-',
                            label=f'food reward={food_reward_list[food_reward_idx]:.0f}', color=cmap(np.linspace(0, 1, len(food_reward_list)))[food_reward_idx], linewidth=3)
                    ax=plt.gca()
                    centerax(ax)
                    plt.yticks([0,0.5,1], ['0','','1'])
                    plt.xlabel('signal prob.')
                    if title == 'att prob Vs noise prob Vs time':
                        plt.ylabel('Average att prob')
                    elif title == 'lick prob Vs noise prob Vs time':
                        plt.ylabel('Average lick prob')
                    plt.xticks(display_tick_inds, [str(prob_bin_vals[ind])[
                            :4] for ind in display_tick_inds])
                    plt.xlabel('signal prob.')
            color_bar=plt.colorbar(plt.cm.ScalarMappable(cmap=cmap),ax=ax)
            ticks = [0, 0.5, 1]
            tick_labels = ['Low food reward', '', 'High food reward']
            color_bar.set_ticks(ticks)
            color_bar.set_ticklabels(tick_labels)


            plt.title(title)
            # quickleg(ax)
            pass



    notify(f'training {thismodel} finished')
modeli += 1

print("=== Done: notebooks/vary reward sigmoid.ipynb ===")
