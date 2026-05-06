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

print("=== Starting: notebooks/vary reward gamma=98.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AFnorate as AFtask


# --- Cell 1 ---
modelname = 'vary_reward_gamma98_V1'

epoch_size = 33333
n_epoch = 15
n_seed = 1
seed = 0
np.random.seed(seed)

# vary conditions
vmin, vmax = 50, 150
food_reward_list = np.linspace(vmin, vmax, 5)
plist = [np.array([.5, .7])]
node_list = [25]
# fixed params
attcost = -1.6
facost = -50
env_param = [0, None, attcost, .25, facost, 0, 0]

# for plotting
no_episodes = 2222  # for eval plot during training
plot_no_episodes = 555 # for raster plot
minlen, maxlen=0,np.inf
nbin=15
bins=np.linspace(0,1,nbin)


# --- Cell 2 ---
# color bars
num_discrete_var = len(food_reward_list)
discrete_colors = rewardcmap(np.linspace(0, 1, num_discrete_var))
discrete_cmap = ListedColormap(discrete_colors)
boundaries = np.arange(num_discrete_var + 1)
norm = BoundaryNorm(boundaries, num_discrete_var)
fig, ax = plt.subplots(figsize=(8, 1))
fig.subplots_adjust(bottom=0.5)
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=discrete_cmap),
    cax=ax,
    orientation='horizontal',
    ticks=np.arange(num_discrete_var) + 0.5
)
cbar.set_label('Food reward')
cbar.set_ticks(np.arange(num_discrete_var) + 0.5)
cbar.set_ticklabels([f'{int(a)}' for a in food_reward_list])
pass


# --- Cell 3 ---
# helper and plot function
def process_one_subdf_df(subdf,no_signal_nodes=25, no_penalty_nodes=1):
    '''process the subdf data into result lists. the input is a df'''
    hit_count = 0
    miss_count = 0
    false_alarm_count = 0
    noise_time_before_lick = 0
    signal_time_before_lick = 0
    total_noise_time = 0
    total_signal_time = 0
    total_reward = 0

    attention_time_points_across_subdfs = []
    subdf_length_across_subdfs = []
    hit_reaction_time_across_subdfs = []
    fa_reaction_time_across_subdfs = []

    # process single ep data
    if subdf['actions'][-1][0] == 1:
        if subdf['states'][-1][0] >  no_signal_nodes +  no_penalty_nodes:
            hit_count += 1
            signal_time_before_lick_curr_subdf = count_no_elements(
                subdf['states'], 1,  no_signal_nodes)
            hit_reaction_time_across_subdfs.append(
                signal_time_before_lick_curr_subdf)
            signal_time_before_lick += signal_time_before_lick_curr_subdf
        else:
            false_alarm_count += 1
            noise_time_before_lick_curr_subdf = count_no_elements(
                subdf['states'], 0, 0)
            fa_reaction_time_across_subdfs.append(
                noise_time_before_lick_curr_subdf)
            noise_time_before_lick += noise_time_before_lick_curr_subdf
    else:
        miss_count += 1
    total_signal_time += count_no_elements(
        subdf['states'], 1,  no_signal_nodes)
    total_noise_time += count_no_elements(subdf['states'], 0, 0)
    total_reward += sum(subdf['rewards'])
    # attention_time_points_across_subdfs.append(np.where(subdf['actions'] % env.no_attention_modes > 0)[0])
    attention_time_points_across_subdfs.append(
        np.where(np.array([elt[1] for elt in subdf['actions']]) == 1)[0])
    subdf_length_across_subdfs.append(len(subdf['states']))
    # end process single ep data
    food_reward_idx = subdf['trial_food_reward_idx']
    return (food_reward_idx,
            hit_count,
            miss_count,
            false_alarm_count,
            noise_time_before_lick,
            signal_time_before_lick,
            total_noise_time,
            total_signal_time,
            total_reward,
            attention_time_points_across_subdfs,
            subdf_length_across_subdfs,
            hit_reaction_time_across_subdfs,
            fa_reaction_time_across_subdfs
            )

def hit_reaction_plot(df, savemodelname='test'):
    ''' '''
    # process data
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

    for i in range(len(df)):
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
        ) = process_one_subdf_df(df.iloc[i])

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


    # 3 curve plot (figure 4a)
    hit_count_list, miss_count_list, false_alarm_list = np.array(
        hit_count_list), np.array(miss_count_list), np.array(false_alarm_list)
    hit_prob = [hit_count_list[i]/(hit_count_list[i]+miss_count_list[i] +
                                    false_alarm_list[i]) for i in range(len(hit_count_list))]
    miss_prob = [miss_count_list[i]/(hit_count_list[i]+miss_count_list[i] +
                                        false_alarm_list[i]) for i in range(len(hit_count_list))]
    fa_prob = [false_alarm_list[i]/(hit_count_list[i]+miss_count_list[i] +
                                    false_alarm_list[i]) for i in range(len(hit_count_list))]

    xs=np.arange(len(food_reward_list))
    fig, ax=plt.subplots(1,1, figsize=(5,4))

    ax.plot(xs, hit_prob, '-*g', label='hit')
    ax.plot(xs, miss_prob, '-*b', label='miss')
    ax.plot(xs, fa_prob, '-*r', label='fa')
    quickleg(ax, bbox_to_anchor=(-0.5,0))
    ax.set_xlabel('food reward')
    ax.set_ylabel('probability')
    # ax.set_xticks(xs, [f'{a:.0f}' for a in food_reward_list]) # actual values
    ax.set_xticks([xs[i] for i in [0,-1]], ['low', 'high']) # low high
    ax.set_xticks(xs)
    ax.set_yticks([0,0.5,1], ['0', '.5','1'])
    ax.set_xticks(xs)
    centerax(ax)
    ax.set_title('hit, miss, false alarm')
    quicksave('a', savemodelname)
    pass

    # reaction time (figure 4b)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    data = []
    for food_reward_idx, v in enumerate(rt_hit_dist):
        v = np.array(v)
        v = v[v > 1]  # ignore the 0 reaction time in the plots
        if len(v) == 0:
            continue
        data.append((food_reward_idx, v))


    for i, (food_reward_idx, curve) in enumerate(data):
        if len(data)>1:
            # print(curve, data)
            sns.kdeplot(curve, label=f'reward:{food_reward_list[food_reward_idx]:.0f}', clip=(1, np.max(curve)), bw_adjust=1,color=rewardcmap(np.linspace(0, 1, len(data)))[i])

    ax.set_ylabel('probability')
    ax.set_xlabel('reaction time')
    centerax(ax)
    ax.set_xticks([0, 10, 20],[0, 10, 20])
    ax.set_yticks([])
    ax.set_yticklabels([])
    # quickleg(ax, bbox_to_anchor=(-0.5, 0))
    ax.set_title('vary reward \n reaction time distribution')
    quicksave('b', savemodelname)
    pass


    # # total attention time distribution, figure 4C
    # fig= plt.figure()
    # for food_reward_idx in np.arange(len(food_reward_list)):
        
    #     subdf=df[(df.trial_food_reward_idx==food_reward_idx)]
    #     sns.kdeplot([len(a) for a in subdf.at_times],  label=f'food reward={food_reward_list[food_reward_idx]:.0f}', bw_adjust=2, common_norm=False, color=rewardcmap(
    #         np.linspace(0, 1, len(food_reward_list)))[food_reward_idx])

    #     plt.xlim(0, 15)
    #     plt.xticks([0, 15], [0,15])
    #     plt.xlabel('# high attentions')
    #     plt.ylabel('prob')
    #     plt.title(f'total attention times, varying food reward')
    #     centerax(plt.gca())
    #     plt.yticks([])
    # pass



    # autocorelation for peroid. fig4g
    fig, ax1 = plt.subplots(1,1, figsize=(5,4))
    for food_reward_idx in np.arange(len(food_reward_list)):
        subdf=df[df.trial_food_reward_idx==food_reward_idx]
        att_seq_list = []
    
        for episode_actions in subdf.actions:
            att_seq_list.append([float(a[1]) for a in episode_actions])
        
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
                    label=f'node:{1}', color=rewardcmap(np.linspace(0, 1, len(food_reward_list)))[food_reward_idx])
    

        # ax1.set_xlabel(r'$\tau$ [s]')
        # # ax1.set_title('auto correlation')
        ax1.set_xlim(-50, 50)
        # ax1.set_ylim(None, 1)
        ax1.set_xticks([-50, 0, 50])
        # ax1.set_yscale('log')
        ax1.set_yticks([1])
        ax1.spines['left'].set_position('zero')
        ax1.spines['bottom'].set_position('zero')
        ax1.tick_params(axis='x', pad=50) 

    
    fig.suptitle(f'vary reward, peroidic')
    plt.tight_layout()
    quicksave('g', savemodelname)
    pass


# --- Cell 4 ---
for iepoch in range(0, n_epoch):
    # training ----------------
    for node in node_list:
        for p1p2 in plist:
            thismodel = f'seed{seed}_{modelname}_n{node}_p{p1p2}_ep{iepoch}'
        
            task = AFtask(spec={'agent':
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
                            clip_range=0.1, ent_coef=0.01, gamma=0.98)
            else:
                previous_model = f'seed{seed}_{modelname}_n{node}_p{p1p2}_ep{iepoch-1}'
                model = PPO.load(f'ycstore/{previous_model}', env=taskbelief,gamma=0.98)
            model.learn(total_timesteps=epoch_size,)
            model.save(f'ycstore/{thismodel}')
            print(f'epoch{iepoch}, node{node}: yctore/{thismodel} saved')
            model = PPO.load(f'ycstore/{thismodel}', env=taskbelief, gamma=0.98) # test load

    # eval collection --------------------
    results=[]
    for inode, node in enumerate(node_list):
        for ip, p1p2 in enumerate(plist):
            task = AFtask(spec={'agent': 
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
            thismodel = f'seed{seed}_{modelname}_n{node}_p{p1p2}_ep{iepoch}'
            model = PPO.load(f'ycstore/{thismodel}',gamma=0.98)

            def eval_wrapper(a):
                episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                        num_steps=100000, deterministic=True)
                return episode

            with multiprocess.Pool(processes=8) as pool:
                all_episode_data = pool.map(eval_wrapper, range(no_episodes))
            
            for episode in all_episode_data:
                # process
                episode['inode'] = inode
                episode['ip'] = ip
                episode['at_times'] = np.where(
                    np.array([elt[1] for elt in episode['actions']]) == 1)[0].astype('int')
                episode['at_seq'] = [float(elt[1])
                                    for elt in episode['actions']]
                if episode['actions'][-1][0] == 1:
                    episode['licktime'] = int(len(episode['states']))
                else:
                    episode['lick_licktimetime'] = -1
                episode['trial_len'] = len(episode['actions'])

                results.append(episode)

    df = pd.DataFrame(results)
    print(f'total reward per trial: {sum(np.concatenate(df.rewards.to_numpy()))/len(df)}')

    hit_reaction_plot(df,'gamma98')
    



# --- Cell 5 ---
for iepoch in range(31,32):
   
    # eval collection --------------------
    results=[]
    for inode, node in enumerate(node_list):
        for ip, p1p2 in enumerate(plist):
            task = AFtask(spec={'agent': 
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
            thismodel = f'seed{seed}_{modelname}_n{node}_p{p1p2}_ep{iepoch}'
            model = PPO.load(f'ycstore/{thismodel}',gamma=1)

            def eval_wrapper(a):
                episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                        num_steps=100000, deterministic=True)
                return episode

            with multiprocess.Pool(processes=8) as pool:
                all_episode_data = pool.map(eval_wrapper, range(no_episodes))
            
            for episode in all_episode_data:
                # process
                episode['inode'] = inode
                episode['ip'] = ip
                episode['at_times'] = np.where(
                    np.array([elt[1] for elt in episode['actions']]) == 1)[0].astype('int')
                episode['at_seq'] = [float(elt[1])
                                    for elt in episode['actions']]
                if episode['actions'][-1][0] == 1:
                    episode['licktime'] = int(len(episode['states']))
                else:
                    episode['lick_licktimetime'] = -1
                episode['trial_len'] = len(episode['actions'])

                results.append(episode)

    df = pd.DataFrame(results)
    print(f'total reward per trial: {sum(np.concatenate(df.rewards.to_numpy()))/len(df)}')

    hit_reaction_plot(df)
    



print("=== Done: notebooks/vary reward gamma=98.ipynb ===")
