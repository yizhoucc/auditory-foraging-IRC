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

print("=== Starting: yctest/p1p2 new env train.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))

# --- Cell 1 ---
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AF2p


# --- Cell 2 ---
modelname = '5678vs9'
# training hyper params
epoch_size = 55555
n_epoch = 55
n_seed = 1
modeli = 0
attcost=-1.6
facost=-50

# reward list
vmin, vmax = 10, 5000
food_reward_list = np.linspace(vmin, vmax, 4)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])
env_param = [0, None, attcost, .25, facost, 0, 0]

seed=0
thismodel = f'seed_{seed}_{modelname}_{modeli}'
print(thismodel)
for seed in range(n_seed):
    np.random.seed(seed)
    task = AF2p(spec={'agent': {'lick_cost': env_param[0],
                                'food_reward': env_param[1],
                                'attention_cost_coeff': env_param[2],
                                'attention_cost_temp': env_param[3],
                                'penalty_cost': env_param[4],
                                'iti_cost': env_param[5],
                                'time_in_game_reward': env_param[6]}})
    task.food_reward_list = food_reward_list
    taskbelief = FuncBeliefModel(env=task, rng=1)
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.01, ent_coef=0.01, tensorboard_log=f"runs/{modelname}",batch_size=512)
task.p1id

# --- Cell 3 ---
for epochi in range(0, 44):
    
    print('epoch', epochi)
    model.learn(total_timesteps=epoch_size,
                tb_log_name=thismodel,
                reset_num_timesteps=False,
                )
    model.save(f'ycstore/{thismodel}_{epochi}')

    # eval
    eps=0.001
    # collect trial data
    import multiprocess
    no_episodes=11111
    def eval_wrapper(a):
        episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                num_steps=100000, deterministic=True)
        return episode

    with multiprocess.Pool(processes=8) as pool:
        all_episode_data = pool.map(eval_wrapper, range(no_episodes))

    plotname='plotname'
    all_reward_res=[]# k: v = foodreward: this food reward res

    for episode in all_episode_data:
        r=episode['trial_food_reward_idx']
        p1p2=episode['p1p2']
        att_time_list,lick_time_list=[],[]
        att_time_list.append(
            np.where(np.array([elt[1] for elt in episode['actions']]) == 1)[0])
        if episode['actions'][-1][0] == 1:
            lick_time_list.append(len(episode['states']))

        all_reward_res.append((r,p1p2[0], p1p2[1],att_time_list, lick_time_list,episode))
    import pandas as pd
    df = pd.DataFrame(all_reward_res,columns = ['reward', 'p1', 'p2', 'att_time', 'lick_time', 'episode'])


    # eval
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

    # auto correlation varying reward
    for food_reward_idx in range(len(food_reward_list)):
        att_seq_list = []
        for episode in all_episode_data:
            if episode['trial_food_reward_idx']==food_reward_idx:
                att_seq_list.append([float(elt[1]) for elt in episode['actions']])
        # Sorting based on increasing trial length
        sorted_list_of_lists = sorted(att_seq_list, key=len)
        print(f'using {len(sorted_list_of_lists)} of trails')
        sequences = sorted_list_of_lists[-len(sorted_list_of_lists)//2:] # Choosing what range of trial lengths to use. Could modify to be more consistent.
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
    plt.legend()
    plt.xlabel('tau')
    plt.xlim(-80,80)
    plt.ylabel('auto correlation varying reward')
    pass
    # plt.title(f'{i}')

    # varying p1
    for r in sorted(df['reward'].unique()):
        for p1 in sorted(df['p1'].unique()):
            p2=sorted(df['p2'].unique())[-1]
            att_seq_list = []
            for episode in all_episode_data:
                if episode['trial_food_reward_idx']==r and p1-eps<episode['p1p2'][0]<p1+eps and p2-eps<episode['p1p2'][1]<eps+p2:
                    att_seq_list.append([float(elt[1]) for elt in episode['actions']])
            
            sorted_list_of_lists = sorted(att_seq_list, key=len)
            print(f'using {len(sorted_list_of_lists)} of trails')
            sequences = sorted_list_of_lists[-len(sorted_list_of_lists)//2:] # Choosing what range of trial lengths to use. Could modify to be more consistent.
            normalized_sequences = []
            max_length = 1
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
            plt.plot(xs,average_autocorrelation,label=f'{p1:.2f} {p2:.2f}')
        plt.xlabel('tau')
        plt.ylabel('autocorrelation')
        plt.title(f'autocorr varying p1 \n food reward={food_reward_list[r]:.0f}\n, {modelname} {plotname}')
        plt.xlim(-80,80)
        plt.legend()
        # quicksave('autocorr', plotname)
        pass

    # # varying p2
    # for r in sorted(df['reward'].unique()):
    #     for p2 in sorted(df['p2'].unique()):
    #         p1=0.5
    #         att_seq_list = []
    #         for episode in all_episode_data:
    #             if episode['trial_food_reward_idx']==r and p1-eps<episode['p1p2'][0]<p1+eps and p2-eps<episode['p1p2'][1]<eps+p2:
    #                 att_seq_list.append([float(elt[1]) for elt in episode['actions']])
            
    #         sorted_list_of_lists = sorted(att_seq_list, key=len)
    #         sequences = sorted_list_of_lists[-len(sorted_list_of_lists)//2:] # Choosing what range of trial lengths to use. Could modify to be more consistent.
    #         normalized_sequences = []
    #         max_length = 1
    #         for sequence in sequences:
    #             normalized_seq = np.array(sequence)
    #             if not np.isnan(normalized_seq.any()):
    #                 normalized_sequences.append(normalized_seq)
    #                 max_length = max(max_length, len(normalized_seq))    
                
    #         padded_sequences = []
    #         for seq in normalized_sequences:
    #             padded_seq = np.pad(seq, (0, max_length - len(seq)), mode='constant')
    #             padded_sequences.append(padded_seq)

    #         autocorrelation_sum = np.zeros(2 * max_length - 1)
    #         for seq in padded_sequences:
    #             autocorrelation_sum += np.nan_to_num(compute_autocorrelation(seq), nan=0)

    #         average_autocorrelation = autocorrelation_sum / len(padded_sequences)
    #         xs=np.arange(0,len(average_autocorrelation),1)
    #         xs=xs-len(xs)//2
    #         plt.plot(xs,average_autocorrelation,label=f'{p1:.2f} {p2:.2f}')
    #     plt.xlabel('tau')
    #     plt.ylabel('autocorrelation')
    #     plt.title(f'autocorr varying p2 \n food reward={food_reward_list[r]:.0f}\n, {modelname} {plotname}')
    #     plt.xlim(-80,80)
    #     plt.legend()
    #     # quicksave('autocorr', plotname)
    #     pass


# --- Cell 4 ---
# heatmap trial indicator (att and lick)
plot_no_episodes=3333

for r in df['reward'].unique():
    att_time_list,trial_len=df[df.reward==r].att_time.to_list(), df[df.reward==r].lick_time.to_list()
    
    # sortind=np.argsort(trial_len[:plot_no_episodes])

    grid = np.zeros((plot_no_episodes, max(trial_len)[0]+2))

    # att
    plot_data =att_time_list[:plot_no_episodes]
    for i, arr in enumerate(plot_data):
        grid[i, arr] = -1

    # lick
    plot_data =lick_time_list[:plot_no_episodes]
    for i, arr in enumerate(plot_data):
        if arr:
            grid[i, arr] = 1
    
    plt.figure()
    c=plt.imshow(grid[:], cmap='bwr', vmin=-1, vmax=1,
                    aspect='auto', interpolation='none')
    # plt.colorbar(c)
    plt.xlabel('time in trial')
    plt.ylabel('episode no.')
    plt.title(f'attn, food reward: {food_reward_list[food_reward_idx]}')

    # legend
    from matplotlib.patches import Patch

    # Create Patch objects for the custom legend
    red_patch = Patch(color='red', label='lick')
    blue_patch = Patch(color='blue', label='attention')

    # Create a new legend
    extra_legend = plt.legend(handles=[red_patch, blue_patch], loc='upper right', frameon=True)
    frame = extra_legend.get_frame()
    frame.set_color('lightgrey')

    # Add the new legend to the existing plot
    plt.gca().add_artist(extra_legend)
    quicksave(f'ep indicator {food_reward_idx}', modelname)
    pass


# --- Cell 5 ---
# model.learn(total_timesteps=epoch_size,
#             tb_log_name=thismodel,
#             reset_num_timesteps=False,
#             )
no_episodes=9999*2*6
def eval_wrapper(a):
    episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                            num_steps=100000, deterministic=True)
    return episode

with multiprocess.Pool(processes=8) as pool:
    all_episode_data = pool.map(eval_wrapper, range(no_episodes))


# auto correlation    
eps=0.001
p2=0.7
for food_reward_idx in range(len(food_reward_list)):
    for p1 in [0.5,0.55,0.6,0.65]:
        att_seq_list = []
        for episode in all_episode_data:
            # print(episode['p1p2'], p1, p2)
            if episode['trial_food_reward_idx']==food_reward_idx and p1-eps<episode['p1p2'][0]<p1+eps and p2-eps<episode['p1p2'][1]<eps+p2:
                att_seq_list.append([float(elt[1]) for elt in episode['actions']])
            
        # Sorting based on increasing trial length
        sorted_list_of_lists = sorted(att_seq_list, key=len)
        sequences = sorted_list_of_lists[-333:-1] # Choosing what range of trial lengths to use. Could modify to be more consistent.
        normalized_sequences = []
        max_length = 1
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
        plt.plot(xs,average_autocorrelation,label=f'{p1:.2f} {p2:.2f}')
        
    plt.xlabel('tau')
    plt.ylabel('autocorrelation')
    plt.title(f'auto corr same reward vary p1\n food reward={food_reward_list[food_reward_idx]:.0f}\n, {modelname} {plotname}')
    plt.xlim(-80,80)
    plt.legend()
    # quicksave('autocorr', plotname)
    pass

# auto correlation    
eps=0.001
p1=0.5
for food_reward_idx in range(len(food_reward_list)):
    for p2 in np.arange(0.55,0.8,0.05):
        att_seq_list = []
        for episode in all_episode_data:
            # print(episode['p1p2'], p1, p2)
            if episode['trial_food_reward_idx']==food_reward_idx and p1-eps<episode['p1p2'][0]<p1+eps and p2-eps<episode['p1p2'][1]<eps+p2:
                att_seq_list.append([float(elt[1]) for elt in episode['actions']])
            
        # Sorting based on increasing trial length
        sorted_list_of_lists = sorted(att_seq_list, key=len)
        sequences = sorted_list_of_lists[-333:-1] # Choosing what range of trial lengths to use. Could modify to be more consistent.
        normalized_sequences = []
        max_length = 1
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
        plt.plot(xs,average_autocorrelation,label=f'{p1:.2f} {p2:.2f}')
        
    plt.xlabel('tau')
    plt.ylabel('autocorrelation')
    plt.title(f'auto corr same reward vary p2\n food reward={food_reward_list[food_reward_idx]:.0f}\n, {modelname} {plotname}')
    plt.xlim(-80,80)
    plt.legend()
    # quicksave('autocorr', plotname)
    pass

# --- Cell 6 ---
task.obs_certainity_possible
task.find_observation_matrix()[1]

# --- Cell 7 ---
# auto correlation    
eps=0.001
p2=0.7
for food_reward_idx in range(len(food_reward_list))[-2:-1]:
    for p1 in [0.5,0.55,0.6,0.65]:
        att_seq_list = []
        for episode in all_episode_data:
            # print(episode['p1p2'], p1, p2)
            if episode['trial_food_reward_idx']==food_reward_idx and p1-eps<episode['p1p2'][0]<p1+eps and p2-eps<episode['p1p2'][1]<eps+p2:
                att_seq_list.append([float(elt[1]) for elt in episode['actions']])
            
        # Sorting based on increasing trial length
        sorted_list_of_lists = sorted(att_seq_list, key=len)
        sequences = sorted_list_of_lists[-333:-1] # Choosing what range of trial lengths to use. Could modify to be more consistent.
        normalized_sequences = []
        max_length = 1
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
        plt.plot(xs,average_autocorrelation,label=f'{p1:.2f} {p2:.2f}')
        
    plt.xlabel('tau')
    plt.yscale('log')
    plt.ylabel('autocorrelation')
    plt.title(f'auto corr same reward vary p1\n food reward={food_reward_list[food_reward_idx]:.0f}\n, {modelname} {plotname}')
    plt.xlim(-80,80)
    plt.legend()
    # quicksave('autocorr', plotname)
    pass


# --- Cell 8 ---
# train for 50 and get plots
modelname = 'varp1'
# training hyper params
epoch_size = 33333
n_epoch = 55
n_seed = 1
modeli = 0

for attcost in [-3]:
    for facost in [-50,]:
            # reward list
            vmin, vmax = 10, 10000
            food_reward_list = np.linspace(vmin, vmax, 5)
            # food_reward_list=np.exp(food_reward_list)

            print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

            # other env params
            env_param = [0, None, attcost, .25, facost, 0, 0]
            seed=0
            thismodel = f'seed_{seed}_{modelname}_{modeli}'
            print(thismodel)
            for seed in range(n_seed):
                # set seed
                np.random.seed(seed)
                task = AF2p(spec={'agent': {'lick_cost': env_param[0],
                                            'food_reward': env_param[1],
                                            'attention_cost_coeff': env_param[2],
                                            'attention_cost_temp': env_param[3],
                                            'penalty_cost': env_param[4],
                                            'iti_cost': env_param[5],
                                            'time_in_game_reward': env_param[6]}})
                task.food_reward_list = food_reward_list
                taskbelief = FuncBeliefModel(env=task, rng=1)
     
                model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                            clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs/{modelname}",)
                


# --- Cell 9 ---


for e in range(n_epoch):
    print('epoch', e, '\n\n\n\asdf')
    model.learn(total_timesteps=epoch_size,
                tb_log_name=thismodel,
                reset_num_timesteps=False,
                )
    no_episodes=9999
    def eval_wrapper(a):
        episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                num_steps=100000, deterministic=True)
        return episode

    with multiprocess.Pool(processes=8) as pool:
        all_episode_data = pool.map(eval_wrapper, range(no_episodes))


    # auto correlation    
    eps=0.001
    p2=0.7

    for food_reward_idx in range(len(food_reward_list)):
        for p1 in [0.5,0.55,0.6,0.65]:
            att_seq_list = []
            for episode in all_episode_data:
                # print(episode['p1p2'], p1, p2)
                if episode['trial_food_reward_idx']==food_reward_idx and p1-eps<episode['p1p2'][0]<p1+eps and p2-eps<episode['p1p2'][1]<eps+p2:
                    att_seq_list.append([float(elt[1]) for elt in episode['actions']])
                
            # Sorting based on increasing trial length
            sorted_list_of_lists = sorted(att_seq_list, key=len)
            sequences = sorted_list_of_lists[-333:-1] # Choosing what range of trial lengths to use. Could modify to be more consistent.
            normalized_sequences = []
            max_length = 1
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
            plt.plot(xs,average_autocorrelation,label=f'{p1:.2f} {p2:.2f}')
            
        plt.xlabel('tau')
        plt.ylabel('autocorrelation')
        plt.title(f'auto corr same reward vary p1\n food reward={food_reward_list[food_reward_idx]:.0f}\n, {modelname} {plotname}')
        plt.xlim(-80,80)
        plt.legend()
        # quicksave('autocorr', plotname)
        pass

    # auto correlation    
    eps=0.001
    p1=0.5
    for food_reward_idx in range(len(food_reward_list)):
        for p2 in np.arange(0.55,0.8,0.05):
            att_seq_list = []
            for episode in all_episode_data:
                # print(episode['p1p2'], p1, p2)
                if episode['trial_food_reward_idx']==food_reward_idx and p1-eps<episode['p1p2'][0]<p1+eps and p2-eps<episode['p1p2'][1]<eps+p2:
                    att_seq_list.append([float(elt[1]) for elt in episode['actions']])
                
            # Sorting based on increasing trial length
            sorted_list_of_lists = sorted(att_seq_list, key=len)
            sequences = sorted_list_of_lists[-999:-1] # Choosing what range of trial lengths to use. Could modify to be more consistent.
            normalized_sequences = []
            max_length = 1
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
            plt.plot(xs,average_autocorrelation,label=f'{p1:.2f} {p2:.2f}')
            
        plt.xlabel('tau')
        plt.ylabel('autocorrelation')
        plt.title(f'auto corr same reward vary p2\n food reward={food_reward_list[food_reward_idx]:.0f}\n, {modelname} {plotname}')
        plt.xlim(-80,80)
        plt.legend()
        # quicksave('autocorr', plotname)
        pass

# --- Cell 10 ---
model=PPO.load('ycstore/seed_0_varRvarP_0_159')
no_episodes=9999
def eval_wrapper(a):
    episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                            num_steps=100000, deterministic=True)
    return episode

with multiprocess.Pool(processes=8) as pool:
    all_episode_data = pool.map(eval_wrapper, range(no_episodes))


# --- Cell 11 ---
# var p1
# auto correlation    
eps=0.001
p2=0.7

for food_reward_idx in range(len(food_reward_list)):
    for p1 in [0.5,0.55,0.6,0.65]:
        att_seq_list = []
        for episode in all_episode_data:
            # print(episode['p1p2'], p1, p2)
            if episode['trial_food_reward_idx']==food_reward_idx and p1-eps<episode['p1p2'][0]<p1+eps and p2-eps<episode['p1p2'][1]<eps+p2:
                att_seq_list.append([float(elt[1]) for elt in episode['actions']])
            
        # Sorting based on increasing trial length
        sorted_list_of_lists = sorted(att_seq_list, key=len)
        sequences = sorted_list_of_lists[-333:-1] # Choosing what range of trial lengths to use. Could modify to be more consistent.
        normalized_sequences = []
        max_length = 1
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
        plt.plot(xs,average_autocorrelation,label=f'{p1:.2f} {p2:.2f}')
        
    plt.xlabel('tau')
    plt.ylabel('autocorrelation')
    plt.yscale('log')
    plt.title(f'auto corr same reward vary p1\n food reward={food_reward_list[food_reward_idx]:.0f}\n, {modelname} {plotname}')
    plt.xlim(-44,44)
    plt.legend()
    # quicksave('autocorr', plotname)
    pass


# --- Cell 12 ---
# heatmap var p1 att
# heatmap
plot_no_episodes=500
for food_reward_idx in range(len(food_reward_list)):
    for p1 in [0.5,0.55,0.6,0.65]:
        att_seq_list = []
        for episode in all_episode_data:
            # print(episode['p1p2'], p1, p2)
            if episode['trial_food_reward_idx']==food_reward_idx and p1-eps<episode['p1p2'][0]<p1+eps and p2-eps<episode['p1p2'][1]<eps+p2:
                att_seq_list.append([float(elt[1]) for elt in episode['actions']])
            
        # Sorting based on increasing trial length
        sorted_list_of_lists = sorted(att_seq_list, key=len)
        
        grid = np.zeros((plot_no_episodes+1, max([len(a) for a in sorted_list_of_lists])+2))

        # att
        plot_data = sorted_list_of_lists[:plot_no_episodes]
        for i, arr in enumerate(plot_data):
            grid[i, np.where(np.array(arr)==1)[0]] = -1


        
        plt.figure()
        plt.title(f'{food_reward_idx}, {p1}')
        # plt.imshow(grid)
        c=plt.imshow(grid, cmap='bwr', vmin=-1, vmax=1,
                        aspect='auto', interpolation='none')
        plt.colorbar(c)
        plt.xlabel('time in trial')
        plt.ylabel('episode no.')
        plt.title(f'attn, food reward: {food_reward_list[food_reward_idx]:.0f}\n p1={p1} p2=0.7')

        # # legend
        # from matplotlib.patches import Patch

        # # Create Patch objects for the custom legend
        # red_patch = Patch(color='red', label='lick')
        # blue_patch = Patch(color='blue', label='attention')

        # # Create a new legend
        # extra_legend = plt.legend(handles=[red_patch, blue_patch], loc='upper right', frameon=True)
        # frame = extra_legend.get_frame()
        # frame.set_color('lightgrey')

        # # Add the new legend to the existing plot
        # plt.gca().add_artist(extra_legend)
        # # quicksave(f'ep indicator {food_reward_idx}', 'testmodel')
        # pass

# --- Cell 13 ---
# heatmap
plot_no_episodes=1000
for food_reward_idx in (all_reward_res.keys()):
        attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=all_reward_res[food_reward_idx]
        sortind=np.argsort(trial_len[:plot_no_episodes])

        grid = np.zeros((plot_no_episodes, max(trial_len)+2))

        # att
        plot_data =att_time_list[:plot_no_episodes]
        for i, arr in enumerate(plot_data):
            grid[i, arr] = -1

        # lick
        plot_data =lick_time_list[:plot_no_episodes]
        for i, arr in enumerate(plot_data):
            if arr:
                grid[i, arr] = 1
        
        plt.figure()
        c=plt.imshow(grid[sortind], cmap='bwr', vmin=-1, vmax=1,
                        aspect='auto', interpolation='none')
        # plt.colorbar(c)
        plt.xlabel('time in trial')
        plt.ylabel('episode no.')
        plt.title(f'attn, food reward: {food_reward_list[food_reward_idx]:.0f}')

        # legend
        from matplotlib.patches import Patch

        # Create Patch objects for the custom legend
        red_patch = Patch(color='red', label='lick')
        blue_patch = Patch(color='blue', label='attention')

        # Create a new legend
        extra_legend = plt.legend(handles=[red_patch, blue_patch], loc='upper right', frameon=True)
        frame = extra_legend.get_frame()
        frame.set_color('lightgrey')

        # Add the new legend to the existing plot
        plt.gca().add_artist(extra_legend)
        # quicksave(f'ep indicator {food_reward_idx}', 'testmodel')
        pass

# --- Cell 14 ---
task.obs_certainity_possible

# --- Cell 15 ---
b=taskbelief.reset()
o=task.observe_step(1)
b, o

# --- Cell 16 ---
task.find_observation_matrix()[1, 9, 0]


# --- Cell 17 ---
# added may1st, figure 6
plt.figure()
for food_reward_idx in all_reward_res.keys():
    
    attention_prob_list,lick_prob_list,noise_prob_list, lick_time_list,att_time_list,trial_len=all_reward_res[food_reward_idx]

    
    plt.hist([len(a) for a in att_time_list],label=f'food reward={food_reward_list[food_reward_idx]:.0f}', bins=np.linspace(0,15,15), alpha=0.3)
    # sns.kdeplot([len(a) for a in att_time_list],label=f'food reward={food_reward_list[food_reward_idx]:.0f}', bw_adjust=3, common_norm=False)
    # sns.histplot([len(a) for a in att_time_list],label=f'food reward={food_reward_list[food_reward_idx]:.0f}')
    # Generate the KDE estimate
    # data=[len(a) for a in att_time_list]
    # kde = sns.kdeplot(data, bw_method='silverman', color='white')
    # y_vals = kde.get_lines()[0].get_ydata()
    # n = len(data)
    # y_vals_counts = y_vals * n
    # plt.plot(kde.get_lines()[0].get_xdata(), y_vals_counts, color='blue')

plt.xlabel('number of high attentions')
plt.ylabel('count')
plt.title(f'total attention times, varying food reward')
# plt.ylim(0,1500)
# plt.xlim(5,60)
plt.legend()
pass

# --- Cell 18 ---
import seaborn as sns
import matplotlib.pyplot as plt

# Example data
data = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5]

# Generate the KDE estimate
kde = sns.kdeplot(data, bw_method='silverman', color='red')

# Get the y values of the KDE line
y_vals = kde.get_lines()[0].get_ydata()

# Convert y values to counts
n = len(data)
y_vals_counts = y_vals * n

# Plot the modified KDE line with counts on the y-axis
# plt.plot(kde.get_lines()[0].get_xdata(), y_vals_counts, color='blue')

# Show the plot
pass


print("=== Done: yctest/p1p2 new env train.ipynb ===")
