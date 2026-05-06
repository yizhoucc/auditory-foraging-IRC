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

print("=== Starting: yctest/for lokesh varying nodes.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))

# --- Cell 1 ---
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
import multiprocess
from auditoryforage.AF_env import AF2p, AuditoryForaging,AuditoryForagingReward2


# --- Cell 2 ---
modelname = 'varynodesv2'
# training hyper params
epoch_size = 22222
no_episodes=7777
n_epoch = 50
n_seed = 1
modeli = 0
attcost=-1.6
facost=-50

# reward list
vmin, vmax = 10, 5000
food_reward_list = np.linspace(vmin, vmax, 5)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])
env_param = [0, None, attcost, .25, facost, 0, 0]
seed=0
n_nodes=[15,25,35,45]
p1p2=np.array([.5,.7])

for n_node in n_nodes:
    thismodel = f'{modelname}_{n_node}'
    print(thismodel)
    np.random.seed(seed)
    task = AuditoryForagingReward2(spec={'agent': 
                                         {'lick_cost': env_param[0],
                                        'food_reward': env_param[1],
                                        'attention_cost_coeff': env_param[2],
                                        'attention_cost_temp': env_param[3],
                                        'penalty_cost': env_param[4],
                                        'iti_cost': env_param[5],
                                        'time_in_game_reward': env_param[6]},
                                'experiment':{                              'no_signal_nodes':n_node}})
    task.food_reward_list = food_reward_list
    taskbelief = FuncBeliefModel(env=task, rng=1)
    task.obs_certainity_possible=p1p2
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.01, ent_coef=0.01, tensorboard_log=f"runs/{modelname}",batch_size=512)
    # print(taskbelief.observation_space)

    for epochi in range(n_epoch):
        print('epoch', epochi)
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=thismodel,
                    reset_num_timesteps=False,
                    )
        model.save(f'ycstore/{thismodel}_{epochi}')

        # eval
        eps=0.001
        # collect trial data
        
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


# --- Cell 3 ---
for n_node in n_nodes:
    thismodel = f'{modelname}_{n_node}'
    print(thismodel)
    np.random.seed(seed)
    task = AuditoryForagingReward2(spec={'agent': 
                                         {'lick_cost': env_param[0],
                                        'food_reward': env_param[1],
                                        'attention_cost_coeff': env_param[2],
                                        'attention_cost_temp': env_param[3],
                                        'penalty_cost': env_param[4],
                                        'iti_cost': env_param[5],
                                        'time_in_game_reward': env_param[6]},
                                'experiment':{                              'no_signal_nodes':n_node}})
    task.food_reward_list = food_reward_list
    taskbelief = FuncBeliefModel(env=task, rng=1)
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.01, ent_coef=0.01, tensorboard_log=f"runs/{modelname}",batch_size=512)
    print(taskbelief.observation_space)

# --- Cell 4 ---
task.no_attention_modes, task.no_signal_nodes
# b=task.init_belief(task.observe_step(0))
# b

# --- Cell 5 ---
task.reset()
task.step([0,1])

# --- Cell 6 ---
task.find_observation_matrix().shape, task.find_transition_matrix().shape

# --- Cell 7 ---
b=task.update_belief(b, [0,0], [0])
b

# --- Cell 8 ---
modelname = 'varynodes'
epochname='11'
plotname='varynodes'


no_episodes=3333
n_epoch = 55
n_seed = 1
modeli = 0
attcost=-1.6
facost=-50

# reward list
vmin, vmax = 10, 5000
food_reward_list = np.linspace(vmin, vmax, 5)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])
env_param = [0, None, attcost, .25, facost, 0, 0]
seed=0
# n_nodes=[5,40]


epochi=5
for food_reward_idx in range(len(food_reward_list)):

    for n_node in n_nodes:
        task = AuditoryForagingReward2(spec={'agent': 
                                            {'lick_cost': env_param[0],
                                            'food_reward': env_param[1],
                                            'attention_cost_coeff': env_param[2],
                                            'attention_cost_temp': env_param[3],
                                            'penalty_cost': env_param[4],
                                            'iti_cost': env_param[5],
                                            'time_in_game_reward': env_param[6]},
                                    'experiment':{                              'no_signal_nodes':n_node}})
        task.food_reward_list = food_reward_list
        taskbelief = FuncBeliefModel(env=task, rng=1)
        model = PPO.load(f'ycstore/{modelname}_{n_node}_{epochi}')



        # eval
        eps=0.001
        # collect trial data
        
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
        plt.plot(xs,average_autocorrelation,label=f'{n_node} food reward={food_reward_list[food_reward_idx]:.0f}')

    plt.legend()
    plt.xlabel('tau')
    plt.xlim(-80,80)
    plt.ylabel('auto correlation varying reward')
    pass

# --- Cell 9 ---
notify()

print("=== Done: yctest/for lokesh varying nodes.ipynb ===")
