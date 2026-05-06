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

print("=== Starting: notebooks/vary node np.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AuditoryForagingReward2 ,AFnorate
from matplotlib.colors import  ListedColormap, BoundaryNorm

# --- Cell 1 ---
# train all agent at the same time (not family)
modelname = 'vary node np v4'

epoch_size = 55555
n_epoch = 33
n_seed = 1
seed = 0
no_episodes = 2222  # for eval plot during training
plot_no_episodes = 555 # for raster plot
vmin, vmax = 3, 100
food_reward_list = np.linspace(vmin, vmax, 5)
attcost = -1.6
facost = -50
p1p2 = np.array([.6, .8])
env_param = [0, None, attcost, .25, facost, 0, 0]
np.random.seed(seed)
node_list = [25, 40, 65, 80]
minlen, maxlen=0,np.inf
nbin=15
bins=np.linspace(0,1,nbin)

# --- Cell 2 ---
# color bars
num_discrete_var = len(node_list)
discrete_colors = nodecmap(np.linspace(0, 1, num_discrete_var))
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
cbar.set_label('Nodes')
cbar.set_ticks(np.arange(num_discrete_var) + 0.5)
cbar.set_ticklabels(node_list)
pass

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
for iepoch in range(0, n_epoch):
    # training ----------------
    for node in node_list:
        thismodel = f'seed{seed}_{modelname}_n{node}_ep{iepoch}'
    
        task = AFnorate(spec={'agent':
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
            previous_model = f'seed{seed}_{modelname}_n{node}_ep{iepoch-1}'
            model = PPO.load(f'ycstore/{previous_model}', env=taskbelief)
        model.learn(total_timesteps=epoch_size,)
        model.save(f'ycstore/{thismodel}')
        print(f'epoch{iepoch}, node{node}: yctore/{thismodel} saved')
        model = PPO.load(f'ycstore/{thismodel}', env=taskbelief) # test load


    # eval collection --------------------
    results=[]
    for inode, node in enumerate(node_list):
        task = AFnorate(spec={'agent': 
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
        thismodel = f'seed{seed}_{modelname}_n{node}_ep{iepoch}'
        model = PPO.load(f'ycstore/{thismodel}')

        def eval_wrapper(a):
            episode = run_one_episode(task=task, taskbelief=taskbelief, agent=model,
                                    num_steps=100000, deterministic=True)
            return episode

        with multiprocess.Pool(processes=8) as pool:
            all_episode_data = pool.map(eval_wrapper, range(no_episodes))
        
        for episode in all_episode_data:
            # process
            episode['inode'] = inode
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


    # eval ploting -------------------
    # for each reward, vary node
    for food_reward_idx, food_reward in enumerate(food_reward_list):
        plotdata=[] # for plot data
        fig,ax = plt.subplots(1,1,figsize=(4,3))
        for inode, node in enumerate(node_list):
            bindata=[[] for _ in range(nbin)] 
            subdf=df[(df.trial_food_reward_idx==food_reward_idx)&(df.inode==inode)]
            sb, nextatt=[],[]
            for alist, b in zip(subdf.at_times, subdf.beliefs):
                try:
                    a,b=trialnextatt(alist,b),trialsb(b)  
                    for j in range(len(b)):
                        thisa,thisb=a[j], b[j]
                        for i, (s,e) in enumerate(zip(bins, bins[1:])):
                            if thisa>0 and s<thisb<=e:
                                bindata[i].append(thisa)
                except:
                    continue
            
            mu, err=[np.nanmean(a) for a in bindata],[np.nanstd(a) for a in bindata]
            mu, err=np.array(mu), np.array(err)
            med=np.array([np.nanmedian(a) for a in bindata])
            percentile_5 = np.array([np.percentile(data, 5) if data else 0 for data in bindata])
            ax.plot(bins, percentile_5,color=nodecmap(np.linspace(0, 1, len(node_list)))[inode])
            centerax(ax)
        # ax.set_ylim(0, 2)
        ax.set_xlabel('belief signal prob.',labelpad=1)
        ax.set_ylabel('wait time',labelpad=1)
        # ax.set_yticks([0,15])
        ax.set_xticks([0,0.5,1],['0','','1'])
        ax.set_title(f'ep{iepoch}, reward:{food_reward}, node list:{node_list}')
        pass

    # # for each node, vary reward
    # for inode, node in enumerate(node_list):
    #     plotdata=[] # for plot data
    #     fig,ax = plt.subplots(1,1,figsize=(4,3))
    #     for food_reward_idx, food_reward in enumerate(food_reward_list):
    #         bindata=[[] for _ in range(nbin)] 
    #         subdf=df[(df.trial_food_reward_idx==food_reward_idx)&(df.inode==inode)]
    #         sb, nextatt=[],[]
    #         for alist, b in zip(subdf.at_times, subdf.beliefs):
    #             try:
    #                 a,b=trialnextatt(alist,b),trialsb(b)  
    #                 for j in range(len(b)):
    #                     thisa,thisb=a[j], b[j]
    #                     for i, (s,e) in enumerate(zip(bins, bins[1:])):
    #                         if thisa>0 and s<thisb<=e:
    #                             bindata[i].append(thisa)
    #             except:
    #                 continue
            
    #         mu, err=[np.nanmean(a) for a in bindata],[np.nanstd(a) for a in bindata]
    #         mu, err=np.array(mu), np.array(err)
    #         med=np.array([np.nanmedian(a) for a in bindata])
    #         percentile_5 = np.array([np.percentile(data, 5) if data else 0 for data in bindata])
    #         ax.plot(bins, percentile_5,color=rewardcmap(np.linspace(0, 1, len(food_reward_list)))[food_reward_idx])
    #         centerax(ax)
    #     # ax.set_ylim(0, 2)
    #     ax.set_xlabel('belief signal prob.',labelpad=1)
    #     ax.set_ylabel('wait time',labelpad=1)
    #     # ax.set_yticks([0,15])
    #     ax.set_xticks([0,0.5,1],['0','','1'])
    #     ax.set_title(f'ep{iepoch}, reward list:{food_reward_list}, node:{node}')
    #     pass


    # imshow 3d 
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vmin,vmax=0,50
    nbin=20
    bins=np.linspace(0,1,nbin)


    for food_reward_idx, food_reward in enumerate(food_reward_list):
        plotdata=[] # for plot data
        for inode, node in enumerate(node_list):
            bindata=[[] for _ in range(nbin)] 
            subdf=df[(df.trial_food_reward_idx==food_reward_idx)&(df.inode==inode)]
            sb, nextatt=[],[]
            for alist, b in zip(subdf.at_times, subdf.beliefs):
                try:
                    a,b=trialnextatt(alist,b),trialsb(b)  
                    for j in range(len(b)):
                        thisa,thisb=a[j], b[j]
                        for i, (s,e) in enumerate(zip(bins, bins[1:])):
                            if thisa>0 and s<thisb<=e:
                                bindata[i].append(thisa)
                except:
                    continue
            
            mu, err=[np.nanmean(a) for a in bindata],[np.nanstd(a) for a in bindata]
            mu, err=np.array(mu), np.array(err)
            med=np.array([np.nanmedian(a) for a in bindata])
            percentile_5 = np.array([np.percentile(data, 5) if data else 0 for data in bindata])
            percentile_95 = np.array([np.percentile(data, 95) if data else 0 for data in bindata])
            plotdata.append([percentile_5, percentile_95, mu])

        y_values = np.arange(len(node_list)-1, -1,-1)
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        for y, (y1, y2, mu) in zip(y_values, plotdata):
            verts = []
            for i in range(len(bins) - 1):
                verts.append([(bins[i], y, y1[i]), (bins[i+1], y, y1[i+1]), (bins[i+1], y, y2[i+1]), (bins[i], y, y2[i])])
            
            poly = Poly3DCollection(verts, facecolors=nodecmap(np.linspace(0, 1, len(node_list)))[len(node_list)-y-1], alpha=0.5, edgecolors='none')
            ax.add_collection3d(poly)
            ax.plot(bins,np.zeros_like(bins)+y, mu,color=nodecmap(np.linspace(0, 1, len(node_list)))[len(node_list)-y-1])

        ax.set_xlim(0, 1)
        # ax.set_ylim(0, 2)
        ax.set_zlim(0, 150)
        ax.set_yticks([0,1,2],[])
        # ax.set_zticks([0,25,50]) 
        ax.set_xticks([0,0.5,1],['0','','1'])
        ax.set_xlabel('belief signal prob.',labelpad=1)
        ax.set_ylabel('signal duration\nincreasing left',labelpad=1)
        ax.set_zlabel('wait time to next attn.',labelpad=1)

        plt.tight_layout()
        ax.view_init(azim=-40, elev=11) 
        quicksave('vary node 3d np','plot wait time', fig=fig)
        pass



# --- Cell 4 ---

    # imshow 3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vmin,vmax=0,50
    nbin=20
    bins=np.linspace(0,1,nbin)


    for food_reward_idx, food_reward in enumerate(food_reward_list):
        plotdata=[] # for plot data
        for inode, node in enumerate(node_list):
            bindata=[[] for _ in range(nbin)] 
            subdf=df[(df.trial_food_reward_idx==food_reward_idx)&(df.inode==inode)]
            sb, nextatt=[],[]
            for alist, b in zip(subdf.at_times, subdf.beliefs):
                try:
                    a,b=trialnextatt(alist,b),trialsb(b)  
                    for j in range(len(b)):
                        thisa,thisb=a[j], b[j]
                        for i, (s,e) in enumerate(zip(bins, bins[1:])):
                            if thisa>0 and s<thisb<=e:
                                bindata[i].append(thisa)
                except:
                    continue
            
            mu, err=[np.nanmean(a) for a in bindata],[np.nanstd(a) for a in bindata]
            mu, err=np.array(mu), np.array(err)
            med=np.array([np.nanmedian(a) for a in bindata])
            percentile_5 = np.array([np.percentile(data, 5) if data else 0 for data in bindata])
            percentile_95 = np.array([np.percentile(data, 95) if data else 0 for data in bindata])
            plotdata.append([percentile_5, percentile_95, mu])

        y_values = np.arange(len(node_list)-1, -1,-1)
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        for y, (y1, y2, mu) in zip(y_values, plotdata):
            verts = []
            for i in range(len(bins) - 1):
                verts.append([(bins[i], y, y1[i]), (bins[i+1], y, y1[i+1]), (bins[i+1], y, y2[i+1]), (bins[i], y, y2[i])])
            
            poly = Poly3DCollection(verts, facecolors=nodecmap(np.linspace(0, 1, len(node_list)))[len(node_list)-y-1], alpha=0.5, edgecolors='none')
            ax.add_collection3d(poly)
            ax.plot(bins,np.zeros_like(bins)+y, mu,color=nodecmap(np.linspace(0, 1, len(node_list)))[len(node_list)-y-1])

        ax.set_xlim(0, 1)
        # ax.set_ylim(0, 2)
        ax.set_zlim(0, 222)
        ax.set_yticks([0,1,2],[])
        # ax.set_zticks([0,25,50]) 
        ax.set_xticks([0,0.5,1],['0','','1'])
        ax.set_xlabel('belief signal prob.',labelpad=1)
        ax.set_ylabel('signal duration\nincreasing left',labelpad=1)
        ax.set_zlabel('wait time to next attn.',labelpad=1)

        plt.tight_layout()
        ax.view_init(azim=-40, elev=11) 
        quicksave('vary node 3d np','plot wait time', fig=fig)
        pass




print("=== Done: notebooks/vary node np.ipynb ===")
