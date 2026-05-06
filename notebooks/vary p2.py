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

print("=== Starting: notebooks/vary p2.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from ult import *
from irc_gym.irc.model import FuncBeliefModel
from stable_baselines3 import PPO
from auditoryforage.AF_env import AuditoryForagingReward2 as AFR2
from auditoryforage.AF_env import AFnorate


# --- Cell 1 ---
# train all agent at the same time (not family)
modelname = 'vary p2'
plist = [np.array([.5, .6]),np.array([.5, 1])]


epoch_size = 33333*5
n_epoch = 22
n_seed = 1
seed = 0
no_episodes = 2222  # for eval plot during training
plot_no_episodes = 555
vmin, vmax = 1, 100
food_reward_list = np.linspace(vmin, vmax, 5)
attcost = -1.6
facost = -50
node=25
env_param = [0, None, attcost, .25, facost, 0, 0]
np.random.seed(seed)
minlen, maxlen=50,100

# --- Cell 2 ---
for iepoch in range(9, n_epoch):
    # training ----------------
    for ip, p1p2 in enumerate(plist):
        thismodel = f'seed{seed}_{modelname}_p{ip}_ep{iepoch}'
    
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
            previous_model = f'seed{seed}_{modelname}_p{ip}_ep{iepoch-1}'
            model = PPO.load(f'ycstore/{previous_model}', env=taskbelief)
        model.learn(total_timesteps=epoch_size,)
        model.save(f'ycstore/{thismodel}')
        print(f'epoch{iepoch}, p{p1p2}: yctore/{thismodel} saved')


    # collect for eval
    results=[]
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
            # process
            episode['ip'] = ip
            episode['at_times'] = np.where(
                np.array([elt[1] for elt in episode['actions']]) == 1)[0].astype('int')
            episode['at_seq'] = [float(elt[1])
                                for elt in episode['actions']]
            if episode['actions'][-1][0] == 1:
                episode['licktime'] = int(len(episode['states']))
            else:
                episode['licktime'] = -1
            episode['trial_len'] = len(episode['actions'])
            results.append(episode)
    df = pd.DataFrame(results)

    # imshow
    vmin,vmax=0,50
    nbin=20
    for food_reward_idx in range(len(food_reward_list)):
        for ip, p1p2 in enumerate(plist):
            grid=np.zeros((vmax, nbin))
            bins=np.linspace(0,1,nbin)
            bindata=[[] for _ in range(nbin)] 
            subdf=df[(df.trial_food_reward_idx==food_reward_idx)&(df.ip==ip)]
            sb, nextatt=[],[]
            for alist, b in zip(subdf.at_times, subdf.beliefs):
                try:
                    a,b=trialnextatt(alist,b),trialsb(b)  
                    for j in range(len(b)):
                        thisa,thisb=a[j], b[j]
                        for i, (s,e) in enumerate(zip(bins, bins[1:])):
                            if s<thisb<=e:
                                bindata[i].append(thisa)
                except:
                    continue
            for i in range(nbin):
                count=Counter(bindata[i])
                col=[]
                for j in range(vmin, vmax):
                    col.append(count[j])
                col=np.array(col)
                col=col/np.sum(col)
                grid[:,i]=col
            fig=plt.figure()
            # grid=np.log(grid)
            c=plt.imshow(grid,aspect='auto', origin='lower')
            plt.colorbar(c)
            plt.xlabel('prob')
            plt.ylabel('gap')
            plt.title(f'reward id: {food_reward_idx}, \np:{p1p2},\nnode:{node}')
            plt.xticks([0,nbin],[0,1])
            # quicksave(f'vary node np, reward id {food_reward_idx}, node {node}', 'test', fig=fig)
            pass
            plt.close()

    # raster df
    for food_reward_idx in range(len(food_reward_list)):
        rasterfig, axs = plt.subplots(len(plist), 1, figsize=(
            3, len(plist)*2,), sharex=True, sharey=True)
        for ip, p1p2 in enumerate(plist):
            subdf=df[(df.trial_food_reward_idx==food_reward_idx) & (df.ip==ip)]
            # df=df[df['at_times'].apply(len) > 0]
            subdf=subdf[subdf.licktime!=-1]
            # df.licktime=df.licktime.astype('int')
            subdf=subdf[~subdf.licktime.isna()]
            if len(subdf)==0: continue
            trial_len = np.array(subdf.trial_len)
            valid_ind = np.where((minlen < trial_len) & (trial_len < maxlen))[0]
            grid = np.zeros((len(valid_ind), max(trial_len)+2))
            # print(food_reward_idx,ip, grid.shape)
            sortind = np.argsort([trial_len[i] for i in valid_ind])

            # att
            plot_data = np.array(subdf.at_times)[valid_ind]
            for i, arr in enumerate(plot_data):
                grid[i, arr] = -1
            # lick
            plot_data = np.array(subdf.licktime)[valid_ind]
            for i, arr in enumerate(plot_data):
                if arr:
                    grid[i, int(arr)] = 1
            c = axs[ip].imshow(grid, cmap=attnlickcmap, vmin=-1, vmax=1,
                            aspect='auto', interpolation='none')

            axs[ip].set_xticks([0, 50, 100])
            axs[ip].set_xlim(0, 100)
            axs[ip].set_ylim(0, 300)
            axs[ip].set_yticks([])
            axs[ip].spines['left'].set_visible(False)
            axs[ip].set_ylabel(f'{ip} p')
        rasterfig.suptitle(f'{food_reward_idx}, food')
        rasterfig.show()

    # total att time
    for food_reward_idx in np.arange(len(food_reward_list)):
        fig= plt.figure()
        for ip, p1p2 in enumerate(plist):
            subdf=df[(df.trial_food_reward_idx==food_reward_idx) &(df.ip==ip)]
            sns.kdeplot([len(a) for a in subdf.at_times],  label=f'food reward={food_reward_list[food_reward_idx]:.0f}, p={p1p2}', bw_adjust=2, common_norm=False, color=pcmap(
                np.linspace(0, 1, len(plist)))[ip])
        color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap=pcmap), ax=plt.gca())
        ticks = [0, 0.5, 1]
        tick_labels = ['Low p1', '', 'High p1']
        color_bar.set_ticks(ticks)
        color_bar.set_ticklabels(tick_labels)
        plt.xlabel('number of high attentions')
        plt.ylabel('prob')
        plt.title(f'total attention times, varying p')
        # plt.ylim(0,1500)
        # plt.legend()
        centerax(plt.gca())
        plt.yticks([])
        plt.xlim(0, 20)
        # plt.xticks([0, 15])
        quicksave('total att time', 'plot sigmoid', fig=fig)
        pass


    # autocorr df. vary reward, p
    for food_reward_idx in range(len(food_reward_list)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        for ip, p1p2 in enumerate(plist):
            subdf=df[(df.trial_food_reward_idx==food_reward_idx) & (df.ip==ip)]
            if len(subdf)==0: 
                print(food_reward_idx, ip, 'skipped')
                continue
            att_seq_list = np.array(subdf.at_seq)
            # print(len(subdf), food_reward_idx, ip)
            autocorr, total_counts = empirical_autocorrelation_new(
                att_seq_list, min_no_samples=1111)
            # print(att_seq_list)
            xs = np.arange(0, len(autocorr), 1)
            xs = xs-len(xs)//2
            ax1.plot(xs, autocorr,
                    label=f'p:{p1p2}', color=pcmap(np.linspace(0, 1, len(plist)))[ip])
            # trial len dist
            sns.kdeplot([len(a) for a in att_seq_list],
                        fill=False, bw_adjust=0.5, ax=ax2, color=pcmap(np.linspace(0, 1, len(plist)))[ip])
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
        color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap=pcmap), ax=ax2)
        ticks = [0, 0.5, 1]
        tick_labels = [f'Low p={plist[0]}', '', f'High p={plist[-1]}']
        color_bar.set_ticks(ticks)
        color_bar.set_ticklabels(tick_labels)
        fig.suptitle(
            f'vary p1, p2={p1p2[1]} food \nreward={food_reward_list[food_reward_idx]:.0f}')
        fig.tight_layout()
        pass

    # att size and gap
    for food_reward_idx, reward in enumerate(food_reward_list):
        fig, axs = plt.subplots(len(plist), 1, figsize=(4, 4), sharex=True)
        for ip, p1p2 in enumerate(plist):
            subdf=df[(df.trial_food_reward_idx==food_reward_idx) &(df.ip==ip)]
            gaps=subdf.at_times.apply(find_gaps).to_list()
            gaps=np.concatenate(gaps,)
            axs[len(plist)-ip-1].hist(gaps, bins=np.arange(33), color=pcmap(np.linspace(0, 1, len(plist))[ip]), density=True, alpha=1)
            # sns.kdeplot(gaps, color=pcmap(np.linspace(0, 1, len(plist))[ip]),bw_adjust=2, common_norm=False) 
        plt.suptitle(f'r:{reward}')
        axs[ip].set_xlabel('gap size')
        plt.ylabel('prob')
    fig, axs = plt.subplots(1,1, figsize=(1,3))
    color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap=pcmap), ax=plt.gca())
    ticks = [0, 0.5, 1]
    tick_labels = ['Low p1', '', 'High p1']
    color_bar.set_ticks(ticks)
    color_bar.set_ticklabels(tick_labels)
    pass

# --- Cell 3 ---
# eval, df
# iepoch=10
print(f'iepoch: {iepoch}')

modelname = 'vary p2'
plist = [np.array([.5, .6]),np.array([.5, .8]),np.array([.5, 1])]
epoch_size = 33333
n_epoch = 66
n_seed = 1
seed = 0
no_episodes = 3333  # for eval plot during training
plot_no_episodes = 555
vmin, vmax = 10, 10000
food_reward_list = np.linspace(vmin, vmax, 5)
attcost = -1.6
facost = -50
node=25
env_param = [0, None, attcost, .25, facost, 0, 0]
np.random.seed(seed)
minlen, maxlen=50,100


results=[]
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
        # process
        episode['ip'] = ip
        episode['at_times'] = np.where(
            np.array([elt[1] for elt in episode['actions']]) == 1)[0].astype('int')
        episode['at_seq'] = [float(elt[1])
                             for elt in episode['actions']]
        if episode['actions'][-1][0] == 1:
            episode['licktime'] = int(len(episode['states']))
        else:
            episode['licktime'] = -1
        episode['trial_len'] = len(episode['actions'])
        results.append(episode)

df = pd.DataFrame(results)

# --- Cell 4 ---
# raster df
for food_reward_idx in range(len(food_reward_list)):
    rasterfig, axs = plt.subplots(len(plist), 1, figsize=(
        3, len(plist)*2,), sharex=True, sharey=True)
    for ip, p1p2 in enumerate(plist):
        subdf=df[(df.trial_food_reward_idx==food_reward_idx) & (df.ip==ip)]

        # df=df[df['at_times'].apply(len) > 0]
        subdf=subdf[subdf.licktime!=-1]
        # df.licktime=df.licktime.astype('int')
        subdf=subdf[~subdf.licktime.isna()]

        if len(subdf)==0: continue
        trial_len = np.array(subdf.trial_len)
        valid_ind = np.where((minlen < trial_len) & (trial_len < maxlen))[0]
        grid = np.zeros((len(valid_ind), max(trial_len)+2))
        # print(food_reward_idx,ip, grid.shape)
        sortind = np.argsort([trial_len[i] for i in valid_ind])

        # att
        plot_data = np.array(subdf.at_times)[valid_ind]
        for i, arr in enumerate(plot_data):
            grid[i, arr] = -1
        # lick
        plot_data = np.array(subdf.licktime)[valid_ind]
        for i, arr in enumerate(plot_data):
            if arr:
                grid[i, int(arr)] = 1


        c = axs[ip].imshow(grid, cmap=attnlickcmap, vmin=-1, vmax=1,
                           aspect='auto', interpolation='none')

        axs[ip].set_xticks([0, 50, 100])
        axs[ip].set_xlim(0, 100)
        axs[ip].set_ylim(0, 300)
        axs[ip].set_yticks([])
        axs[ip].spines['left'].set_visible(False)
        axs[ip].set_ylabel(f'{ip} p')
    rasterfig.suptitle(f'{food_reward_idx}, food')
    rasterfig.show()

# --- Cell 5 ---
# total att time
for food_reward_idx in np.arange(len(food_reward_list)):
    fig= plt.figure()
    for ip, p1p2 in enumerate(plist):
        subdf=df[(df.trial_food_reward_idx==food_reward_idx) &(df.ip==ip)]
        sns.kdeplot([len(a) for a in subdf.at_times],  label=f'food reward={food_reward_list[food_reward_idx]:.0f}, p={p1p2}', bw_adjust=2, common_norm=False, color=pcmap(
            np.linspace(0, 1, len(plist)))[ip])


    color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap=pcmap), ax=plt.gca())
    ticks = [0, 0.5, 1]
    tick_labels = ['Low p1', '', 'High p1']
    color_bar.set_ticks(ticks)
    color_bar.set_ticklabels(tick_labels)


    plt.xlabel('number of high attentions')
    plt.ylabel('prob')
    plt.title(f'total attention times, varying p')
    # plt.ylim(0,1500)
    # plt.legend()

    centerax(plt.gca())
    plt.yticks([])
    plt.xlim(0, 20)
    # plt.xticks([0, 15])
    quicksave('total att time', 'plot sigmoid', fig=fig)
    pass


# --- Cell 6 ---
# autocorr df. vary reward, p
for food_reward_idx in range(len(food_reward_list)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    for ip, p1p2 in enumerate(plist):
        subdf=df[(df.trial_food_reward_idx==food_reward_idx) & (df.ip==ip)]
        if len(subdf)==0: 
            print(food_reward_idx, ip, 'skipped')
            continue
        att_seq_list = np.array(subdf.at_seq)
        # print(len(subdf), food_reward_idx, ip)
        autocorr, total_counts = empirical_autocorrelation_new(
            att_seq_list, min_no_samples=1111)
        # print(att_seq_list)
        xs = np.arange(0, len(autocorr), 1)
        xs = xs-len(xs)//2
        ax1.plot(xs, autocorr,
                 label=f'p:{p1p2}', color=pcmap(np.linspace(0, 1, len(plist)))[ip])
        # trial len dist
        sns.kdeplot([len(a) for a in att_seq_list],
                    fill=False, bw_adjust=0.5, ax=ax2, color=pcmap(np.linspace(0, 1, len(plist)))[ip])
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

    color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap=pcmap), ax=ax2)
    ticks = [0, 0.5, 1]
    tick_labels = [f'Low p={plist[0]}', '', f'High p={plist[-1]}']
    color_bar.set_ticks(ticks)
    color_bar.set_ticklabels(tick_labels)



    fig.suptitle(
        f'vary p1, p2={p1p2[1]} food \nreward={food_reward_list[food_reward_idx]:.0f}')
    fig.tight_layout()
    pass

# --- Cell 7 ---
# att size and gap
for food_reward_idx, reward in enumerate(food_reward_list):
    fig, axs = plt.subplots(len(plist), 1, figsize=(4, 4), sharex=True)
    for ip, p1p2 in enumerate(plist):
        subdf=df[(df.trial_food_reward_idx==food_reward_idx) &(df.ip==ip)]
        gaps=subdf.at_times.apply(find_gaps).to_list()
        gaps=np.concatenate(gaps,)
        axs[len(plist)-ip-1].hist(gaps, bins=np.arange(33), color=pcmap(np.linspace(0, 1, len(plist))[ip]), density=True, alpha=1)
        # sns.kdeplot(gaps, color=pcmap(np.linspace(0, 1, len(plist))[ip]),bw_adjust=2, common_norm=False) 
    plt.suptitle(f'r:{reward}')
    axs[ip].set_xlabel('gap size')
    plt.ylabel('prob')
fig, axs = plt.subplots(1,1, figsize=(1,3))
color_bar = plt.colorbar(plt.cm.ScalarMappable(cmap=pcmap), ax=plt.gca())
ticks = [0, 0.5, 1]
tick_labels = ['Low p1', '', 'High p1']
color_bar.set_ticks(ticks)
color_bar.set_ticklabels(tick_labels)
pass

# --- Cell 8 ---
# imshow df. vary reward, p2
vmin,vmax=0,50
nbin=20
for food_reward_idx in range(len(food_reward_list)):
    for ip, p1p2 in enumerate(plist):
        grid=np.zeros((vmax, nbin))
        bins=np.linspace(0,1,nbin)
        bindata=[[] for _ in range(nbin)] 
        subdf=df[(df.trial_food_reward_idx==food_reward_idx)&(df.ip==ip)]
        sb, nextatt=[],[]
        for alist, b in zip(subdf.at_times, subdf.beliefs):
            try:
                a,b=trialnextatt(alist,b),trialsb(b)  
                for j in range(len(b)):
                    thisa,thisb=a[j], b[j]
                    for i, (s,e) in enumerate(zip(bins, bins[1:])):
                        if s<thisb<=e:
                            bindata[i].append(thisa)
            except:
                continue
        for i in range(nbin):
            count=Counter(bindata[i])
            col=[]
            for j in range(vmin, vmax):
                col.append(count[j])
            col=np.array(col)
            col=col/np.sum(col)
            grid[:,i]=col
        fig=plt.figure()
        # grid=np.log(grid)
        c=plt.imshow(grid,aspect='auto', origin='lower')
        plt.colorbar(c)
        plt.xlabel('prob')
        plt.ylabel('gap')
        plt.title(f'reward id: {food_reward_idx}, \np:{p1p2},\nnode:{node}')
        plt.xticks([0,nbin],[0,1])
        # quicksave(f'vary node np, reward id {food_reward_idx}, node {node}', 'test', fig=fig)
        pass
        plt.close()


print("=== Done: notebooks/vary p2.ipynb ===")
