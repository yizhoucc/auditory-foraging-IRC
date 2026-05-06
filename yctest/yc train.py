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

print("=== Starting: yctest/yc train.ipynb ===")

# --- Cell 0 ---
import sys
import os
sys.path.append(os.path.join(REPO, "irc_gym"))

# --- Cell 1 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from auditoryforage.AF_env import AuditoryForagingReward2 as AFR2
from stable_baselines3 import PPO
from irc_gym.irc.model import FuncBeliefModel
from multiprocessing import Pool
import multiprocessing as mp
import multiprocessing
from ult import *

# --- Cell 2 ---
import wandb
from wandb.integration.sb3 import WandbCallback


# --- Cell 3 ---
# template for training
print('''
    date 5.4
    goal:
      nice periodic,
    new: 
      log scale food reward list
      67 instead of 56.
    comments:
      not great, attention too little and lick too much, since no punishment
''')
# training hyper params
epoch_size = 11111
eval_size = 999
n_epoch = 77
n_seed=1

# reward list
vmin, vmax=np.log(10),np.log(10000)
food_reward_list = np.linspace(vmin, vmax, 11)
food_reward_list=np.exp(food_reward_list)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [0, 1500, 0.087213, .25, -30, 0, 0]
#
modelname='date_4_67_log'

# --- Cell 4 ---
# template for training
print('''
    date 5.5
    goal:
      nice periodic,
    new: 
      log scale food reward list
      56. 
    comments:
      try 56 first.
      log scale is not working well. agent not doing anything. maybe too many small rewards.
''')
# training hyper params
epoch_size = 11111
eval_size = 999
n_epoch = 66
n_seed=1

# reward list
vmin, vmax=np.log(10),np.log(10000)
food_reward_list = np.linspace(vmin, vmax, 11)
food_reward_list=np.exp(food_reward_list)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [0, 1500, 0.087213, .25, -30, 0, 0]
#
modelname='date_5_56'

# --- Cell 5 ---
# template for training
print('''
    date 5.5
    goal:
      nice periodic,
    new: 
        smaller range log, 56
    comments:
''')
# training hyper params
epoch_size = 15000
eval_size = 700
n_epoch = 40
n_seed=1

# reward list
vmin, vmax=np.log(50),np.log(6000)
food_reward_list = np.linspace(vmin, vmax, 9)
food_reward_list=np.exp(food_reward_list)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [0, 1500, 0.087213, .25, -30, 0, 0]
#
modelname='date_5_56_small'

# --- Cell 6 ---
# template for training
print('''
    date 5.5
    goal:
      nice periodic,
    new: 
        linear 56
    comments:
''')
# training hyper params
epoch_size = 15000
eval_size = 700
n_epoch = 50
n_seed=1

# reward list
vmin, vmax=np.log(10),np.log(6000)
food_reward_list = np.linspace(vmin, vmax, 9)
food_reward_list=np.exp(food_reward_list)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-10, 1500, 0.087213, .45, -30, 0, 0]
#
modelname='date_5_56_log'

# --- Cell 7 ---
# template for training
print('''
    date 5.6
    goal:
      nice periodic,
    new: 
        linear 56
    comments:
      10 - 3000 shows a nice change
''')
# training hyper params
epoch_size = 15000
eval_size = 700
n_epoch = 60
n_seed=1

# reward list
vmin, vmax=(10),(6000)
food_reward_list = np.linspace(vmin, vmax, 9)
# food_reward_list=np.exp(food_reward_list)
print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-10, 1500, 0.087213, .45, -30, 0, 0]
#
modelname='date_6_56_linear'

# --- Cell 8 ---
# template for training
print('''
    date 5.6
    goal:
      nice periodic,
    new: 
        linear 56
        use the env param that generate good peroid the first time
    comments:

''')
# training hyper params
epoch_size = 15000
eval_size = 700
n_epoch = 80
n_seed=1
p1p2=np.array([0.5,0.6])
p1p2=None

# reward list
vmin, vmax=10,10000
food_reward_list = np.linspace(vmin, vmax, 5)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-10, 1500, 0.087213, .25, -50, -10, 0]
#
modelname='date_6_56_linear_2'

# --- Cell 9 ---
# template for training
print('''
    date 5.6
    goal:
      nice periodic,
    new: 
        linear 56
        use the env param that generate good peroid the first time
    comments:
    some nice gradient plots. some opposite result for the attn vs sig prob plot. coudl because the cost of attn > lick?
''')
# training hyper params
epoch_size = 15000
eval_size = 700
n_epoch = 80
n_seed=1
p1p2=np.array([0.5,0.6])
p1p2=None

# reward list
vmin, vmax=100,10000
food_reward_list = np.linspace(vmin, vmax, 5)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-10, 1500, 0.087213, .1, -50, 0, 0]
#
modelname='date_6_56_linear_small'

# --- Cell 10 ---
# template for training
print('''
    date 5.6
    goal:
      nice periodic,
    new: 
        linear 56
        use the env param that generate good peroid the first time
    comments:
    some nice gradient plots. some opposite result for the attn vs sig prob plot. coudl because the cost of attn > lick?
''')
# training hyper params
epoch_size = 15000
eval_size = 700
n_epoch = 80
n_seed=1
p1p2=np.array([.6,.7])
p1p2=None

# reward list
vmin, vmax=100,10000
food_reward_list = np.linspace(vmin, vmax, 5)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-10, 1500, 0.087213, .1, -50, 0, 0]
#
modelname='date_6_67_linear_small'

# --- Cell 11 ---
# template for training
print('''
    date 5.6
    goal:
      nice periodic,
    new: 
        linear 56
        use the env param that generate good peroid the first time
    comments:
    some nice gradient plots. some opposite result for the attn vs sig prob plot. coudl because the cost of attn > lick?
''')
# training hyper params
epoch_size = 15000
eval_size = 700
n_epoch = 80
n_seed=1
p1p2=np.array([.5,.7])
p1p2=None

# reward list
vmin, vmax=10,10000
food_reward_list = np.linspace(vmin, vmax, 10)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-10, 1500, 0.087213, .25, -50, 0, 0]
#
modelname='date_7_largerange'

# --- Cell 12 ---
# play with attn cost
# task = AFR2(spec={'agent': {'lick_cost': env_param[0],
#                             'food_reward': env_param[1],
#                             'attention_cost_coeff':.01,
#                             'attention_cost_temp': 0.25, # attn cost, related to unceratinty.
#                             'penalty_cost': env_param[4], # fa
#                             'iti_cost': env_param[5],
#                                 'time_in_game_reward': env_param[6]}})
# task.food_reward_list = food_reward_list

# taskbelief = FuncBeliefModel(env=task, rng=1)
# run_one_episode(task, taskbelief, model)
# task.attention_cost, task.penalty_cost, task.iti_cost, task.time_in_game_reward

# --- Cell 13 ---
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
    if p1p2:
        task.obs_certainity_possible=p1p2
    
    # init wandb log
    run = wandb.init(
        project="lokesh_rl",
        save_code=True,  # optional
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs",)

    # train
    for i in range(n_epoch):
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=f'seed_{seed}',
                    reset_num_timesteps=False,
                    callback=WandbCallback(
                        gradient_save_freq=100,
                        verbose=2,
                        model_save_path=f"models/{run.id}",
                    ),)
        model.save(f'ycstore/seed_{seed}_{modelname}')

        # run mp job for eval
        def eval_agent(a):
            t = 0
            total_r = 0  # trial total reward
            x = taskbelief.reset()
            done = False
            while not done:
                # model.predict(x)
                a, _ = model.predict(x, deterministic=False)
                x, r, done, _ = taskbelief.step(a)
                # print(x, r,task.food_reward)
                t += 1
                total_r += r  # updating trial total reward
            return taskbelief.env.food_reward, total_r

        with multiprocessing.Pool(processes=27) as pool:
            results = pool.map(eval_agent, range(eval_size))

        # process mp res
        history_r = defaultdict(list)
        for food_reward, total_r in results:
            history_r[food_reward].append(total_r)

        # plot, normalized
        for k in sorted(history_r.keys()):
            v = history_r[k]
            v = (v - min(v)) / (max(v) - min(v))
            v = 1-v
            sns.kdeplot(v, fill=False, label=f'{k:.0f}', bw_adjust=0.5,common_norm=False)
        plt.legend()
        plt.title(f'epoch {i}')
        pass

        # plot, not normalized
        for k in sorted(history_r.keys()):
            v = history_r[k]
            sns.kdeplot(v, fill=False, label=f'{k:.0f}', bw_adjust=0.5,common_norm=False)
        plt.legend()
        plt.title(f'epoch {i}')
        pass
    run.finish()
    notify(f'training {modelname} finished')

# --- Cell 14 ---
# template for training
print('''

''')
# training hyper params
epoch_size = 15000
eval_size = 700
n_epoch = 99
n_seed=1
p1p2=np.array([.6,.7])
p1p2=None

# reward list
vmin, vmax=10,5000
food_reward_list = np.linspace(vmin, vmax, 5)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-10, 1500, 0.087213, .1, -50, 0, 0]
modelname='date_7_67_lickcost'



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
    if p1p2:
        task.obs_certainity_possible=p1p2
    
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs",)

    # train
    for i in range(n_epoch):
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=f'seed_{seed}_{modelname}',
                    reset_num_timesteps=False,
                    )
        model.save(f'ycstore/seed_{seed}_{modelname}')

    notify(f'training {modelname} finished')

# --- Cell 15 ---
# template for training
print('''

''')
# training hyper paramsa
epoch_size = 15000
eval_size = 700
n_epoch = 99
n_seed=1
p1p2=np.array([.5,.6])
p1p2=None

# reward list
vmin, vmax=10,5000
food_reward_list = np.linspace(vmin, vmax, 9)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-10, 1500, 0.087213, .1, -50, 0, 0]
modelname='date_7_56_lickcost'



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
    if p1p2:
        task.obs_certainity_possible=p1p2
    
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs",)

    # train
    for i in range(n_epoch):
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=f'seed_{seed}_{modelname}',
                    reset_num_timesteps=False,
                    )
        model.save(f'ycstore/seed_{seed}_{modelname}')

    notify(f'training {modelname} finished')

# --- Cell 16 ---
# template for training
print('''

''')
# training hyper paramsa
epoch_size = 15000
eval_size = 700
n_epoch = 99
n_seed=1
p1p2=np.array([.5,.6])
p1p2=None

# reward list
vmin, vmax=10,5000
food_reward_list = np.linspace(vmin, vmax, 9)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-0, 1500, 0.087213, .1, -50, 0, 0]
modelname='date7_56'



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
    if p1p2:
        task.obs_certainity_possible=p1p2
    
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs",)

    # train
    for i in range(n_epoch):
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=f'seed_{seed}_{modelname}',
                    reset_num_timesteps=False,
                    )
        model.save(f'ycstore/seed_{seed}_{modelname}')

    notify(f'training {modelname} finished')

# --- Cell 17 ---
# template for training
print('''

''')
# training hyper paramsa
epoch_size = 15000
eval_size = 700
n_epoch = 99
n_seed=1
p1p2=np.array([.6,.7])
p1p2=None

# reward list
vmin, vmax=10,5000
food_reward_list = np.linspace(vmin, vmax, 9)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-0, 1500, 0.087213, .1, -50, 0, 0]
modelname='date7_67'




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
    if p1p2:
        task.obs_certainity_possible=p1p2
    
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs",)

    # train
    for i in range(n_epoch):
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=f'seed_{seed}_{modelname}',
                    reset_num_timesteps=False,
                    )
        model.save(f'ycstore/seed_{seed}_{modelname}')

    notify(f'training {modelname} finished')

# --- Cell 18 ---
# template for training
print('''

''')
# training hyper paramsa
epoch_size = 15000
eval_size = 700
n_epoch = 99
n_seed=1
p1p2=np.array([.6,.7])
p1p2=None

# reward list
vmin, vmax=10,5000
food_reward_list = np.linspace(vmin, vmax, 9)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-0, 1500, 0.087213, .1, -30, 0, 0]
modelname='date7_67_orginalpenalty'




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
    if p1p2:
        task.obs_certainity_possible=p1p2
    
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs",)

    # train
    for i in range(n_epoch):
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=f'seed_{seed}_{modelname}',
                    reset_num_timesteps=False,
                    )
        model.save(f'ycstore/seed_{seed}_{modelname}')

    notify(f'training {modelname} finished')

# --- Cell 19 ---
# template for training
print('''

''')
# training hyper paramsa
epoch_size = 15000
eval_size = 700
n_epoch = 99
n_seed=1
p1p2=np.array([.7,.8])
p1p2=None

# reward list
vmin, vmax=10,5000
food_reward_list = np.linspace(vmin, vmax, 9)
# food_reward_list=np.exp(food_reward_list)

print('food reard list: ', [f'{a:.0f}' for a in food_reward_list])

# other env params
env_param = [-0, 1500, 0.087213, .1, -30, 0, 0]
modelname='date7_67_orginalpenalty'

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
    if p1p2:
        task.obs_certainity_possible=p1p2
    
    model = PPO('MlpPolicy', taskbelief, verbose=0, device='cpu',
                clip_range=0.1, ent_coef=0.01, tensorboard_log=f"runs",)

    # train
    for i in range(n_epoch):
        model.learn(total_timesteps=epoch_size,
                    tb_log_name=f'seed_{seed}_{modelname}',
                    reset_num_timesteps=False,
                    )
        model.save(f'ycstore/seed_{seed}_{modelname}')

    notify(f'training {modelname} finished')

print("=== Done: yctest/yc train.ipynb ===")
