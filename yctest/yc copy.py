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

print("=== Starting: yctest/yc copy.ipynb ===")

# --- Cell 0 ---
import os
import sys
sys.path.append(os.path.join(REPO, "irc_gym"))
from auditoryforage.AF_env import AuditoryForaging as AF
from auditoryforage.AF_env import AuditoryForagingReward as AFR
from stable_baselines3 import PPO
from multiprocessing import Pool
import multiprocessing as mp
from matplotlib import pyplot as plt
import numpy as np


# --- Cell 1 ---
food_reward_list = [1000, 1100, 1200, 1300, 1400]
att_coeff = 0.087213 #0.087213
att_temp = 0.25 # 0.25
penalty_cost = -30 #  -30
time_in_game_reward = 0. # old 0
# num_epochs = 11
# seed_value = 1

# --- Cell 2 ---

model_list={}
for food_reward in food_reward_list:
    # init
    task=AF()
    # assign values
    task.food_reward=food_reward
    task.time_in_game_reward=time_in_game_reward
    task.attention_cost_coeff=att_coeff
    task.attention_cost_temp=att_temp
    task.penalty_cost=penalty_cost
    # train
    model = PPO('MlpPolicy', task, verbose=1)
    model.learn(total_timesteps=9999/25*3600*1)
    # save
    model_list[food_reward]=model
    model.save(f'./ycstore/{food_reward}')


# --- Cell 3 ---


def performance_reward_rate(model, task, total_trials=100 ):
    '''model reward rate'''
    history_r=[]
    for _ in range(total_trials):
        t=0
        total_r=0
        x=task.reset()
        done=False
        while not done: 
            # model.predict(x)
            a,_=model.predict(x)
            x, r, done, _ = task.step(a)
            t+=1
            total_r+=r
        history_r.append((total_r/t))
    return  history_r,
        

# --- Cell 4 ---
from collections import defaultdict
performances=defaultdict(list)
model_list={}
for food_reward in food_reward_list:
    # init
    task=AF()
    # assign values
    task.food_reward=food_reward
    task.time_in_game_reward=time_in_game_reward
    task.att_coeff=att_coeff
    task.att_temp=att_temp
    task.penalty_cost=penalty_cost
    # load
    model = PPO.load(f'ycstore/{food_reward}', env=task, device='cpu')
    model_list[food_reward]=model
    # eval
    # performances['rewardrate'].append(performance_reward_rate(model, task,total_trials=10000)[0])


# --- Cell 5 ---
# import numpy as np
# from matplotlib import pyplot as plt
# means=np.mean(np.array(performances['rewardrate']), axis=1)
# stds=np.std(np.array(performances['rewardrate']), axis=1)*0.1
# plt.fill_between(list(range(len(means))), means - stds, means + stds, color='skyblue', alpha=0.5, label='± 1 std')
# plt.plot(list(range(len(means))), means)
# plt.xlabel('food_reward')

# --- Cell 6 ---
import numpy as np
from scipy.stats import poisson
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


data=np.array(performance_reward_rate(model, task))

# Define the Poisson probability mass function (PMF)
def poisson_pmf(x, mu):
    return poisson.pmf(x, mu)

x_values = np.arange(0, np.max(data) + 1)
hist_values, bin_edges = np.histogram(data, bins=np.arange(0, np.max(data) + 2))
mu_fit, _ = curve_fit(poisson_pmf, x_values, hist_values, p0=[5])

# Visualize the original data and the fitted Poisson distribution
plt.hist(data, bins=np.arange(0, np.max(data) + 1), density=True, alpha=0.6, label='Data')
x = np.arange(0, np.max(data) + 1)
plt.plot(x, poisson_pmf(x, mu_fit), 'r-', lw=2, label='Fitted Poisson distribution')
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Fit data to Poisson distribution')
plt.legend()
pass

print("Estimated Poisson distribution parameter (mean):", mu_fit[0])

# --- Cell 7 ---

def par(fun, paramls):
    with Pool() as pool:
        results = pool.map(fun, paramls)
    return results

def eval_agent(food_reward):
    # init
    task=AF()
    # assign values
    task.food_reward=food_reward
    task.time_in_game_reward=time_in_game_reward
    task.att_coeff=att_coeff
    task.att_temp=att_temp
    task.penalty_cost=penalty_cost
    # load
    model = PPO.load(f'ycstore/{food_reward}', env=task, device='cpu')
    model_list[food_reward]=model
    # eval
    return (performance_reward_rate(model, task,total_trials=100)[0])

with Pool() as pool:
    res = pool.map(eval_agent, food_reward_list)


# res=par(eval_agent, food_reward_list)
res

# --- Cell 8 ---
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv,SubprocVecEnv

env_list = [AF for _ in range(4)]
vec_env = SubprocVecEnv(env_list)
obs = vec_env.reset()
inference_data = []
for _ in range(1000): 
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = vec_env.step(action)
        # Collect inference data (e.g., observations, actions, rewards)
        inference_data.append((obs, action, reward, done))
        if done.any():
            obs = vec_env.reset()
            break
vec_env.close()

print("=== Done: yctest/yc copy.ipynb ===")
