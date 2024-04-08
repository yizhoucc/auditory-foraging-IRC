import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')
from auditoryforage.AF_env import AuditoryForaging as AF

from stable_baselines3 import PPO

food_reward_list = [1000, 1100, 1200, 1300, 1400, 1500]
att_coeff = 0.087213
att_temp = 0.25
penalty_cost = -30
time_in_game_reward = 0.
num_epochs = 11
seed_value = 1

for food_reward in food_reward_list:
    task=AF()
    task.food_reward=food_reward
    model = PPO('MlpPolicy', task, verbose=1)
    model.learn(total_timesteps=100)
