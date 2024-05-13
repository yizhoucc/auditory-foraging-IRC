import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')
from auditoryforage.AF_env import AuditoryForaging as AF
from auditoryforage.AF_env import AuditoryForagingReward as AFR
from stable_baselines3 import PPO
from irc_gym.irc.model import FuncBeliefModel


food_reward_list = [1000, 1100, 1200, 1300, 1400, 1500]
att_coeff = 0.087213
att_temp = 0.25
penalty_cost = -30
time_in_game_reward = 0.
num_epochs = 44
seed_value = 1

# for food_reward in food_reward_list:
#     task=AF()
#     task.food_reward=food_reward
#     model = PPO('MlpPolicy', task, verbose=1)
#     model.learn(total_timesteps=100)

task=AF()
task.food_reward=1000
task.attention_cost_coeff=att_coeff
task.attention_cost_temp=att_temp
task.penalty_cost=penalty_cost
task.time_in_game_reward=time_in_game_reward
task.food_reward_list=food_reward_list
taskbelief=FuncBeliefModel(env=task, rng=1)
model = PPO('MlpPolicy', taskbelief, verbose=1, device='cpu')
model.learn(total_timesteps=100000)

# from gym.spaces import Discrete, MultiDiscrete, Box
# MultiDiscrete([4,5])
# MultiDiscrete([4])








# from irc.manager import IRCManager
# food_reward=1100
# # ##############################################################################

# defaults = {
#     'agent.env._target_': 'auditoryforage.AF_env.AuditoryForagingReward',
#     # 'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
#     'agent.model._target_': 'irc.model.FuncBeliefModel',
# }
# manager = IRCManager(defaults=defaults)

# ##############################################################################

# env_param = [0, food_reward, att_coeff, att_temp, penalty_cost, 0, time_in_game_reward]
# print("########################################################################")
# print(f"\n \n Training going on for {num_epochs} epochs\n \n ")
# print("########################################################################")
# agent = manager.train_agent(env_param, num_epochs=num_epochs, seed = seed_value)