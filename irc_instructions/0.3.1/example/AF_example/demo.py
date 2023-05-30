import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 15, 'lines.linewidth': 2,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 1200,
})

import os
import sys
sys.path.append(f'{os.getcwd()}/irc_package')

from irc.manager import AgentManager

from auditoryforage.utils import plot_AF_episode

########################################################

# from irc.manager import AgentManager
# defaults = {'env._target_': 'irc.examples.IdenticalBoxesEnv'}
# manager = AgentManager(defaults=defaults)

defaults = {
    'env._target_': 'auditoryforage.AF_env.AuditoryForaging'
}
manager = AgentManager(defaults=defaults)

########################################################

# agent, key = manager.train_agent() # using default environment parameter

# env_param = (0.2, 0.05, 0.8, 0.1, 10., -1.)
# agent, key = manager.train_agent(env_param=env_param, num_epochs=10)

env_param = [-3.0, 200.0, .08, .25, -5000, 0]
num_epochs = 110
agent, key = manager.train_agent(env_param = env_param, num_epochs=num_epochs)
# agent, fig = manager.inspect_agent(key, figsize=(5, 2.5)) #removed for screen

########################################################

# agent, fig = manager.inspect_agent(key) #removed for screen

########################################################

# if `env` is not specified, agent interacts with its assumed environment
episode = agent.run_one_episode()

# run the agent in an environment different from assumption
# env = IdenticalBoxesEnv()
from auditoryforage.AF_env import AuditoryForaging
env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})
episode = agent.run_one_episode(env=env, max_steps=500)
# fig = plot_AF_episode(episode, env, agent)  #removed for screen

# env.spec  #removed for screen

########################################################

# param_grid = [
#     [0.1, 0.2, 0.3],    # p_appear
#     [],                 # p_vanish
#     [0.6, 0.8],         # p_cue
#     [],                 # lambda_center
#     [10., 5.],          # r_food
#     [],                 # r_move
# ]
param_grid = [
    [-3.0], # lick_cost
    [16.0], # food_reward
    [20, 25, 30, 35, 40], # attention_cost_coeff
    [6, 10, 14, 18, 22], # attention_cost_temp
    [-50],
    [0]
]
seeds = range(2)
manager.train_agents_on_grid(param_grid=param_grid, seeds=seeds, num_epochs=20)