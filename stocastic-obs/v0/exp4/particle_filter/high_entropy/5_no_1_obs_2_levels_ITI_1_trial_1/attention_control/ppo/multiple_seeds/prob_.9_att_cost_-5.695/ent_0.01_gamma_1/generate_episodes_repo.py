import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')

from irc.manager import IRCManager
from auditoryforage.AF_env import AuditoryForaging
from auditoryforage.utils import save_pickle_file, open_pickle_file

############

defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

############

episode_repo = {}

############

env_param_list = [[0, 10.0, .16, .25, 0, 0], [0, 100.0, .16, .25, 0, 0]]
seed_list = [0, 1, 2, 3, 4]
num_epochs = 100
num_episodes_per_case = 2

for env_param in env_param_list:
    if tuple(env_param) not in episode_repo.keys():
        episode_repo[tuple(env_param)] = {}
    env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})
    for seed in seed_list:
        if episode_repo[tuple(env_param)][seed] not in episode_repo.keys():
            episode_repo[tuple(env_param)][seed] = []
        agent = manager.train_agent(env_param, num_epochs=num_epochs, seed = seed)
        for _ in range(num_episodes_per_case):
            episode_repo[tuple(env_param)][seed].append(agent.run_one_episode(env=env, num_steps=10000, q_states = [[i] for i in range(env.no_nodes)]))            

############

folder_name = './store/filters/'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_name = folder_name + 'episodes_repo.pkl'

if os.path.exists(file_name):
    episode_repo_old = open_pickle_file(file_name)
    for env_param, data in episode_repo_old.items():
        if env_param not in episode_repo.keys():
            episode_repo[env_param] = data

save_pickle_file(episode_repo, file_name)