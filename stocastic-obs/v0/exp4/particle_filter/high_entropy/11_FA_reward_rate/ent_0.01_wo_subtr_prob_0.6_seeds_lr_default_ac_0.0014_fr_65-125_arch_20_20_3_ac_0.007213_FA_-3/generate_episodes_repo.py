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

folder_name = './store/comparison/'
file_name = folder_name + 'diff_food_rewards.pkl'
detection_measures = open_pickle_file(file_name)
food_reward_list = detection_measures['input']['food_reward_list']
max_ind_list = [count.index(max(count)) for count in detection_measures['output']['hit_count_list']]
episode_repo = {}

############

attention_cost_coeff = 0.149215543
attention_cost_temp = .25
food_reward_list = [10.0, 100.0]

food_reward_index = 1

env_param_list = []
for food_val in food_reward_list:
    env_param_list.append([0, food_val, attention_cost_coeff, attention_cost_temp, 0, 0])
    
num_epochs = 100
num_episodes_per_case = 4

for env_param in env_param_list:

    if env_param[food_reward_index] not in food_reward_list:
        print('Error here')
        break
    else:
        seed = max_ind_list[food_reward_list.index(env_param[food_reward_index])]

    # Note: key is gonna be (tuple(env_param), seed)        
    key = tuple([tuple(env_param), seed])

    print(key)

    if key not in episode_repo.keys():
        episode_repo[key] = []
    
    env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})
    agent = manager.train_agent(env_param, num_epochs=num_epochs, seed = seed)
    for _ in range(num_episodes_per_case):
        episode_repo[key].append(agent.run_one_episode(env=env, num_steps=10000, q_states = [[i] for i in range(env.no_nodes)]))            

############

folder_name = './store/filters/'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

file_name = folder_name + 'episodes_repo.pkl'

# need to fix this later, not high priority
# if os.path.exists(file_name):
#     episode_repo_old = open_pickle_file(file_name)
#     for env_param, data in episode_repo_old.items():
#         if env_param not in episode_repo.keys():
#             episode_repo[env_param] = data

save_pickle_file(episode_repo, file_name)