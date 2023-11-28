import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')

from irc.manager import IRCManager

defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

from auditoryforage.AF_env import AuditoryForaging

#############

from auditoryforage.utils import save_pickle_file, open_pickle_file

folder_name = './store/filters/'
file_name = folder_name + 'episodes_repo.pkl'

if os.path.exists(file_name):
    episode_repo = open_pickle_file(file_name)
else:
    print('run gnerate_episosdes_repo.py file first.')

#############

env_param_list = []
agent_list = []
env_list = []
num_epochs = 100


for env_param_tuple, seed in episode_repo.keys():
    env_param = list(env_param_tuple)
    env_param_list.append(env_param)
    agent_list.append(manager.train_agent(env_param, num_epochs=num_epochs, seed = seed))
    env_list.append(AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}}))

#############

from auditoryforage.filters.PF_agent import ParticleFilter

end_index = None
no_particles_list = [10000]
sampling_freq_list = [1]

env = env_list[0] #could be anything since env params are same for all candidates
particle_filter = ParticleFilter(agent_list, env_param_list, env)

PF_likelihood_repo = {}

for key in episode_repo.keys():
    correct_env_index = env_param_list.index(list(key[0]))
    # Note: key is gonna be ((tuple(env_param), seed), index_corresp_to_env_param)
    new_key = tuple([key, correct_env_index])
    PF_likelihood_repo[new_key] = []
    for episode in episode_repo[key]: 
        PF_IO = particle_filter.multiple_filtering(episode, no_particles_list, sampling_freq_list, end_index, req_output_posterior = True, do_save = False)
        PF_posterior = PF_IO['output']['sorted_results']['agent']['PF_posterior_ranked']
        agent_ranked = PF_IO['output']['sorted_results']['agent']['agent_ranked']
        PF_posterior_full = [0] * len(env_param_list)
        for ind in range(len(agent_ranked)):
            PF_posterior_full[agent_ranked[ind]] = PF_posterior[ind]
        PF_likelihood_repo[new_key].append(PF_posterior_full)


folder_name = './store/filters/'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
file_name = folder_name + 'PF_likelihood_repo.pkl'
save_pickle_file(PF_likelihood_repo, file_name)
        

#############