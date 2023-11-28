import os, sys, copy
sys.path.append(f'{os.getcwd()}/irc_gym')

from irc.manager import IRCManager

########################################################################

defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

########################################################################

from auditoryforage.AF_env import AuditoryForaging

########################################################################

food_reward = 350.0 #CHANGE

coeff = .08 #CHANGE
temp = .25 #CHANGE

env_param = [-3.0, food_reward, coeff, temp, -5000, 0] #CHANGE
file_name = './store/exp_data/exp_data.pkl' #CHANGE
reward_level = 'high' #CHANGE

########################################################################

env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})

########################################################################

env_param_list = []

env_param_list.append([-3.0, 50.0, coeff, temp, -5000, 0]) #CHANGE
env_param_list.append([-3.0, 350.0, coeff, temp, -5000, 0]) #CHANGE
params_folder_name = f'high_fr_{food_reward}_ac_{coeff}_{temp}/' #CHANGE

num_epochs = 400 #CHANGE
num_epochs_list = [num_epochs] * len(env_param_list)

agent_list = []
for ind in range(len(env_param_list)):
    agent_list.append(manager.train_agent(env_param_list[ind], num_epochs=num_epochs_list[ind]))

########################################################################

from auditoryforage.utils import open_pickle_file, save_pickle_file
from auditoryforage.filters.PF_agent import ParticleFilter
import numpy as np

# end_index_max = None #CHANGE
# no_particles_list = [30000] #CHANGE
end_index_max = 100 #CHANGE
no_particles = 10000 #CHANGE

sampling_freq = 1

total_time = 0
data_IRC = open_pickle_file(file_name)
episodes_collection = data_IRC[reward_level + '_reward_blocks']['episodes']
no_episodes = len(episodes_collection)
trunc_episode_wise_comparison = {}
model_comparison = {}
model_comparison['input'] = {}
model_comparison['input']['input_env_params'] = {}
model_comparison['output'] = {}
model_comparison['output']['weighted_avg_posterior'] = {}
model_comparison['input'][reward_level + '_reward_data'] = episodes_collection
for ind in range(len(env_param_list)):
    model_comparison['input']['input_env_params'][ind] = env_param_list[ind]
    model_comparison['output']['weighted_avg_posterior'][ind] = 0


for episode_no in range(no_episodes):
    trunc_episode_wise_comparison[episode_no] = {}
    no_trials = len(episodes_collection[episode_no])
    for trial_no in range(no_trials):
        trunc_episode_wise_comparison[episode_no][trial_no] = {}
        episode = episodes_collection[episode_no][trial_no]
        end_index = min(len(episode['states']), end_index_max) if end_index_max is not None else None
        folder_name = params_folder_name + f'en_{episode_no}_tn_{trial_no}/' #CHANGE
        particle_filter = ParticleFilter(agent_list, env_param_list, env, true_params_folder_name = folder_name, verbose = True)
        PF_IO = particle_filter.generate_wrt_reference_episode(episode, no_particles, sampling_freq, end_index, do_save = True, req_output_posterior = True)
        agent_posterior_dict = copy.deepcopy(PF_IO['output']['sorted_results']['agent'])
        trunc_episode_wise_comparison[episode_no][trial_no]['agent'] = copy.deepcopy(agent_posterior_dict)
        trunc_episode_wise_comparison[episode_no][trial_no]['end_index'] = end_index
        total_time += end_index
        for ind in range(len(agent_posterior_dict['agent_ranked'])):
            model_comparison['output']['weighted_avg_posterior'][agent_posterior_dict['agent_ranked'][ind]] += end_index * agent_posterior_dict['PF_posterior_ranked'][ind]
model_comparison['output']['trunc_episode_wise'] = trunc_episode_wise_comparison

for model_ind, unnormalized_prob in model_comparison['output']['weighted_avg_posterior'].items():
    model_comparison['output']['weighted_avg_posterior'][model_ind] = unnormalized_prob/total_time

file_name = './store/filters/PF_agent/' + params_folder_name + f'avg_posterior_np_{no_particles}_sf_{sampling_freq}_eim_{end_index_max}_rn_{np.random.randint(0,100)}.pkl'
save_pickle_file(model_comparison, file_name)

########################################################################