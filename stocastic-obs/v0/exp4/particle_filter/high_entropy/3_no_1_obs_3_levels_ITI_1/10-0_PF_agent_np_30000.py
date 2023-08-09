import os
import sys
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

env_param = [-3.0, 200.0, 13, 5, -5000, 0] #CHANGE
true_params_folder_name = '1-13_5/' #CHANGE
file_name = './auditoryforage/filters/ref_episode_13_5.pkl' #CHANGE

########################################################################

env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})

########################################################################

from auditoryforage.utils import open_pickle_file
episode = open_pickle_file(file_name)

########################################################################

env_param_list = []

env_param_list.append([-3.0, 200.0, 13, 5, -5000, 0]) #CHANGE
env_param_list.append([-3.0, 200.0, .06, .25, -5000, 0]) #CHANGE
env_param_list.append([-3.0, 200.0, .08, .25, -5000, 0]) #CHANGE

num_epochs = 400
num_epochs_list = [num_epochs] * len(env_param_list)

agent_list = []
for ind in range(len(env_param_list)):
    agent_list.append(manager.train_agent(env_param_list[ind], num_epochs=num_epochs_list[ind]))

########################################################################

from auditoryforage.filters.PF_agent import ParticleFilter

end_index = None #CHANGE
no_particles_list = [30000] #CHANGE

sampling_freq_list = [1]
particle_filter = ParticleFilter(agent_list, env_param_list, env, true_params_folder_name, verbose = True)
particle_filter.multiple_filtering(episode, no_particles_list, sampling_freq_list, end_index, req_output_posterior = True)

########################################################################
