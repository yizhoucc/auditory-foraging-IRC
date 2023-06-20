# %matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 15, 'lines.linewidth': 2,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 1200,
})

import yaml
import numpy as np

import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')

from irc.manager import IRCManager

from auditoryforage.utils import plot_AF_episode

################################################################ 

defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

################################################################

env_param = [-3.0, 200.0, .08, .25, -5000, 0]
num_epochs = 369

agent = manager.train_agent(env_param, num_epochs=num_epochs)
agent, fig = manager.inspect_agent(env_param, figsize=(5, 2.5))

################################################################

from auditoryforage.AF_env import AuditoryForaging

# Change below line if you want to try the trained model on a different set of environemnt.
# env_param = [-3.0, 8.0, 13, 8, -8, -4]

env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})
episode = agent.run_one_episode(env=env, num_steps=10000, q_states = [[i] for i in range(env.no_nodes)])
# fig = plot_AF_episode(episode, env, agent, nodes_from_zero = 40, time_steps_before_lick = 20)

# env.spec


################################################################

from auditoryforage.utils import particle_filter
end_index = 200
no_particles = 500


state_list = [list_of_state[0] for list_of_state in episode['states']]
lick_actions = [env.dict_action_possible[action][0] for action in episode['actions']]

# state_list = state_list[:end_index]
# lick_actions = lick_actions[:end_index]

state_list = state_list
lick_actions = lick_actions

particle_filter_output = particle_filter(agent = agent, env = env, lick_actions = lick_actions, state_list = state_list, no_particles = no_particles, sampling_freq = 1)

################################################################

import pickle 

reference_episode = episode
particle_filter_compare_data = {'reference_episode': episode, "particle_filter_output": particle_filter_output}
file_name = "PF_IO_episodes.pkl"

open_file = open(file_name, "wb")
pickle.dump(particle_filter_compare_data, open_file)
open_file.close()

open_file = open(file_name, "rb")
particle_filter_compare_data_loaded = pickle.load(open_file)
open_file.close()

################################################################

generated_episode = particle_filter_compare_data['particle_filter_output']['generated_episodes'][0]

################################################################

# computing error rate
generated_attention = np.array([env.dict_action_possible[int(action)][1] for action in generated_episode['actions']])
actual_attention = np.array([env.dict_action_possible[int(action)][1] for action in episode['actions']])
error_rate = np.sum(np.abs(generated_attention-actual_attention))/len(generated_attention)
print(f'error rate is {error_rate}')

################################################################

# plotting generated episode
# generated_episode['rewards'] = np.zeros(len(generated_episode['actions']))
# fig = plot_AF_episode(generated_episode, env, agent, nodes_from_zero = 40, time_steps_before_lick = 20)

################################################################

# plotting likelihoods of particles
# plt.stem(particle_filter_compare_data['particle_filter_output']['particles_likelihoods'])
# plt.show()