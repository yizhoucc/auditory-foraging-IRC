import pickle
import multiprocessing

################################################################

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

# change back to old no. of epochs
num_epochs = 400

agent = manager.train_agent(env_param, num_epochs=num_epochs)
agent, fig = manager.inspect_agent(env_param, figsize=(5, 2.5))

################################################################

from auditoryforage.AF_env import AuditoryForaging

# Change below line if you want to try the trained model on a different set of environemnt.
# env_param = [-3.0, 8.0, 13, 8, -8, -4]

env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})
env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})
# episode = agent.run_one_episode(env=env, num_steps=10000, q_states = [[i] for i in range(env.no_nodes)])
# fig = plot_AF_episode(episode, env, agent, nodes_from_zero = 40, time_steps_before_lick = 20)
# env.spec

################################################################

file_name = 'store/particle_filter/PF_np_100_sf_1_69.pkl'
open_file = open(file_name, "rb")
particle_filter_IO = pickle.load(open_file)
open_file.close()
episode = particle_filter_IO['input']['root_episode']

################################################################

from auditoryforage.particle_filter import ParticleFilter

end_index = 2500

no_particles_list = [100, 200, 500, 1000, 2000]
sampling_freq_list = [1, 10, 20, 50]

particle_filter = ParticleFilter(agent, env)
particle_filter_IO = particle_filter.multiple_filtering(episode, no_particles_list, sampling_freq_list, end_index)


