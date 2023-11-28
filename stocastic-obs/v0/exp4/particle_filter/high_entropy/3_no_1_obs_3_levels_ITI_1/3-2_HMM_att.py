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

########################################################################

defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

########################################################################

env_param = [-3.0, 200.0, .08, .25, -5000, 0]

# change back to old no. of epochs
num_epochs = 400

agent = manager.train_agent(env_param, num_epochs=num_epochs)

########################################################################

from auditoryforage.AF_env import AuditoryForaging
from auditoryforage.utils import plot_AF_episode

# Change below line if you want to try the trained model on a different set of environemnt.
# env_param = [-3.0, 8.0, 13, 8, -8, -4]

env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})

########################################################################

from auditoryforage.utils import open_pickle_file
file_name = './auditoryforage/filters/ref_episode_0.08_0.25.pkl'
episode = open_pickle_file(file_name)

########################################################################

from auditoryforage.filters.HMM_attention import HMM
end_index = 5
HMM_IO = HMM(agent, env, episode, end_index).compute_posterior_across_attention_series()