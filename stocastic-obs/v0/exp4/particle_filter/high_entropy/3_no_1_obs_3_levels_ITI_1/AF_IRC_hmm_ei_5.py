import sys, os
sys.path.append(f'{os.getcwd()}/irc_gym')
from irc.manager import IRCManager

###################################

defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

###################################

env_param = [-3.0, 200.0, .08, .25, -5000, 0]
num_epochs = 400
agent = manager.train_agent(env_param, num_epochs=num_epochs)
agent, fig = manager.inspect_agent(env_param, figsize=(5, 2.5))

###################################

from auditoryforage.AF_env import AuditoryForaging
from auditoryforage.utils import open_pickle_file
env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})
PF_file_name = 'store/particle_filter/PF_np_100000_sf_1_56.pkl'
temp_particle_filter_IO = open_pickle_file(PF_file_name)
episode = temp_particle_filter_IO['input']['root_episode']

###################################

end_index = 5

###################################

# Running HMM.

from auditoryforage.attention_posterior import AttentionPosterior
attention_posterior_IO = AttentionPosterior(agent, env, episode, end_index).compute_posterior_across_attention_series()
