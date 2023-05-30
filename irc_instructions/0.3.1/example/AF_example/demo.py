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





# agent, key = manager.train_agent() # using default environment parameter

# env_param = (0.2, 0.05, 0.8, 0.1, 10., -1.)
# agent, key = manager.train_agent(env_param=env_param, num_epochs=10)


########################################################


env_param = [-3.0, 200.0, .08, .25, -5000, 0]
num_epochs = 110
agent, key = manager.train_agent(env_param = env_param, num_epochs=num_epochs)
agent, fig = manager.inspect_agent(key, figsize=(5, 2.5))