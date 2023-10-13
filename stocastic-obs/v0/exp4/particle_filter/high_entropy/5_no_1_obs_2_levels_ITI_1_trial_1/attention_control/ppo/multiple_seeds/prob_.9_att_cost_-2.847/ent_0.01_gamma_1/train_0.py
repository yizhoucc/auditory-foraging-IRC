import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')

from irc.manager import IRCManager

##############################################################################

defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

##############################################################################

env_param = [0, 0, .08, .25, 0, 0]

# change back to old no. of epochs
num_epochs = 400

for seed in range(5):
    print(f'running seed {seed}')
    agent = manager.train_agent(env_param, num_epochs=num_epochs, seed = seed)

