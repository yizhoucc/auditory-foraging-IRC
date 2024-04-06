import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')

from irc.manager import IRCManager



defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

food_reward_list = [1000, 1100, 1200, 1300, 1400, 1500]
att_coeff = 0.087213
att_temp = 0.25
penalty_cost = -30
time_in_game_reward = 0.
num_epochs = 10
seed_value = 1

for food_reward in food_reward_list:

    env_param = [0, food_reward, att_coeff, att_temp,
                 penalty_cost, 0, time_in_game_reward]
    print("########################################################################")
    print(f"\n \n Training going on for {num_epochs} epochs\n \n ")
    print("########################################################################")
    agent = manager.train_agent(
        env_param, num_epochs=num_epochs, seed=seed_value)

