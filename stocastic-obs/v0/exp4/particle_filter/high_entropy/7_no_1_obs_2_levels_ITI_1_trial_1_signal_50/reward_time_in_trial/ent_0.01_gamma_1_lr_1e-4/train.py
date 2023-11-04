import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')

##############################################################################

# Check if at least one argument is provided
if len(sys.argv) < 8:
    print("ERROR! Correct usage is python train.py <food_reward> <att_coeff> <att_temp> <penalty_cost> <time_in_game_reward><num_epochs> <seed_value>")
    sys.exit(1)
else:
    food_reward = float(sys.argv[1])
    att_coeff = float(sys.argv[2])
    att_temp = float(sys.argv[3])
    penalty_cost = float(sys.argv[4])
    time_in_game_reward = float(sys.argv[5])
    num_epochs = int(sys.argv[6])
    seed_value = int(sys.argv[7])
    

##############################################################################

from irc.manager import IRCManager

##############################################################################

defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

##############################################################################

env_param = [0, food_reward, att_coeff, att_temp, penalty_cost, 0, time_in_game_reward]
print("########################################################################")
print(f"\n \n Training going on for {num_epochs} epochs\n \n ")
print("########################################################################")
agent = manager.train_agent(env_param, num_epochs=num_epochs, seed = seed_value)