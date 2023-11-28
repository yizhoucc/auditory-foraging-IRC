import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')

from irc.manager import IRCManager

############

defaults = {
    'agent.env._target_': 'auditoryforage.AF_env.AuditoryForaging',
    'agent.model._target_': 'irc.model.FuncBeliefModel',
}
manager = IRCManager(defaults=defaults)

############

def count_no_elements(np_array, min_allowed, max_allowed):
    return len([elt for elt in list(np_array.flatten()) if elt >= min_allowed and elt <= max_allowed])

############

from auditoryforage.AF_env import AuditoryForaging

num_epochs = 100
no_episodes = 1000
lick_cost_list = [0]
# food_reward_list = [0, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0, 75.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 1000.0]
food_reward_list = [0, 5.0, 10.0, 15.0, 20.0, 25.0, 50.0, 75.0, 100.0]
seed_list = [0, 1, 2, 3, 4]

attention_cost_coeff_list = [.16]
attention_cost_temp_list = [.25]
penalty_cost_list = [0]
iti_cost_list = [0]


hit_count_list = []
miss_count_list = []
false_alarm_list = []
noise_time_before_lick_list = []
signal_time_before_lick_list = []
total_noise_time_list = []
total_signal_time_list = []

for lick_cost in lick_cost_list:
    for food_reward in food_reward_list:
        for attention_cost_coeff in attention_cost_coeff_list:
            for attention_cost_temp in attention_cost_temp_list:
                for penalty_cost in penalty_cost_list:
                    for iti_cost in iti_cost_list:
                        hit_count_list.append([])
                        miss_count_list.append([])
                        false_alarm_list.append([])
                        noise_time_before_lick_list.append([])
                        signal_time_before_lick_list.append([])
                        total_noise_time_list.append([])
                        total_signal_time_list.append([])
                        env_param = [lick_cost, food_reward, attention_cost_coeff, attention_cost_temp, penalty_cost, iti_cost]
                        env = AuditoryForaging(spec={'agent':{'lick_cost':env_param[0],'food_reward':env_param[1],'attention_cost_coeff':env_param[2], 'attention_cost_temp': env_param[3], 'penalty_cost': env_param[4], 'iti_cost': env_param[5]}})
                        for seed in seed_list:
                            hit_count = 0 
                            miss_count = 0 
                            false_alarm_count = 0 
                            noise_time_before_lick = 0 
                            signal_time_before_lick = 0 
                            total_noise_time = 0 
                            total_signal_time = 0
                            agent = manager.train_agent(env_param, num_epochs=num_epochs, seed = seed)
                            for episide_no in range(no_episodes):
                                episode = agent.run_one_episode(env=env, num_steps=100000, q_states = [[i] for i in range(env.no_nodes)])
                                if episode['rewards'][-1] > 0:
                                    hit_count += 1
                                    signal_time_before_lick += count_no_elements(episode['states'], 1, env.no_signal_nodes)
                                elif episode['states'][-1] > env.no_signal_nodes + 1:
                                    miss_count += 1
                                elif episode['states'][-1] == env.no_signal_nodes + 1:
                                    false_alarm_count += 1
                                    noise_time_before_lick += count_no_elements(episode['states'], 0, 0)
                                else:
                                    print("Error somewhere")
                                total_signal_time += count_no_elements(episode['states'], 1, env.no_signal_nodes)
                                total_noise_time += count_no_elements(episode['states'], 0, 0)
                            hit_count_list[-1].append(hit_count)
                            miss_count_list[-1].append(miss_count)
                            false_alarm_list[-1].append(false_alarm_count)
                            noise_time_before_lick_list[-1].append(noise_time_before_lick)
                            signal_time_before_lick_list[-1].append(signal_time_before_lick)
                            total_noise_time_list[-1].append(total_noise_time)
                            total_signal_time_list[-1].append(total_signal_time)

############

from auditoryforage.utils import save_pickle_file

folder_name = './store/comparison/'
file_name = folder_name + 'diff_food_rewards.pkl'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

detection_measures = {}
input = {}
output = {}
seed_list
input['no_episodes'] = no_episodes
input['seed_list'] = seed_list
input['food_reward_list'] = food_reward_list
input['attention_cost_coeff_list'] = attention_cost_coeff_list
input['attention_cost_temp_list'] = attention_cost_temp_list
input['penalty_cost_list'] = penalty_cost_list
input['iti_cost_list'] = iti_cost_list
output['hit_count_list'] = hit_count_list
output['miss_count_list'] = miss_count_list
output['false_alarm_list'] = false_alarm_list
output['noise_time_before_lick_list'] = noise_time_before_lick_list
output['signal_time_before_lick_list'] = signal_time_before_lick_list
output['total_noise_time_list'] = total_noise_time_list
output['total_signal_time_list'] = total_signal_time_list
detection_measures['input'] = input
detection_measures['output'] = output
save_pickle_file(detection_measures, file_name)