import os
import sys
import numpy as np
sys.path.append(f'{os.getcwd()}/irc_gym')

no_episodes=10
food_reward_list = [1000, 1100, 1200, 1300, 1400, 1500]
agent=1
env=1


def count_no_elements(np_array, min_allowed, max_allowed):
    return len([elt for elt in list(np_array.flatten()) if elt >= min_allowed and elt <= max_allowed])



hit_count_list = [[0] for _ in range(len(food_reward_list))]
miss_count_list = [[0] for _ in range(len(food_reward_list))]
false_alarm_list = [[0] for _ in range(len(food_reward_list))]
noise_time_before_lick_list = [[0] for _ in range(len(food_reward_list))]
signal_time_before_lick_list = [[0] for _ in range(len(food_reward_list))]
total_noise_time_list = [[0] for _ in range(len(food_reward_list))]
total_signal_time_list = [[0] for _ in range(len(food_reward_list))]
best_seed_list = [[0] for _ in range(len(food_reward_list))]

attention_time_points_list = [[] for _ in range(len(food_reward_list))]
episode_length_list = [[0] for _ in range(len(food_reward_list))]
hit_reaction_time_list = [[] for _ in range(len(food_reward_list))]
fa_reaction_time_list = [[0] for _ in range(len(food_reward_list))]

total_reward_list_across_exps = [[0] for _ in range(len(food_reward_list))]





for episide_no in range(no_episodes):
    episode = agent.run_one_episode()

    hit_count = 0 
    miss_count = 0 
    false_alarm_count = 0 
    noise_time_before_lick = 0 
    signal_time_before_lick = 0 
    total_noise_time = 0 
    total_signal_time = 0
    total_reward = 0

    attention_time_points_across_episodes = []
    episode_length_across_episodes = []
    hit_reaction_time_across_episodes = []
    fa_reaction_time_across_episodes = []

    # process single ep data
    if episode['actions'][-1][0] == 1:
        if episode['states'][-1][0] > env.no_signal_nodes + env.no_penalty_nodes:
            hit_count += 1
            signal_time_before_lick_curr_episode = count_no_elements(episode['states'], 1, env.no_signal_nodes)
            hit_reaction_time_across_episodes.append(signal_time_before_lick_curr_episode)
            signal_time_before_lick += signal_time_before_lick_curr_episode
        else:
            false_alarm_count += 1
            noise_time_before_lick_curr_episode = count_no_elements(episode['states'], 0, 0)
            fa_reaction_time_across_episodes.append(noise_time_before_lick_curr_episode)
            noise_time_before_lick += noise_time_before_lick_curr_episode
    else:
        miss_count += 1                            
    total_signal_time += count_no_elements(episode['states'], 1, env.no_signal_nodes)
    total_noise_time += count_no_elements(episode['states'], 0, 0)
    total_reward += sum(episode['rewards'])
    attention_time_points_across_episodes.append(np.where(np.array([elt[1] for elt in episode['actions']] )== 1)[0])
    episode_length_across_episodes.append(len(episode['states']))
    # end process single ep data

    # log
    food_reward_idx=episode['trial_food_reward_idx']

    hit_count_list[food_reward_idx]+=(hit_count)
    miss_count_list[food_reward_idx]+=(miss_count)
    false_alarm_list[food_reward_idx]+=(false_alarm_count)

    noise_time_before_lick_list[food_reward_idx]+=(noise_time_before_lick)
    signal_time_before_lick_list[food_reward_idx]+=(signal_time_before_lick)
    total_noise_time_list[food_reward_idx]+=(total_noise_time)
    total_signal_time_list[food_reward_idx]+=(total_signal_time)

    attention_time_points_list[food_reward_idx].append(attention_time_points_across_episodes)
    episode_length_list[food_reward_idx].append(episode_length_across_episodes)
    hit_reaction_time_list[food_reward_idx].append(hit_reaction_time_across_episodes)
    fa_reaction_time_list[food_reward_idx].append(fa_reaction_time_across_episodes)













############
from auditoryforage.utils import save_pickle_file

folder_name = './store/comparison/'
file_name = folder_name + f'#seeds_{len(seed_list)}_#epochs_{num_epochs}.pkl'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

detection_measures = {}
input = {}
output = {}
seed_list
input['no_episodes'] = no_episodes
input['seed_list'] = seed_list
input['food_reward_list'] = food_reward_list
input['att_coeff_list'] = att_coeff_list
input['att_temp_list'] = att_temp_list
input['penalty_cost_list'] = penalty_cost_list
input['iti_cost_list'] = iti_cost_list
output['hit_count_list'] = hit_count_list
output['miss_count_list'] = miss_count_list
output['false_alarm_list'] = false_alarm_list
output['noise_time_before_lick_list'] = noise_time_before_lick_list
output['signal_time_before_lick_list'] = signal_time_before_lick_list
output['total_noise_time_list'] = total_noise_time_list
output['total_signal_time_list'] = total_signal_time_list
output['best_seed_list'] = best_seed_list
output['attention_time_points_list'] = attention_time_points_list
output['episode_length_list'] = episode_length_list
output['hit_reaction_time_list'] = hit_reaction_time_list
output['fa_reaction_time_list'] = fa_reaction_time_list
output['total_reward_list_across_exps'] = total_reward_list_across_exps
detection_measures['input'] = input
detection_measures['output'] = output
save_pickle_file(detection_measures, file_name)