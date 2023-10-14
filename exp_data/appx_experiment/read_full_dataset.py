import csv
import numpy as np
import os
import sys
sys.path.append(f'{os.getcwd()}/irc_gym')

from auditoryforage.AF_env import AuditoryForaging
env = AuditoryForaging()

filename = f'../data_files/orig/3_data_behavior.csv'
hit_skip_trials = 15
one_sec_in_node_units = 50

def make_episode(noise_dur, rt, outcome):
    # 0=hit, 1=miss,2-false alarm,3=Correct 
    # last state's action not taken
    episode = {}
    iti_node = env.no_signal_nodes + env.no_penalty_nodes + 1
    if outcome == 0:
        episode['states'] = [[iti_node]] + [[0]] * round(noise_dur * one_sec_in_node_units) + [[1 + node] for node in range(round((rt - noise_dur) * one_sec_in_node_units))] + [[iti_node]]
        episode['actions'] = (len(episode['states'])-2) * [0] + [env.no_attention_modes]
    elif outcome == 1:
        episode['states'] = [[iti_node]] + [[0]] * round(noise_dur * one_sec_in_node_units) + [[1 + node] for node in range(env.no_signal_nodes)] + [[iti_node]]
        episode['actions'] = (len(episode['states'])-1) * [0]
    elif outcome == 2:
        episode['states'] = [[iti_node]] + [[0]] * round(rt * one_sec_in_node_units) + [[env.no_signal_nodes + 1]]
        episode['actions'] = (len(episode['states'])-2) * [0] + [env.no_attention_modes]
    else:
        print('some mistake here!')
    episode['states'] = np.array(episode['states'])
    return episode


with open(filename) as csvfile:
    data_list = list(csv.reader(csvfile))

compiled_data = {}
title_list = data_list[0]
subj_ind = title_list.index('subject_id')
outcome_ind = title_list.index('outcome')
block_id_ind = title_list.index('block_id')
catch_ind = title_list.index('catch')
rt_ind = title_list.index('rt')
noise_dur_ind = title_list.index('noise_dur')
reached_end_flag = 0

row_ind = 1
while row_ind < len(data_list):
    
    # skip block 0
    if float(data_list[row_ind][block_id_ind]) == 0:
        while float(data_list[row_ind][block_id_ind]) == 0:
            row_ind += 1
        continue

    curr_subj_id = data_list[row_ind][subj_ind]
    curr_block_id = data_list[row_ind][block_id_ind]
    
    if data_list[row_ind][subj_ind] not in compiled_data.keys():
        compiled_data[curr_subj_id] = {}
    
    if data_list[row_ind][block_id_ind] not in compiled_data[data_list[row_ind][subj_ind]].keys():
        compiled_data[data_list[row_ind][subj_ind]][curr_block_id] = []

    # skip 14 trials after first hit in new block
    while float(data_list[row_ind][outcome_ind]) != 0:
        row_ind += 1        
    row_ind += hit_skip_trials

    while data_list[row_ind][subj_ind] == curr_subj_id and data_list[row_ind][block_id_ind] == curr_block_id:
        
        # skip catch trials
        if float(data_list[row_ind][catch_ind]) == 0:
            outcome = float(data_list[row_ind][outcome_ind])
            rt = float(data_list[row_ind][rt_ind]) if data_list[row_ind][rt_ind] != '' else None
            noise_dur = float(data_list[row_ind][noise_dur_ind]) if data_list[row_ind][noise_dur_ind] != '' else None
            compiled_data[data_list[row_ind][subj_ind]][curr_block_id].append(make_episode(noise_dur, rt, outcome))
        
        if row_ind < len(data_list) - 1:
            row_ind += 1
        else:
            row_ind = len(data_list)
            break

from auditoryforage.utils import save_pickle_file
file_name = '../data_files/compiled/3_data_behavior.pkl'
save_pickle_file(compiled_data, file_name)