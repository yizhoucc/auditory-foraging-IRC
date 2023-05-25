# Got rid of data points where mice did 'correct reject'.
# Got rid of data points where ITI is more than 3 (can be worked around if needed, ignored for simplicity now).
# Note ToneCloudDurSec and LickwrtTrialStart is in seconds, need to convert to units of 20 milliseconds
# Minimum start_trial is 0. For first block, end_trial will be 59
# Have to offset observations, actions, rewards like in the original toy version.

import csv
import numpy as np

class DataToEpisode():
    def __init__(self, filename, env = None, start_trial = 0, end_trial = None, one_second_in_preferred_units = 50) -> None:
        self.filename = filename
        self.no_signal_nodes = env.no_signal_nodes
        self.no_penalty_nodes = env.no_penalty_nodes
        self.no_ITI_nodes =  env.no_ITI_nodes
        self.no_nodes = env.no_nodes
        self.no_attention_modes = env.no_attention_modes
        self.dict_action_possible = env.dict_action_possible
        self.observation_possible = env.observation_possible
        self.obs_certainity_possible = env.obs_certainity_possible
        self.start_trial = start_trial
        self.end_trial = end_trial
        self.one_second_in_preferred_units = one_second_in_preferred_units
        self.dict_action_meaning = {}
        for key in self.dict_action_possible.keys():
            self.dict_action_meaning[self.dict_action_possible[key]] = key

    def consider_if_true(self, data_list, ind):
        return ((float(data_list[ind][data_list[0].index('AnimalResponse')]) != 3) and (float(data_list[ind][data_list[0].index('ITIinseconds')]) <= 3))
            
    
    def read_and_filter_csv(self):
        with open(self.filename) as csvfile:
            data_list = list(csv.reader(csvfile))
        data_dict = {} 
        self.end_trial = len(data_list)-2 if self.end_trial is None else self.end_trial
        for col_ind in range(len(data_list[0])):
            data_dict[data_list[0][col_ind]] = [float(data_list[ind][col_ind]) for ind in range(self.start_trial+1,self.end_trial+2) if self.consider_if_true(data_list, ind)]
        return data_dict

    def data_as_lists(self):
        data_dict = self.read_and_filter_csv()
        self.reward_data = data_dict['RewardSize']
        self.animal_response_data = data_dict['AnimalResponse']
        # self.pupil_data = data_dict['Pupil']
        # self.run_speed_data = data_dict['runSpeed']
        self.ITI_duration =[round(self.one_second_in_preferred_units * time) for time in data_dict['ITIinseconds']] 
        self.lick_wrt_trial_start_data = [round(self.one_second_in_preferred_units * time) if not np.isnan(time) else time for time in data_dict['LickwrtTrialStart']]
        self.noise_duration_data = [round(self.one_second_in_preferred_units * time) for time in data_dict['ToneCloudDurSec']]

    
    # Add rewards.
    # Add actual pupil values.
    # Utils visualization has some bug (showing red color where it shouldn't and not ending in green).
    # Wrong, this is the case where penalty is set to 1, which is worng. Should I do episodic? Think more!
    # Make sure you get rid of the last time step for appropriate keys like in the actual dictionary.
    # Make sure you make it list of lists wherever appropriate.
    # Also after finishing everything, double check if things make sense.
    # Maybe use the same visualization as before to see (pending fixing some bug in visualization).
    # Make multiple instances of the observations and run IRC on them.
    # Let below one be for continuing case, write similar one for the episodic case.

    # It should be noise, signal, penalty, and ITI
    # ITI value should be assigned in a way it ends in 301!
    
    # continuing case, not episodic.
    def lists_to_episode(self):
        self.data_as_lists()
        self.episode = {}
        self.episode['states'] = []
        self.episode['received_food'] = []
        self.lick_choice_list = []
        self.block_log_list = []
        for ind in range(len(self.ITI_duration)):
            if len(self.block_log_list) == 0:
                temp_dict = {}
                temp_dict['reward'] = self.reward_data[0]
                temp_dict['start'] = 0
                self.block_log_list.append(temp_dict)
            if self.block_log_list[-1]['reward'] != self.reward_data[ind]:
                self.block_log_list[-1]['stop'] = len(self.episode['states']) - 1
                temp_dict = {}
                temp_dict['reward'] = self.reward_data[ind]
                temp_dict['start'] = len(self.episode['states'])
                self.block_log_list.append(temp_dict)
            self.episode['states'] += [self.no_nodes - 1 - (self.ITI_duration[ind] - 1 - i) for i in range(self.ITI_duration[ind])]
            self.lick_choice_list += len([self.no_nodes - 1 - (self.ITI_duration[ind] - 1 - i) for i in range(self.ITI_duration[ind])]) * [0]
            self.episode['received_food'] += len([self.no_nodes - 1 - (self.ITI_duration[ind] - 1 - i) for i in range(self.ITI_duration[ind])]) * [0]
            if self.animal_response_data[ind] == 3:
                raise NotImplementedError("Didn't implement for the case of Correct Rejection.")
            elif np.isnan(self.lick_wrt_trial_start_data[ind]):
                self.episode['states'] += [0 for _ in range(self.noise_duration_data[ind])]
                self.episode['states'] += [1+i for i in range(self.no_signal_nodes)]                
                self.lick_choice_list += (self.noise_duration_data[ind]+self.no_signal_nodes) * [0]
                self.episode['received_food'] += (self.noise_duration_data[ind]+self.no_signal_nodes) * [0]
            elif self.lick_wrt_trial_start_data[ind] <= self.noise_duration_data[ind]:
                self.episode['states'] += [0 for _ in range(self.lick_wrt_trial_start_data[ind])]
                self.episode['states'] += [1+self.no_signal_nodes+i for i in range(self.no_penalty_nodes)] 
                self.lick_choice_list += (self.lick_wrt_trial_start_data[ind]-1) * [0] + [1] + self.no_penalty_nodes * [0]  # need to change this to episodic case, maybe implement both.
                self.episode['received_food'] += (self.lick_wrt_trial_start_data[ind] + self.no_penalty_nodes) * [0]
            else:
                self.episode['states'] += [0 for _ in range(self.noise_duration_data[ind])]
                self.episode['states'] += [1+i for i in range(self.lick_wrt_trial_start_data[ind]-self.noise_duration_data[ind])]
                self.lick_choice_list += (self.lick_wrt_trial_start_data[ind]-1) * [0] + [1]
                self.episode['received_food'] += (self.lick_wrt_trial_start_data[ind]-1) * [0] + [self.reward_data[ind]] # recieved food reward at the same time step of lick
        self.block_log_list[-1]['stop'] = len(self.episode['states'])

        #fake
        self.pupil_data = np.random.uniform(low=0.0, high=1.0, size=len(self.episode['states']))
        self.attention_choice_list = np.digitize(self.pupil_data, np.linspace(np.min(self.pupil_data), np.max(self.pupil_data)+1e-10, num=self.no_attention_modes+1))-1

        self.episode['actions'] = [self.dict_action_meaning[(self.lick_choice_list[ind],self.attention_choice_list[ind])] for ind in range(len(self.lick_choice_list))]
        self.episode['observations'] = []
        for ind in range(len(self.episode['actions'])):
            if self.episode['states'][ind] == 0:
                self.episode['observations'].append(list(self.observation_possible).index(1 - np.random.binomial(size=1, n=1, p= self.obs_certainity_possible[self.attention_choice_list[ind]])[0]))
            elif self.episode['states'][ind] in range(1, self.no_signal_nodes + 1):
                self.episode['observations'].append(list(self.observation_possible).index(np.random.binomial(size=1, n=1, p= self.obs_certainity_possible[self.attention_choice_list[ind]])[0]))
            elif self.episode['states'][ind] in range(self.no_signal_nodes + 1, self.no_signal_nodes + self.no_penalty_nodes + 1):
                self.episode['observations'].append(self.observation_possible[-2])
            elif self.episode['states'][ind] in range(self.no_signal_nodes + self.no_penalty_nodes + 1, self.no_nodes):
                self.episode['observations'].append(self.observation_possible[-1])
    
    def edit_episode_to_IRC_format(self, unformatted_episode):
        formatted_episode = {}
        formatted_episode['actions'] = unformatted_episode['actions'][:-1]
        formatted_episode['received_food'] = unformatted_episode['received_food'][:-1]
        formatted_episode['states'] = [[state] for state in unformatted_episode['states']]
        formatted_episode['observations'] = [[obs] for obs in unformatted_episode['observations']]
        return formatted_episode

    def chop_episode(self, episode, start, stop):
        chopped_episode = {}
        chopped_episode['actions'] = episode['actions'][start:stop+1]
        chopped_episode['received_food'] = episode['received_food'][start:stop+1]
        chopped_episode['states'] = episode['states'][start:stop+1]
        chopped_episode['observations'] = episode['observations'][start:stop+1]
        return chopped_episode

    def find_episodic_start_stop_time_points(self, continuing_episode):
        states = continuing_episode['states']
        episodic_start_list = [0] + [ind for ind in range(1,len(states)) if (states[ind-1] == self.no_nodes - 1 and states[ind] ==  0)]
        episodic_stop_list = [ind for ind in range(1,len(states)) if states[ind] ==  1 + self.no_signal_nodes]
        if len(episodic_stop_list) == len(episodic_start_list) - 1:
            episodic_stop_list.append(len(states))
        elif len(episodic_stop_list) != len(episodic_start_list):
            raise Exception("There is an error in the method find_episodic_start_stop_time_points")
        return episodic_start_list, episodic_stop_list

    
    def data_for_IRC(self, is_continuing = False):
        low_reward_blocks = {}
        low_reward_blocks['episodes'] = []
        low_reward_blocks['block_indices'] = []
        high_reward_blocks = {}
        high_reward_blocks['episodes'] = []
        high_reward_blocks['block_indices'] = []
        data_IRC = {}
        block_ind = 0
        if is_continuing:
            for block_log in range(len(self.block_log_list)):
                formatted_chopped_episode = self.edit_episode_to_IRC_format(self.chop_episode(self.episode, block_log['start'], block_log['stop']))
                if block_log['reward'] == 1:
                    low_reward_blocks['episodes'].append(formatted_chopped_episode)
                    low_reward_blocks['block_indices'].append(block_ind)
                elif block_log['reward'] == 2:
                    high_reward_blocks['episodes'].append(formatted_chopped_episode)
                    high_reward_blocks['block_indices'].append(block_ind)
                else:
                    raise NotImplementedError("Reward size can only be 1 or 2.")
                block_ind += 1
        else:
            for block_log in range(len(self.block_log_list)):
                chopped_continuing_episode = self.chop_episode(block_log['start'], block_log['stop'])
                episodic_start_list, episodic_stop_list = self.find_episodic_start_stop_time_points(chopped_continuing_episode)
                temp_list = []
                for ind in range(len(episodic_start_list)):
                    temp_list.append(self.edit_episode_to_IRC_format(self.chop_episode(chopped_continuing_episode, episodic_start_list[ind], episodic_stop_list[ind])))
                    if block_log['reward'] == 1:
                        low_reward_blocks['episodes'].append(temp_list)
                        low_reward_blocks['block_indices'].append(block_ind)
                    elif block_log['reward'] == 2:
                        high_reward_blocks['episodes'].append(temp_list)
                        high_reward_blocks['block_indices'].append(block_ind)
                    else:
                        raise NotImplementedError("Reward size can only be 1 or 2.")
                block_ind += 1
        data_IRC['low_reward_blocks'] = low_reward_blocks
        data_IRC['high_reward_blocks'] = high_reward_blocks
        return data_IRC

