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
        # print(ind)
        # print(float(data_list[ind][data_list[0].index('AnimalResponse')]))
        # print(float(data_list[ind][data_list[0].index('ITIinseconds')]))
        return ((float(data_list[ind][data_list[0].index('AnimalResponse')]) != 3) and (float(data_list[ind][data_list[0].index('ITIinseconds')]) <= 3))
            
    
    def read_and_filter_csv(self):
        with open(self.filename) as csvfile:
            data_list = list(csv.reader(csvfile))
        data_dict = {} 
        self.end_trial = len(data_list)-2 if self.end_trial is None else self.end_trial

        print('here')
        print(len(data_list))
        print('there')
        print(data_list[-1])

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

    
    # Add actual pupil values.
    # Utils visualization has some bug (showing red color where it shouldn't and not ending in green).
    # Wrong, this is the case where penalty is set to 1, which is worng. Should I do episodic? Think more!
    # Make sure you get rid of the last time step for appropriate keys like in the actual dictionary.
    # Make sure you make it list of lists wherever appropriate.
    # Also after finishing everything, double check if things make sense.
    # Make multiple instances of the observations and run IRC on them.
    # Let below one be for continuing case, write similar one for the episodic case.

    # It should be noise, signal, penalty, and ITI
    # ITI value should be assigned in a way it ends in 301!
    
    # continuing case, not episodic.
    def lists_to_episode(self):
        self.data_as_lists()
        self.episode = {}
        self.episode['states'] = []
        self.lick_choice_list = []
        for ind in range(len(self.ITI_duration)):
            self.episode['states'] += [self.no_nodes - 1 - (self.ITI_duration[ind] - 1 - i) for i in range(self.ITI_duration[ind])]
            self.lick_choice_list += len([self.no_nodes - 1 - (self.ITI_duration[ind] - 1 - i) for i in range(self.ITI_duration[ind])]) * [0]
            if self.animal_response_data[ind] == 3:
                raise NotImplementedError("Didn't implement for the case of Correct Rejection.")
            elif np.isnan(self.lick_wrt_trial_start_data[ind]):
                self.episode['states'] += [0 for _ in range(self.noise_duration_data[ind])]
                self.episode['states'] += [1+i for i in range(self.no_signal_nodes)]                
                self.lick_choice_list += (self.noise_duration_data[ind]+self.no_signal_nodes) * [0]
            elif self.lick_wrt_trial_start_data[ind] <= self.noise_duration_data[ind]:
                self.episode['states'] += [0 for _ in range(self.lick_wrt_trial_start_data[ind])]
                self.episode['states'] += [1+self.no_signal_nodes+i for i in range(self.no_penalty_nodes)] 
                self.lick_choice_list += (self.lick_wrt_trial_start_data[ind]-1) * [0] + [1] + self.no_penalty_nodes * [0]  # need to change this to episodic case, maybe implement both.
            else:
                self.episode['states'] += [0 for _ in range(self.noise_duration_data[ind])]
                self.episode['states'] += [1+i for i in range(self.lick_wrt_trial_start_data[ind]-self.noise_duration_data[ind])]
                self.lick_choice_list += (self.lick_wrt_trial_start_data[ind]-1) * [0] + [1]

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