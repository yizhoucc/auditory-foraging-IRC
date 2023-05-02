# Got rid of data points where mice did 'correct reject'.
# Note ToneCloudDurSec and LickwrtTrialStart is in seconds, need to convert to units of 20 milliseconds
# Minimum start_trial is 0. For first block, end_trial will be 59

import csv
import numpy as np

class DataToEpisode():
    def __init__(self, filename, env = None, start_trial = 0, end_trial = None, one_second_in_preferred_units = 50) -> None:
        self.filename = filename
        self.no_signal_nodes = env.no_signal_nodes if env != None else None
        self.no_penalty_nodes = env.no_penalty_nodes if env != None else None
        self.no_ITI_nodes =  env.no_ITI_nodes
        self.start_trial = start_trial
        self.end_trial = end_trial
        self.one_second_in_preferred_units = one_second_in_preferred_units

    def read_and_filter_csv(self):
        with open(self.filename) as csvfile:
            data_list = list(csv.reader(csvfile))
        data_dict = {} 
        self.end_trial = len(data_list)-2 if self.end_trial is None else self.end_trial
        for col_ind in range(len(data_list[0])):
            data_dict[data_list[0][col_ind]] = [float(data_list[ind][col_ind]) for ind in range(self.start_trial+1,self.end_trial+2) if float(data_list[ind][1]) != 3]
        return data_dict

    def data_as_lists(self):
        data_dict = self.read_and_filter_csv()
        self.reward_data = data_dict['RewardSize']
        self.animal_response_data = data_dict['AnimalResponse']
        self.pupil_data = data_dict['Pupil']
        self.run_speed_data = data_dict['runSpeed']
        self.lick_wrt_trial_start_data = [round(self.one_second_in_preferred_units * time) if not np.isnan(time) else time for time in data_dict['LickwrtTrialStart']]
        self.noise_duration_data = [round(self.one_second_in_preferred_units * time) for time in data_dict['ToneCloudDurSec']]
        
        #fake
        self.ITI_duration = [round(self.one_second_in_preferred_units * 2.5) for _ in self.noise_duration_data]

    def lists_to_episode(self):
        self.data_as_lists()
        self.episode = {}
        self.episode['states'] = []
        for ind in range(len(self.ITI_duration)):
            self.episode['states'] += [i for i in range(self.ITI_duration[ind])]
            if self.animal_response_data[ind] == 3:
                raise NotImplementedError("Didn't implement for the case of Correct Rejection.")
            elif np.isnan(self.lick_wrt_trial_start_data[ind]):
                self.episode['states'] += [self.no_ITI_nodes for _ in range(self.noise_duration_data[ind])]
                self.episode['states'] += [self.no_ITI_nodes+1+i for i in range(self.no_signal_nodes)]
            elif self.lick_wrt_trial_start_data[ind] <= self.noise_duration_data[ind]:
                self.episode['states'] += [self.no_ITI_nodes for _ in range(self.lick_wrt_trial_start_data[ind])]
                self.episode['states'] += [self.no_ITI_nodes+1+self.no_signal_nodes+i for i in range(self.no_penalty_nodes)]
            else:
                self.episode['states'] += [self.no_ITI_nodes for _ in range(self.noise_duration_data[ind])]
                self.episode['states'] += [self.no_ITI_nodes+1+i for i in range(self.lick_wrt_trial_start_data[ind]-self.noise_duration_data[ind])]



if __name__ == "__main__":
    filename = 'W3333_29.csv'
    end_trial = 59
    data_to_episode = DataToEpisode(filename = filename, end_trial = end_trial)
    data_to_episode.lists_to_episode()
    print(data_to_episode.epsiode['states'])