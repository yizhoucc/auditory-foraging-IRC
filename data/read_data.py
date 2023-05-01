# Got rid of data points where mice did 'correct 
# Note ToneCloudDurSec and LickwrtTrialStart is in seconds, need to convert to milliseconds
import matplotlib.pyplot as plt
import csv

def read_and_filter_csv(file_name):
    with open(file_name) as csvfile:
        data_list = list(csv.reader(csvfile))
    data_dict = {}
    for col_ind in range(len(data_list[0])):
        data_dict[data_list[0][col_ind]] = [float(data_list[ind][col_ind]) for ind in range(1,len(data_list)) if float(data_list[ind][1]) != 3]
    return data_dict

def data_as_lists(file_name):
    data_dict = read_and_filter_csv(file_name)

    reward_data = data_dict['RewardSize']
    noise_duration_data = data_dict['ToneCloudDurSec']
    pupil_data = data_dict['Pupil']
    run_speed_data = data_dict['runSpeed']
    
    one_second_in_preferred_units = 1000
    animal_response_data = [one_second_in_preferred_units * time for time in data_dict['AnimalResponse']]
    lick_wrt_trial_start_data = [one_second_in_preferred_units * time for time in data_dict['LickwrtTrialStart']]
    
    return reward_data, animal_response_data, noise_duration_data, pupil_data, run_speed_data, lick_wrt_trial_start_data

file_name = 'W3333_29.csv'
reward_data, animal_response_data, noise_duration_data, pupil_data, run_speed_data, lick_wrt_trial_start_data = data_as_lists(file_name)
plt.stem(lick_wrt_trial_start_data)
plt.show()