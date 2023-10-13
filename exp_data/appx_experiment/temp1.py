import csv, copy
import numpy as np

filename = f'../data_files/orig/3_data_behavior.csv'

with open(filename) as csvfile:
    data_list = list(csv.reader(csvfile))
# data_dict = {} 
# start_trial = 0
# end_trial = len(data_list)-2
# for col_ind in range(len(data_list[0])):
#     data_dict[data_list[0][col_ind]] = [float(data_list[ind][col_ind]) for ind in range(start_trial+1,end_trial+2)]
# print(data_dict)
print(data_list[0])