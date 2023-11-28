from getkey import getkey
from utils.utils import open_file, save_file
import time, os, pickle

def measure_spacebar_time():
    while True:
        print("\n Press the space bar to start.")
        key = getkey()
        if key == ' ':
            start_time = time.time()
            while True:
                key = getkey()
                if key == ' ':
                    end_time = time.time()
                    break
            break
        else:
            print("Invalid input. Press the space bar.")
    time_taken = end_time - start_time
    return time_taken

def ask_target_time():
    while True:
        print("\n Enter target time (in seconds), and then press Enter key.")
        key = input()
        if key.isnumeric():
            print(f'Entered time is {key} seconds')
            return int(key)
        else:
            print("Please enter a number.")

def interest_to_continue():
    while True:
        print("\n Enter 's' to conitnue and 'n' to not.")
        key = getkey()
        if key == 's':
            return True
        elif  key == 'n':
            return False
        else:
            print("Enter only 's' or 'n'")

def record_data():
    folder_name = 'store/'
    file_name = 'data.pkl'
    data = open_file(folder_name, file_name)
    while True:
        target_time = ask_target_time()
        actual_time = measure_spacebar_time()
        if target_time in data.keys():
            data[target_time].append(actual_time)
        else:
            data[target_time] = [actual_time]
        if not interest_to_continue():
            break
    save_file(data, folder_name, file_name)

if __name__ == '__main__':
    record_data()