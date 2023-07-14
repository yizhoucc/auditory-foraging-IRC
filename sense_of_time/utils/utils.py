import os, pickle

def open_file(folder_name, file_name):
    file_directory = folder_name + file_name
    if os.path.isfile(file_directory):
        open_file = open(file_directory, "rb")
        data = pickle.load(open_file)
        open_file.close()
    else:
        data = {}
    return data

def save_file(data, folder_name, file_name):
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    file_name = folder_name + file_name
    open_file = open(file_name, "wb")
    pickle.dump(data, open_file)
    open_file.close()