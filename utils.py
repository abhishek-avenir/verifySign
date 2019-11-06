
import os
import pickle

def save_to_pickle(data, file_path):
	dir_name = os.path.dirname(file_path)
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	with open(file_path, 'wb') as f:
		pickle.dump(data, f)

def load_pickle(file_path):
	with open(file_path, 'rb') as f:
		return pickle.load(f)