import os

def mkdir(folder_path):
    if not os.path.exists(os.path.abspath(folder_path)):
        os.makedirs(os.path.abspath(folder_path))

def is_dir(folder_path):
	return os.path.isdir(os.path.abspath(folder_path))

def is_exist(folder_path):
	return os.path.exists(os.path.abspath(folder_path))
	
def path_join(*paths):
	return os.path.join(*paths)

def list_dir(path):
	return os.listdir(os.path.abspath(path))