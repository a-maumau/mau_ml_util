import os

def mkdir(folder_path):
    if not os.path.exists(os.path.expanduser(folder_path)):
        os.makedirs(os.path.expanduser(folder_path))

def is_dir(folder_path):
	return os.path.isdir(os.path.expanduser(folder_path))

def is_exist(folder_path):
	return os.path.exists(os.path.expanduser(folder_path))
	
def path_join(*paths):
	return os.path.join(*paths)

def list_dir(path):
	return os.listdir(os.path.expanduser(path))