import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

def load_np(path, allow_pickle=False):
    """
    custom loading function for numpy files, because we require extra steps to import from a different file directory
    Returns a numpy loaded file
    """
    file = os.path.join(parent_dir, 'calib', path)
    
    if not allow_pickle:
        return np.load(file)
    else:
        return np.load(file, allow_pickle=True)

def full_path(path):
    file = os.path.join(parent_dir, 'calib', path)
    return file