import numpy as np


def normalize_array(arr, m0, m1):
    arr_min, arr_max = np.min(arr), np.max(arr)
    
    # Normalizing to [0, 1]
    arr_normalized = (arr - arr_min) / (arr_max - arr_min)
    
    # Scaling to [m0, m1]
    arr_scaled = (m1 - m0) * arr_normalized + m0
    
    return arr_scaled