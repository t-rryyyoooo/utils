import numpy as np
"""
The function in this file addresses numpy array.
Window size means HU ranges in CT images.

"""

def setWindowSize(image_array, min_value=-110, max_value=250):
    image_array = np.where(image_array < min_value, min_value, image_array)
    image_array = np.where(image_array > max_value, max_value, image_array)
    
    return image_array


