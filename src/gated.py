import os
from PIL import Image
import numpy as np

def load_gated(data_root, sample, gated_types=['gated0_10bit', 'gated17_10bit', 'gated31_10bit'], dark_level=[87, 87, 87], width=1280, height=720):
    gated_array = np.zeros((width * height, len(gated_types)), dtype=float)
    for i, gated_type in enumerate(gated_types):
        picture_file_gated = os.path.join(data_root, gated_type, sample + '.png')
        picture = cv2.imread(picture_file_gated, -1)
        #picture = Image.open(picture_file_gated)
        #gated_array[:, i] = np.clip(np.array(picture).flatten().astype(np.float32) - dark_level[i], 0.0, 1023.0)
        gated_array[:, i] = np.clip(picture.flatten().astype(np.float32) - dark_level[i], 0.0, 1023.0)

    return gated_array

def get_valid(gated_array, dark=0.00, retro=np.inf, var=0.025, max_val=1023):

    retro = retro * max_val # retrereflective
    dark = dark * max_val # not or less illuminated
    var = var * max_val # too less variation

    # select valid pixels
    max_value = np.amax(gated_array, axis=1)
    min_value = np.amin(gated_array, axis=1)
    mean_value = np.mean(gated_array, axis=1)
    is_valid = ((max_value - min_value) > var) & (max_value < retro) & (mean_value > dark)  # & ~minimumAtCenterSlice  # this is the default configuration

    return is_valid
