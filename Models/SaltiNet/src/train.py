from .pathnet import get_model, get_nick_model
from .utils import load_image

import numpy as np
from scipy import ndimage
from PIL import Image
import cv2
import keras
import math

import os

_LONGEST_OSIE_SCANPATH_DURATION = 2657.0
_LONGEST_SALICON_SCANPATH_DURATION = 6000.0

_QUANTIZATION_SLICES = 12

_OSIE_IMAGE_SHAPE = [600, 800]
_SALICON_IMAGE_SHAPE = [480, 640]
_360_IMAGE_SHAPE = [300, 600]

start = 0

def _create_sal_volumes(fixations, dataset, from_save=True, to_save=True):
    if dataset in ["OSIE", ]:
        step_time = _LONGEST_OSIE_SCANPATH_DURATION / _QUANTIZATION_SLICES
    elif dataset in ["SALICON", ]:
        step_time = _LONGEST_SALICON_SCANPATH_DURATION / _QUANTIZATION_SLICES

    count = 0
    sal_volumes = list()
    for index, image in enumerate(fixations):
        if index < start:
            continue
        curr_sal_vol = np.zeros([12, 300, 600], dtype=np.float32)
        for fix in image:
            if fix[-1, 2] > _LONGEST_SALICON_SCANPATH_DURATION or fix[-1, 2] == float("inf"):
                continue

            if dataset in ["OSIE", ]:
                for i in range(fix.shape[0]):
                    time_step = int(np.sum(fix[:i, 2]) / step_time)
                    height_val = int(fix[i][1] / _OSIE_IMAGE_SHAPE[0] * _360_IMAGE_SHAPE[0])
                    width_val = int(fix[i][0] / _OSIE_IMAGE_SHAPE[1] * _360_IMAGE_SHAPE[1])
                    curr_sal_vol[time_step, height_val, width_val] = 1
            elif dataset in ["SALICON", ]:
                for i in range(fix.shape[0]):
                    time_step = int(fix[i, 2] / step_time)
                    height_val = int(fix[i][1] / _SALICON_IMAGE_SHAPE[0] * _360_IMAGE_SHAPE[0])
                    width_val = int(fix[i][0] / _SALICON_IMAGE_SHAPE[1] * _360_IMAGE_SHAPE[1])
                    if time_step >= 12:
                        print("TIME_STEP ERROR ", time_step)
                        time_step = 11
                    if height_val >= 300:
                        height_val = 299
                    if width_val >= 600:
                        width_val = 599
                    curr_sal_vol[time_step, height_val, width_val] = 1
        #saving salience volumes for testing
        curr_sal_vol = ndimage.gaussian_filter(curr_sal_vol, [4, 20, 20])
        for i, temp in enumerate(curr_sal_vol):
            time_sum = temp.sum()
            curr_sal_vol[i] /= time_sum

        for i in range(12):
            curr_sal_vol[i] /= np.amax(curr_sal_vol[i])
        
        print(index)
        sal_volumes.append(curr_sal_vol)

        count += 1

        if to_save and count % 250 == 0 and index != start:
            number = int(math.ceil(index / 250))
            save_val = np.array(sal_volumes)
            print(f"SAL VOL LENGTH: {save_val.shape}")
            np.save(f"sal_volumes_{number}.npy", save_val)
            sal_volumes = list()
            count = 0
    
    sal_volumes = np.array(sal_volumes)

    return sal_volumes

def _load_sal_vals(index):
    file_name = f"sal_volumes_{index}.npy"
    if os.path.isfile(file_name):
        return np.load(file_name)

def train_salti(fixations, stimuli, dataset = "SALICON"):
    model = get_nick_model()

    #inp, image_size = load_image("Datasets/ftp.ivc.polytech.univ-nantes.fr/Images/Stimuli/P12_10000x5000.jpg")
    '''
    inp = np.zeros([1, 3, 300, 600])
    out = model.predict(inp, batch_size = 1)
    '''
    #creating outputs
    sal_volumes = _create_sal_volumes(fixations, dataset, from_save = False)

    #creating inputs
    out = list()
    for i in range(stimuli.shape[0]):
        print(i)
        image_now = cv2.resize(stimuli[i], (600, 300), interpolation = cv2.INTER_CUBIC)
        out.append(image_now)
    stimuli = np.array(out)
    stimuli = np.array(stimuli, dtype=np.float32) / 255.0
    stimuli = np.moveaxis(stimuli, -1, 1)

    filepath = "nick_model.h5"
    ckpt = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only = False)

    model.fit(x=stimuli, y=sal_volumes, batch_size=16, epochs=100, verbose=1, callbacks=[ckpt])

    print(model.summary())
    print(sal_volumes.shape) 
