from .pathnet import get_model, get_nick_model
from .utils import load_image

import numpy as np
from scipy import ndimage
from PIL import Image
import cv2
import tensorflow as tf
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

def _create_sal_volumes(fixations, save_index, dataset, from_save=True, to_save=True):
    if dataset in ["OSIE", ]:
        step_time = _LONGEST_OSIE_SCANPATH_DURATION / _QUANTIZATION_SLICES
    elif dataset in ["SALICON", ]:
        step_time = _LONGEST_SALICON_SCANPATH_DURATION / _QUANTIZATION_SLICES

    count = 0
    sal_volumes = list()
    for index, image in enumerate(fixations):
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
        
        print("Creating Sal Volumes: ", save_index, index)
        sal_volumes.append(curr_sal_vol)

    sal_volumes = np.array(sal_volumes)
    print(f"SAL VOL LENGTH: {sal_volumes.shape}")
    np.save(f"sal_volumes_{save_index}.npy", sal_volumes)

    return sal_volumes

def _load_sal_vals(index):
    file_name = f"sal_volumes_{index}.npy"
    if os.path.isfile(file_name):
        return np.load(file_name)
    return None

def train_salti():
    import Eval.dataset as ds
    import config as cfg

    dataset = ds.SaliencyDataset(config=cfg.DATASET_CONFIG)
    dataset.load("SALICON")

    model = get_nick_model()

    if os.path.isfile("nick_train_index.npy"):
        nick_train_staring_pos = np.load("nick_train_index.npy")[0]
    else:
        nick_train_staring_pos = 1

    if nick_train_staring_pos == 249:
        nick_train_staring_pos = 13

    print("STARING POSITION ", nick_train_staring_pos[0])

    #number of saved_states
    for i in range(nick_train_staring_pos, 21):
        start = (i - 1) * 250
        end = i * 250
        print(i, start, end)
        sal_volumes = _load_sal_vals(i)
        if sal_volumes is None:
            seq = dataset.get("sequence", index = range(start, end))
            sal_volumes = _create_sal_volumes(seq, i, "SALICON")
        stimuli = dataset.get("stimuli", index = range(start,end))
        print(stimuli.shape)
        print(sal_volumes.shape)

        #creating inputs
        out = list()
        for i in range(stimuli.shape[0]):
            print("Resizing:", i)
            image_now = cv2.resize(stimuli[i], (600, 300), interpolation = cv2.INTER_CUBIC)
            out.append(image_now)
        stimuli = np.array(out)
        stimuli = np.array(stimuli, dtype=np.float32) / 255.0
        stimuli = np.moveaxis(stimuli, -1, 1)

        filepath = "nick_model.h5"
        ckpt = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only = False)

        model.fit(x=stimuli, y=sal_volumes, batch_size=16, epochs=25, verbose=1, callbacks=[ckpt])

        np.save("nick_train_index.npy", np.array([i]))

