from .pathnet import get_model, get_nick_model
from .utils import load_image

import numpy as np
from scipy import ndimage
from PIL import Image
import cv2
import keras

import os

_LONGEST_OSIE_SCANPATH_DURATION = 2657.0
_LONGEST_SALICON_SCANPATH_DURATION = 6000.0

_QUANTIZATION_SLICES = 12

_OSIE_IMAGE_SHAPE = [600, 800]
_SALICON_IMAGE_SHAPE = [480, 640]
_360_IMAGE_SHAPE = [300, 600]

def _create_sal_volumes(fixations, dataset, from_save=True, to_save=True):
    if os.path.isfile("sal_volumes.npy") and from_save:
        sal_volumes = np.load("sal_volumes.npy", allow_pickle = True)
        return sal_volumes

    if dataset in ["OSIE", ]:
        step_time = _LONGEST_OSIE_SCANPATH_DURATION / _QUANTIZATION_SLICES
    elif dataset in ["SALICON", ]:
        step_time = _LONGEST_SALICON_SCANPATH_DURATION / _QUANTIZATION_SLICES

    sal_volumes = list()
    for image in fixations:
        curr_sal_vol = np.zeros([12, 300, 600])
        for fix in image:
            if fix[-1, 2] > _LONGEST_SALICON_SCANPATH_DURATION:
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
                    if time_step == 12:
                        time_step -= 1
                    if height_val == 300:
                        height_val -= 1
                    if width_val == 600:
                        width_val -= 1
                    curr_sal_vol[time_step, height_val, width_val] = 1
        #saving salience volumes for testing
        curr_sal_vol = ndimage.gaussian_filter(curr_sal_vol, [4, 20, 20])
        for i, temp in enumerate(curr_sal_vol):
            time_sum = temp.sum()
            curr_sal_vol[i] /= time_sum

        for i in range(12):
            curr_sal_vol[i] /= np.amax(curr_sal_vol[i])
        
        sal_volumes.append(curr_sal_vol)

    sal_volumes = np.array(sal_volumes)
    if to_save:
        np.save("sal_volumes.npy", sal_volumes)
    return sal_volumes
                


def train_salti(fixations, stimuli, dataset = "SALICON"):
    model = get_nick_model()

    #inp, image_size = load_image("Datasets/ftp.ivc.polytech.univ-nantes.fr/Images/Stimuli/P12_10000x5000.jpg")
    '''
    inp = np.zeros([1, 3, 300, 600])
    out = model.predict(inp, batch_size = 1)
    '''
    #creating outputs
    sal_volumes = _create_sal_volumes(fixations, dataset)

    #creating inputs
    out = list()
    for i in range(stimuli.shape[0]):
        image_now = cv2.resize(stimuli[i], (600, 300), interpolation = cv2.INTER_CUBIC)
        out.append(image_now)
    stimuli = np.array(out)
    stimuli = np.array(stimuli, dtype=np.float32) / 255.0
    stimuli = np.moveaxis(stimuli, -1, 1)

    filepath = "nick_model.hdf5"
    ckpt = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)


    model.fit(x=stimuli, y=sal_volumes, batch_size=32, epochs=100, verbose=1, callbacks=[ckpt])

    print(model.summary())
    print(sal_volumes.shape) 
