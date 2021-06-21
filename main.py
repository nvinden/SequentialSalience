#import utils

import os
import numpy as np

import tensorflow as tf
import Eval.dataset as ds
import Eval.metrics.metrics as met

from Models import *
from Models.Eymol import eymol

import warnings

import config as cfg

from Models.PathGAN.src.train import train

from Models.SaltiNet.src.train import train_salti

from Models.SaltiNet.src.pathnet import get_model

from Models.SaltiNet.src.utils import load_image


warnings.filterwarnings('ignore')


def main():
    ds_name = "SALICON"
    dataset = ds.SaliencyDataset(config=cfg.DATASET_CONFIG)
    dataset.load(ds_name)
    '''
    dataset = ds.SaliencyDataset(config=cfg.DATASET_CONFIG)
    ds_names = dataset.dataset_names()
    for ds_name in ds_names:
        dataset.load(ds_name)
        print(dataset.data_type)
    '''

    seq = dataset.get("sequence", index = range(3000))
    stim = dataset.get("stimuli", index = range(3000))

    '''
    longest = 0
    for i, image in enumerate(seq):
        for j, person in enumerate(image):
            curr_longest = person[-1, 2]
            if curr_longest == float("inf"):
                print(i, j)
                continue
        
            if curr_longest > 6000:
                print(f"OVER 6000 {curr_longest}")

            if curr_longest > longest:
                print(curr_longest)
                longest = curr_longest
    print(longest)
    '''
    

    '''
    seq = dataset.get("sequence")
    stim = dataset.get("stimuli")
    stim_names = dataset.get("stimuli_path")
    heat = dataset.get("heatmap")
    fix_t = dataset.get("fixation_time")
    fix_dw = dataset.get("fixation_dw")
    '''

    train_salti(seq, stim)

    #utils.test_IOR_ROI(seq, stim, stim_names, ds_name)
    #train(seq = seq, stim = stim, stim_names = stim_names, dataset = ds_name)
    '''
    dataset = ds.SaliencyDataset(config=cfg.DATASET_CONFIG)
    for dset in dataset.dataset_names():
        if dset == "PASCAL-KYUN":
            continue
        print(dset)
        dataset.load(dset)
        seq = dataset.get("sequence")
        print(f"Dataset: {dset} is shape {seq.shape}")
    '''

    '''
    seq_dataset_list = ("OSIE", "SUN09", "LOWRES", "KTH")
    dataset = ds.SaliencyDataset(config=cfg.DATASET_CONFIG)

    for ds_name in seq_dataset_list:
        if ds_name in ["OSIE", ]:
            continue
        dataset.load(ds_name)
        seq = dataset.get("sequence")
        stim = dataset.get("stimuli")
        stim_names = dataset.get("stimuli_path")
        utils.test_itti_koch(seq, stim, stim_names, ds_name)
    
    for ds_name in seq_dataset_list:
        dataset.load(ds_name)

        seq = dataset.get("sequence")
        stim = dataset.get("stimuli")
        stim_names = dataset.get("stimuli_path")
        utils.test_eymol(seq, stim, stim_names, ds_name)
    '''



if __name__ == '__main__':
    main()