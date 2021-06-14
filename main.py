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


warnings.filterwarnings('ignore')


def main():
    ds_name = "OSIE"
    dataset = ds.SaliencyDataset(config=cfg.DATASET_CONFIG)
    dataset.load(ds_name)
    seq = dataset.get("sequence")
    stim = dataset.get("stimuli")
    stim_names = dataset.get("stimuli_path")

    #utils.test_IOR_ROI(seq, stim, stim_names, ds_name)
    train(seq = seq, stim = stim, stim_names = stim_names, dataset = ds_name)
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