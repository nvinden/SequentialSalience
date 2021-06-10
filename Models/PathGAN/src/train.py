from . import models, predict, utils
from keras.models import *
from keras.optimizers import SGD, RMSprop
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import preprocess_input
#from keras.utils.training_utils import multi_gpu_model
import keras
import h5py
import numpy as np
import scipy.io as io
import tensorflow as tf
import argparse
import cv2
import os

debug = True
alpha_content_loss = 0.05
global_epoch_number = 0

def _create_seq_input_real(seq, dataset):
    #creates list of 8 real inputs
    #seq: (batch_size, number of observers, fixations, 5)
    #out: list of 8 (batch_size, 63, 4)
    if dataset == "OSIE":
        image_size = [600, 800]
    else:
        print("NO KNOW DATA TRANSFORMATION YET")
        raise Exception

    batch_size = seq.shape[0]

    out = []
    for i in range(8):
        curr = np.zeros([batch_size, 63, 4])
        in_batch = seq[:, i]
        for j in range(batch_size):
            real_fix = in_batch[j]
            curr[j, :real_fix.shape[0], :2] = real_fix[:, [1, 0]] / image_size
            curr[j, real_fix.shape[0] - 1, 3] = 1
        out.append(curr)

    return out

def _create_seq_input_gen(gen, stim_batch):
    out = []
    for i in range(8):
        noise  = np.random.normal(0,3, stim_batch.shape)
        noisy_stim_batch = stim_batch + noise
        _, x = gen.predict([noisy_stim_batch, stim_batch], batch_size=stim_batch.shape[0], verbose=0)
        out.append(x)
    return out
        


def train(seq, stim, stim_names, dataset):
    loss_weights            = [1., 0.05] #0.05
    adversarial_iteration   = 2
    batch_size              = 40 #100
    mini_batch_size         = 800 #4000
    G                       = 1
    epochs                  = 200
    n_hidden_gen            = 1000
    lr                      = 1e-4
    content_loss            = 'mse'
    lstm_activation         = 'tanh'
    dropout                 = 0.1
    dataset_path            = '/root/sharedfolder/predict_scanpaths/finetune_saltinet_isun/input/salient360_EVAL_noTime.hdf5'
    model360                = 'false'
    weights_generator       =  '-' #'Models/PathGAN/weights/generator_single_weights.h5'
    opt = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)

    if debug == True:
        batch_size = 10
        stim = stim[0:20, :, :, :]
        seq = seq[0:20, :]

    params = {
        'n_hidden_gen':n_hidden_gen,
        'lstm_activation':lstm_activation,
        'dropout':dropout,
        'optimizer':opt,
        'loss': content_loss,
        'weights':weights_generator,
        'G':G
    }

    gen_save_path = "gen.h5"
    dec_save_path = "dec.h5"
    

    #Creating batches of stimuli
    stim_batches = []
    for curr in stim:
        stim_batches.append(cv2.resize(curr, dsize = [224, 224]))
    stim_batches = np.array(stim_batches)
    stim_batches = preprocess_input(stim_batches)
    n_batches = stim_batches.shape[0] / batch_size
    stim_batches = np.array(np.split(stim_batches, n_batches))
    if stim_batches[-1].shape[0] != stim_batches[0].shape[0]:
        stim_batches.pop()
    stim_batches = np.array(stim_batches)

    #creating batches of sequences
    n_batches = seq.shape[0] / batch_size
    seq_batches = np.array(np.split(seq, n_batches))
    if seq_batches[-1].shape[0] != seq_batches[0].shape[0]:
        seq_batches.pop()
    seq_batches = np.array(seq_batches)

    if os.path.isfile(gen_save_path):
        gen = tf.keras.models.load_model(gen_save_path)
    else:
        gen, _ = models.generator(**params)

    if os.path.isfile(dec_save_path):
        dec = tf.keras.models.load_model(dec_save_path)
    else:
        dec = models.decoder(lstm_activation=lstm_activation, optimizer=opt, weights="-")

    params_gan = {
        'content_loss': content_loss,
        'optimizer': opt,
        'loss_weights': [1.0, 1.0],
        'generator': gen, 
        'decoder': dec,
        'G': G
    }

    for epoch in range(1, epochs + 1):
        global_epoch_number = epoch

        #Loading Models
        if epoch > 5:
            params_gan['loss_weights'] = [1, 0.5]
        else:
            params_gan['loss_weights'] = [0, 0.5]

        _, gen = models.generator(**params)
        if os.path.isfile(gen_save_path):
            gen.load_weights(gen_save_path)

        dec = models.decoder(lstm_activation=lstm_activation, optimizer=opt, weights="-")
        if os.path.isfile(dec_save_path):
            gen.load_weights(dec_save_path)

        _, gen_dec = models.gen_dec(**params_gan)

        for stim_batch, seq_batch in zip(stim_batches, seq_batches):
            #Creating Sequence Output
            seq_real = _create_seq_input_real(seq_batch, dataset)
            seq_gen = _create_seq_input_gen(gen_dec, stim_batch)

            real_dec_result = np.ones([batch_size, 63, 1])
            gen_dec_result = np.zeros([batch_size, 63, 1])

            '''

            #training discriminator
            for seq_real_curr in seq_real:
                dec.train_on_batch([seq_real_curr, stim_batch], real_dec_result)
            
            for seq_gen_curr in seq_gen:
                dec.train_on_batch([seq_gen_curr, stim_batch], gen_dec_result)

            '''

            #training generator
            for i in range(8):
                #Creating Generator Input
                noise  = np.random.normal(0,3, stim_batch.shape)
                noisy_stim_batch = stim_batch + noise
                outs = gen_dec.train_on_batch(x = [noisy_stim_batch, stim_batch], y = [gen_dec_result, seq_real[i]])
                print(outs)

            print("Batch Complete...")
    
    dec.save_weights(dec_save_path)
    gen.save_weights(gen_save_path)

    print(gen.summary())
    print(dec.summary())

    print(stim.shape)
    print(stim[0][0])