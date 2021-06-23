# (c) Copyright 2017 Marc Assens. All Rights Reserved.

__author__ = "Marc Assens"
__version__ = "1.0"

import keras
#keras.backend.set_image_dim_ordering("th")
from keras.models import load_model

from scipy import ndimage
import scipy.io as io
import numpy as np
import json

from . import utils

import os

import warnings
warnings.simplefilter("ignore")

pretrained_path = "Models/SaltiNet/src/pathnet_model_v2.h5"

def get_model():
	model = load_model(pretrained_path)
	return model

def get_nick_model():
	if os.path.isfile("nick_model.h5"):
		model = load_model('nick_model.h5')
	elif os.path.isfile("nick_model_config.json"):
		#model = load_model('nick_model.h5')
		json_file = open("nick_model_config.json")
		json_config = json.load(json_file)
		model = keras.models.model_from_json(json_config)
	else:
		model = keras.Sequential()
		org_model = get_model()
		for layer in org_model.layers[:-3]:
			model.add(layer)
		model.add(keras.layers.UpSampling2D(size=(4, 4), data_format="channels_first"))
		model.add(keras.layers.Activation(keras.activations.sigmoid))

		'''
		json_config = model.to_json()
		with open('nick_model_config.json', 'w') as outfile:
			json.dump(json_config, outfile)
		'''

	optim = keras.optimizers.SGD(learning_rate=0.001)
	model.compile(optimizer=optim, loss = 'binary_crossentropy')

	return model



def sample_slice(pred, size, n_samples=1, use_heuristic=True, samples=[]):
	"""
		Takes random points from the slice
		taking into account the probabilities of each
		pixel
		
		Params:
			pred = 2d array with probability of each pixel
			
			size = (height, width) of the image
			
			n_samples = number of samples taken from this slice
	"""
	
	array_shape = (300, 600) 
	
	def num_to_pos(n, size):
		"""
			Convert an index of the flatten 2d image to a 
			coordenate (x, y)
			
			Params: 
				size = (height, width)
		"""
		# x and y pos
		x = n % array_shape[1]
		y = n / array_shape[1]
		return (x, y)
	
	# Normalize to predictions
	p = pred / np.sum(pred)
	# Flatten
	p = p.flatten()

	# Store the samples 
	#samples = []

	for i in range(n_samples):
		if samples and use_heuristic:
			gaussian = np.zeros((300, 600))
			gaussian[int(samples[-1][1]), int(samples[-1][0])] = 1
			gaussian = ndimage.filters.gaussian_filter(gaussian, [200, 200])
			p = pred / np.sum(pred)
			p =  p * gaussian
			p = p / np.sum(p)
			p = p.flatten()
			
			
		n = np.random.choice(array_shape[0] * array_shape[1], p=p)
		pos = num_to_pos(n, size=size)
		samples.append(pos)
		
	
	return samples

def sample_volume(vol, n_samples=24, size=(3000, 6000), use_heuristic=True):
	# Choose how many samples take per slice
	# i.e. 
	#     n_slices = 8 , n_samples = 12  => samples_per_slice = [2, 2, 2, 2, 1, 1, 1, 1]   
	n_slices = vol.shape[0]
	samples_per_slice = [n_samples / n_slices] * n_slices
	for i in range(n_samples % n_slices):
		samples_per_slice[i] += 1
		
		
	# Sample each slice of the volume
	samples = []
	for i, n_samples in enumerate(samples_per_slice):
		samples = sample_slice(vol[i], size, n_samples=int(n_samples), use_heuristic=use_heuristic, samples=samples)
	
	# Normalize positions
	array_shape = (300, 600)
	for i in range(len(samples)):
		x = int((float(samples[i][0]) / array_shape[1]) * size[1])
		y = int((float(samples[i][1]) / array_shape[0]) * size[0])
		samples[i] = (x, y)
		
	return samples


def predict(img_path):
	"""
		Predict 40 scanpaths given an image

		Params:
			image
	"""

	# Load image 
	img, img_size = utils.load_image(img_path)
	# Load model and predict volume
	model = get_model()
	print(model)
	preds = model.predict(img)


	scanpaths = []
	for i in range(40):
	    n_samples = utils.get_number_fixations()
	    s = sample_volume(preds[0], n_samples=n_samples , size=img_size, use_heuristic=True)

	    for j in range(n_samples):
	        # [user, index, time, x, y]
	        t = utils.get_duration_fixation()
	        pos_1s = [i+1, j+1, t, s[j][0], s[j][1]]
	        scanpaths.append(pos_1s)

	# Generate a np.array to output
	scanpaths_array = np.array(scanpaths, dtype=np.float32)

	return scanpaths_array

def predict_and_save(imgs_path):
	""" 
		Predicts multiple images and saves them in .mat format
		on an output path

		Param:
			img_path : path where the images are
			ids: list with image ids
			out_path: path where the .mat files will be saved

		i.e.:
			img_path = '/root/sharedfolder/360Salient/'
			ids =  [29, 31]
			out_path =  '/root/sharedfolder/360Salient/results/'
	"""

	# Preproces and load images

	path_to_save = "Results/SaltiNet_csv"

	paths = utils.paths_for_images(imgs_path)

	for i, path in enumerate(paths):
		path = os.path.join(imgs_path, path)
		print('Working on image %d of %d' % (i+1, len(paths)))

		print(path)

		# Predict the scanpaths
		scanpaths = predict(path)

		# Turn into a float np.array
		scanpaths = np.array(scanpaths, dtype=np.float32)

		# Save in output folder
		name = path.split("/")[-1].replace(".jpg", ".csv")
		path = os.path.join(path_to_save, name)
		np.savetxt(path, scanpaths)
		#io.savemat(out_path + '%s.mat' % name, {name: scanpaths})

		print('Saved scanpaths from image %s in file %s' % (path, name))

	print('Done!')
	return True
