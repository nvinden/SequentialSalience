import os

import torch

import Eval.dataset as ds
import Eval.metrics.metrics as met

from Models import *
from Models.Eymol import eymol

from Models.IRL.irl_dcb import config, builder, trainer, utils
from Models.IRL import dataset

import Models.SaltiNet.src as SaltiNet
import Models.SaltiNet.src.utils as s_utils
import Models.SaltiNet.src.pathnet as s_pathnet

import Models.PathGAN.src.predict as p_predict

import numpy as np

import json

import warnings
warnings.filterwarnings('ignore')

CONFIG = {
		'data_path' : 'Datasets/',
		'dataset_json': 'Eval/data/dataset.json',
		'auto_download' : True,
}

metrics = ["eyenalysis", "DTW"]

results_folder = "Results"

def run_diagnostics(val1, val2, metrics):
  #returns tuple of values in the same format
  #as inputted witht the results of the tests ran
  out = []
  for metric in metrics:
    #getting the certain metric from metric.py
    fun = getattr(met, metric)
    result = fun(val1, val2)
    out.append(result)
  return out

def _create_close_values(M, distances=[5, 10, 25, 50, 100, 200, 500]):
  #M: matrix of fixation points
  #distances: euclidean distances measures
  out = np.zeros(shape=len(distances))
  for i in range(M.shape[0]):
    for j in range(i + 1, M.shape[0]):
      dist = met.euclidean(M[i], M[j])
      for dist_idx, dist_max in enumerate(distances):
        if dist <= dist_max:
          out[dist_idx] += 1
  return out
        
def save_json(results, test, index, predicted, actual_list, pic_name, pic_shape, model_name):
  #results: (float64 2d matrix) numpy matrix size 15xnumber_of_tests with results to each of the tests
  #test: (string) which dataset used
  #index: (int) number used to identify which image it is in the dataset
  #model_name: (string) name of the model used to get the results
  #Predicted: Array of predicted eye values for all 15.
  #Actual: Actual ground truth of image
  #pic path: path to the pic used.
  #model_name: what model was used to achieve these results
  out = {}
  path = os.path.join(results_folder, model_name, test + "_" + pic_name + ".json")
  out["best_score"] = results.min(axis=0).tolist()
  out["all_scores"] = results.tolist()
  out["metrics"] = metrics
  out["dataset"] = test
  min_idx = np.argmin(results, axis = 0)
  actual = []
  for val in min_idx:
    actual.append(actual_list[val].tolist())
  out["predicted"] = predicted.tolist()
  out["actual"] = actual
  out["len_predicted"] = predicted.shape[0]
  len_actual_list = []
  for val in actual:
    len_actual_list.append(len(val))
  out["len_actual"] = len_actual_list
  out["image_size"] = pic_shape
  out["picture_name"] = pic_name

  if not os.path.exists(os.path.join(results_folder, model_name)):
    os.makedirs(os.path.join(results_folder, model_name))

  with open(path, 'w') as f:
    json.dump(out, f)


def test_eymol():
  #testing on OSIE, SALICON
  tests = ["SUN09", ]
  dataset = ds.SaliencyDataset(config=CONFIG)
  for test in tests:
      i = 0
      dataset.load(test)
      fixations = dataset.get('sequence')
      pictures = dataset.get('stimuli')
      pictures_path = dataset.get('stimuli_path')

      for fix_batch, samp_pic, samp_pic_path in zip(fixations, pictures, pictures_path):
        try:
          #result arrays for metric results, P's and Q's in total
          results = np.empty((0, len(metrics)), dtype=np.float64)
          actual_list = []
          i += 1
          gaze_positions = eymol.compute_simulated_scanpath(samp_pic, seconds=5)
          fix2 = eymol.get_fixations(gaze_positions)
          P_curr = fix2[:, 0:2]
          for fix1 in fix_batch:
            Q_curr = fix1[:, 0:2]

            out = run_diagnostics(P_curr, Q_curr, metrics)

            #concatenating results to result matix
            if test == "OSIE":
              results = np.concatenate((results, np.expand_dims(out, axis=0)), axis=0)
            elif test == "SUN09":
              out = np.array(out, dtype=np.float64)
              out = np.expand_dims(out, axis=0)
              results = np.concatenate((results, out), axis=0)
            actual_list.append(Q_curr)

          print(f"Run number: {i}")
          save_json(results, test, i, P_curr, actual_list, samp_pic_path.split("/")[-1], samp_pic.shape, "Eymol")
        except:
          print("Fail")

def _get_IRL_generator():
  if torch.cuda.is_available():
    device = "cuda:0"
  else:
    device = "cpu"

  hparams = "Models/IRL/hparams/coco_search18.json"
  dataset_root = "Models/IRL/dataset/"
  hparams = config.JsonConfig(hparams)

  # dir of pre-computed beliefs
  DCB_dir_HR = os.path.join(dataset_root, 'DCBs/HR/')
  DCB_dir_LR = os.path.join(dataset_root, 'DCBs/LR/')

  # bounding box of the target object (for search efficiency evaluation)
  bbox_annos = np.load(os.path.join(dataset_root, 'bbox_annos.npy'),
                        allow_pickle=True).item()

  # load ground-truth human scanpaths
  with open(os.path.join(dataset_root,
                  'coco_search18_fixations_TP_train.json')) as json_file:
      human_scanpaths_train = json.load(json_file)
  with open(os.path.join(dataset_root,
                  'coco_search18_fixations_TP_validation.json')) as json_file:
      human_scanpaths_valid = json.load(json_file)

  # exclude incorrect scanpaths
  if hparams.Train.exclude_wrong_trials:
      human_scanpaths_train = list(
          filter(lambda x: x['correct'] == 1, human_scanpaths_train))
      human_scanpaths_valid = list(
          filter(lambda x: x['correct'] == 1, human_scanpaths_valid))

  # process fixation data
  ds = dataset.process_data(human_scanpaths_train, human_scanpaths_valid,
                          DCB_dir_HR, DCB_dir_LR, bbox_annos, hparams)

  built = builder.build(hparams, True, device, ds['catIds'])
  tr = trainer.Trainer(**built, dataset=ds, device=device, hparams=hparams)

  env = built["env"]["valid"]
  generator = built["model"]["gen"]
  
  # generating scanpaths
  all_actions = []
  for i_sample in range(1):
      for batch in tr.valid_img_loader:
        try:
          tr.env_valid.set_data(batch)
          img_names_batch = batch['img_name']
          cat_names_batch = batch['cat_name']
          with torch.no_grad():
              tr.env_valid.reset()
              trajs = utils.collect_trajs(tr.env_valid,
                                          tr.generator,
                                          tr.patch_num,
                                          tr.max_traj_len,
                                          is_eval=True,
                                          sample_action=True)
              all_actions.extend([
                  (cat_names_batch[i], img_names_batch[i],
                    'present', trajs['actions'][:, i])
                  for i in range(tr.env_valid.batch_size)
              ])
        except FileNotFoundError:
          print("Could not find image")
  scanpaths = utils.actions2scanpaths(
      all_actions, tr.patch_num, tr.im_w, tr.im_h)
  utils.cutFixOnTarget(scanpaths, tr.bbox_annos)

  scanpaths = np.array(scanpaths)
  np.save("Models/IRL/results.npy", scanpaths, allow_pickle=True)
  return scanpaths

def test_IRL():
  if not os.path.exists("Models/IRL/results.npy"):
    gen_scanpaths = _get_IRL_generator()
  else:
    gen_scanpaths = np.load("Models/IRL/results.npy", allow_pickle=True)

  with open('Models/IRL/dataset/coco_search18_fixations_TP_validation.json') as f:
    real_scanpaths = json.load(f)

  i = 0
  for gen_image in gen_scanpaths:
    i += 1
    image_name = gen_image['name']
    number_of_paths = 0
    real_paths_list = []

    for real_image in real_scanpaths:
      if real_image['name'] == image_name:
        number_of_paths += 1
        real_paths_list.append(real_image)

    #run analysis on images iff there are 10 paths found in val set
    if number_of_paths == 10:
      try:
        results = np.empty((0, len(metrics)), dtype=np.float64)
        real_list = []

        gen_x = gen_image['X']
        gen_y = gen_image['Y']

        gen_x_npy = np.array(gen_x, dtype=np.float64)
        gen_x_npy = np.expand_dims(gen_x_npy, axis=1)

        gen_y_npy = np.array(gen_y, dtype=np.float64)
        gen_y_npy = np.expand_dims(gen_y_npy, axis=1)
          
        gen = np.concatenate((gen_x_npy, gen_y_npy), axis = 1)
        
        for real_path in real_paths_list:
          real_x = real_path["X"]
          real_y = real_path["Y"]

          real_x_npy = np.array(real_x, dtype=np.float64)
          real_x_npy = np.expand_dims(real_x_npy, axis=1)

          real_y_npy = np.array(real_y, dtype=np.float64)
          real_y_npy = np.expand_dims(real_y_npy, axis=1)
          
          real = np.concatenate((real_x_npy, real_y_npy), axis = 1)
          real_list.append(real)

          out = run_diagnostics(gen, real, metrics)
          results = np.concatenate((results, np.expand_dims(out, axis=0)), axis=0)
        
        save_json(results, "COCOSearch-18", i, gen, real_list, image_name, "Not shown", "IRL")
      except IndexError:
        print("Results could not be made")

def create_saltinet_csv():
  saltinet_dataset_route = "Datasets/ftp.ivc.polytech.univ-nantes.fr/Images/Stimuli"

  arr = []
  for i in range(98):
    arr.append(i + 1)
  
  s_pathnet.predict_and_save(saltinet_dataset_route)

def test_saltinet():
  saltinet_csv_dir = "Results/SaltiNet_csv"
  saltinet_image_dir = "Datasets/ftp.ivc.polytech.univ-nantes.fr/Images/Stimuli"

  for csv_name in os.listdir(saltinet_csv_dir):
    pass
  
def create_pathgan_csv():
  pathgan_dataset_route = "Datasets/ftp.ivc.polytech.univ-nantes.fr/Images/Stimuli"
  pathgan_results_route = "Results/PathGAN_csv/"

  p_predict.predict_and_save(pathgan_dataset_route, pathgan_results_route)

