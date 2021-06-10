import os

import torch

import Eval.dataset as ds
import Eval.metrics.metrics as met

from Models import *
from Models.Eymol import eymol

'''
from Models.IRL.irl_dcb import config, builder, trainer, utils
from Models.IRL import dataset
'''

import Models.SaltiNet.src as SaltiNet
import Models.SaltiNet.src.utils as s_utils
import Models.SaltiNet.src.pathnet as s_pathnet

import Models.PathGAN.src.predict as p_predict

import Models.IttiKoch.pySaliencyMap as ik

import numpy as np
import cv2
from PIL import Image

import json
import csv
import math

import warnings
warnings.filterwarnings('ignore')

CONFIG = {
		'data_path' : 'Datasets/',
		'dataset_json': 'Eval/data/dataset.json',
		'auto_download' : True,
}

metrics = ["eyenalysis", "DTW"]

results_folder = "Results"

fixation_cutoff = 12

def run_diagnostics(val1, val2):
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
        
def save_json(results, test, index, predicted, actual_list, pic_name, pic_shape, model_name, best_2_best_15=None, first_2_best_100=None, best_2_best_100=None):
  #results: (float64 2d matrix) numpy matrix size 15xnumber_of_tests with results to each of the tests
  #test: (string) which dataset used
  #index: (int) number used to identify which image it is in the dataset
  #model_name: (string) name of the model used to get the results
  #Predicted: Array of predicted eye values for all 15.
  #Actual: Actual ground truth of image
  #pic path: path to the pic used.
  #model_name: what model was used to achieve these results
  out = {}

  #creating pathname for the json file
  if model_name == "IttiKoch":
    path = os.path.join(results_folder, model_name, test + "_"  + str(predicted.shape[0]) + "_" + pic_name + ".json")
  else:
    path = os.path.join(results_folder, model_name, test + "_" + pic_name + ".json")
  
  out["best_score"] = results.min(axis=0).tolist()
  out["all_scores"] = results.tolist()
  out["metrics"] = metrics
  out["dataset"] = test
  min_idx = np.argmin(results, axis = 0)
  actual = []
  for val in min_idx:
    curr = actual_list[val]
    if curr.shape[1] == 5:
      curr = curr[:, 0:2]
      curr = curr[:, [1,0]]
    actual.append(curr.tolist())
  out["predicted"] = predicted.tolist()
  out["actual"] = actual
  out["len_predicted"] = predicted.shape[0]
  len_actual_list = []
  for val in actual:
    len_actual_list.append(len(val))
  out["len_actual"] = len_actual_list
  out["image_size"] = pic_shape
  out["picture_name"] = pic_name

  if best_2_best_15 is not None:
    out["best_2_best_15"] = best_2_best_15.tolist()

  if first_2_best_100 is not None:
    out["first_2_best_100"] = first_2_best_100.tolist()

  if best_2_best_100 is not None:
    out["best_2_best_100"] = best_2_best_100.tolist()

  if not os.path.exists(os.path.join(results_folder, model_name)):
    os.makedirs(os.path.join(results_folder, model_name))

  #added comment

  with open(path, 'w') as f:
    json.dump(out, f)

def fix_to_json(pred, real_list, image = None, dataset = None, name = None, directory = None, to_image = False):
  #pred: length x 2 dimension predicted fixations
  #real_list: list of length x 5 or 2 dimension real fixation
  #image: numpy image
  #name: name of the file
  results = []
  for real in real_list:
    if dataset in ["OSIE", "SUN09", "LOWRES", "KTH"]:
      real = real[:, 0:2]
      real = real[:, [1,0]]

    if to_image == True:
      print("NICK NEEDS TO IMPLEMENT THIS LATER")

    if real.shape[0] == 1 or pred.shape[0] == 1:
      continue

    try:
      curr_result = run_diagnostics(pred, real)
      results.append(curr_result)
    except:
      print("fix_to_json failure")
    
  save_json(np.array(results), dataset, 0, pred, real_list, name.split("/")[-1].split(".")[0], image.shape, directory)

def create_csv_from_JSON_directory(directory, name):
  csv_dir = "Results/csv"
  with open(os.path.join(csv_dir, name + ".csv"), 'w', newline='') as csvfile:
    write = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    write.writerow(["EyeAnalysis", "DTW", "Dataset", "len_predicted", "len_actual_Eye", "len_actual_DTW", "width", "height", "best_2_best_15", "first_2_best_100", "best_2_best_100"])
    for filename in os.listdir(directory):
      full_path = os.path.join(directory, filename)
      with open(full_path) as f:
        curr = json.load(f)
      EyeAnaylsis = curr["best_score"][0]
      DTW = curr["best_score"][1]
      Dataset = curr["dataset"]
      len_predicted = curr["len_predicted"]
      len_actual_Eye = curr["len_actual"][0]
      len_actual_DTW = curr["len_actual"][1]

      #for 360 salient
      if curr["image_size"] == "Not shown":
        width = "n/a"
        height = "n/a"
      else:
        width = curr["image_size"][0]
        height = curr["image_size"][1]

      if "best_2_best_15" in curr.keys():
        print(curr["best_2_best_15"])
        best_2_best_15 = curr["best_2_best_15"][0][1]
      else:
        best_2_best_15 = "n/a"

      if "first_2_best_100" in curr.keys():
        first_2_best_100 = curr["first_2_best_100"][0][1]
      else:
        first_2_best_100 = "n/a"

      if "best_2_best_100" in curr.keys():
        best_2_best_100 = curr["best_2_best_100"][0][1]
      else:
        best_2_best_100 = "n/a"
      
      write.writerow([EyeAnaylsis, DTW, Dataset, len_predicted, len_actual_Eye, len_actual_DTW, width, height, best_2_best_15, first_2_best_100, best_2_best_100])
    
def create_size_and_length_compare_csv(directory, name):
  #creates a csv file containing info about the DTW and
  #after morphing images size and gazepath length to see how they
  #effect the DTW and Eyeanalysis scores
  #directory: directory that contains the JSONs to load
  #name: Name of the out file saved in your Results csv folder
  np.set_printoptions(suppress=True) 

  csvfile = os.path.join("Results/csv", name + ".csv")

  with open(csvfile, 'w', newline='') as f:
    write = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    write.writerow(["Filename", "EyeAnalysis", "DTW", "image_height", "image_width", "image_area", "len_actual", "len_predicted", "total_length"])

    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      with open(path) as f:
          curr = json.load(f)
      image_size = curr['image_size']
      pred = np.array(curr['predicted'], dtype=np.float64)[:,3:5]
      pred = pred[:,[1,0]]
      pred[:,0] = pred[:,0] / image_size[1]
      pred[:,1] = pred[:,1] / image_size[0]

      actu = np.array(curr['actual'][1], dtype=np.float64)[:,1:3]

      for i in range(1, 41):
        scaling_factor_height = i / 20
        scaling_factor_width = i / 20

        for j in range(1, 5):
          length_scaling_factor_pred = math.ceil(j / 4 * pred.shape[0])
          length_scaling_factor_actu = math.ceil(j / 4 * actu.shape[0])

          pred_used = np.copy(pred[0:length_scaling_factor_pred,:])
          actu_used = np.copy(actu[0:length_scaling_factor_actu,:])

          pred_used[:,0] = pred_used[:,0] * scaling_factor_height * image_size[1]
          pred_used[:,1] = pred_used[:,1] * scaling_factor_width * image_size[0]

          actu_used[:,0] = actu_used[:,0] * scaling_factor_height * image_size[1]
          actu_used[:,1] = actu_used[:,1] * scaling_factor_width * image_size[0]

          if pred_used.shape[0] != 1 and actu_used.shape[0]:
            try:
              result = run_diagnostics(pred_used, actu_used)
              write.writerow([filename, result[0], result[1], scaling_factor_height * image_size[1], scaling_factor_width * image_size[0], scaling_factor_height * image_size[1] * scaling_factor_width * image_size[0], actu_used.shape[0], pred_used.shape[0], actu_used.shape[0]+ pred_used.shape[0]])
            except:
              print("Failure")
              print(pred_used)
              print(actu_used)



def test_eymol(seq, stim, stim_names, dataset):
  i = 0
  fixations = seq
  pictures = stim
  pictures_path = stim_names
  test = dataset

  for fix_batch, samp_pic, samp_pic_path in zip(fixations, pictures, pictures_path):
    #try:
    #result arrays for metric results, P's and Q's in total
    results = np.empty((0, len(metrics)), dtype=np.float64)
    actual_list = []
    i += 1
    gaze_positions = eymol.compute_simulated_scanpath(samp_pic, seconds=5)
    fix2 = eymol.get_fixations(gaze_positions)
    P_curr = fix2[:, 0:2]
    for fix1 in fix_batch:
      Q_curr = fix1[:, 0:2]

      out = run_diagnostics(P_curr, Q_curr)

      #concatenating results to result matix
      if test in ["KTH", "OSIE", "LOWRES"]:
        results = np.concatenate((results, np.expand_dims(out, axis=0)), axis=0)
      elif test == "SUN09":
        out = np.array(out, dtype=np.float64)
        out = np.expand_dims(out, axis=0)
        results = np.concatenate((results, out), axis=0)
      actual_list.append(Q_curr)

    print(f"Run number: {i}")
    save_json(results, test, i, P_curr, actual_list, samp_pic_path.split("/")[-1].split(".")[0], samp_pic.shape, "Eymol")
    #except:
      #print("Fail")
      #dont maintain this

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

def _get_best_from_fixation(pred, real_list, shape, range_of_fixations):
  best_results = np.full((len(metrics)), np.inf)
  for i, real_ind in enumerate(real_list):
    real = np.copy(real_ind)

    if i + 1 >= range_of_fixations:
      break
    #creating entry for real fixations (note shape is flipped because widthxheight needs to
    # translate into longxlat)
    real_fix = real[:,1:3]
    real_fix[:,0] = real_fix[:,0] * shape[1]
    real_fix[:,1] = real_fix[:,1] * shape[0]

    #creating entry for predicted fixations
    pred_fix = pred[:,3:5]
    pred_fix = pred_fix[:,[1,0]]

    try:
      results = run_diagnostics(real_fix, pred_fix)

      for result_idx in range(len(best_results)):
        if best_results[result_idx] > results[result_idx]:
          best_results[result_idx] = results[result_idx]
    except:
      continue
    
  return np.array(best_results, dtype=np.float64)

def test_saltinet():
  np.set_printoptions(suppress=True) 
  saltinet_pred_dir = "Results/SaltiNet_csv"
  saltinet_real_dir = "Datasets/ftp.ivc.polytech.univ-nantes.fr/Images/H/Scanpaths"

  for csv_int, csv_name in enumerate(os.listdir(saltinet_pred_dir)):
    best_2_best_15 = np.empty((0, len(metrics)), dtype=np.float64)
    first_2_best_15 = np.empty((0, len(metrics)), dtype=np.float64)
    first_2_best_100 = np.empty((0, len(metrics)), dtype=np.float64)
    best_2_best_100 = np.empty((0, len(metrics)), dtype=np.float64)

    csv_path = os.path.join(saltinet_pred_dir, csv_name)
    pred = np.genfromtxt(csv_path, dtype=np.float64)
    pred_list = np.split(pred, np.where(np.diff(pred[:,0]))[0]+1)

    file_no = csv_name.split("_")[0][1:]
    csv_path_real = os.path.join(saltinet_real_dir, "Hscanpath_" + file_no + ".txt")
    real = np.genfromtxt(csv_path_real, delimiter=",", dtype=np.float64)
    real = real[1:, :]
    real_list = []

    shape = csv_name.split("_")[-1].split(".")[0].split("x")
    shape = [int(dim) for dim in shape]

    for i in range(0, real.shape[0], 100):
      real_list.append(real[i:i+fixation_cutoff,:])

    #calculating data for the first predicted against the real fixations (15 and 100)
    first_fix = pred_list[0]

    first_best_15_result = _get_best_from_fixation(first_fix, real_list, shape, 15)
    first_best_100_result = _get_best_from_fixation(first_fix, real_list, shape, 100)

    first_2_best_15 = np.concatenate((first_2_best_15, np.expand_dims(first_best_15_result, axis=0)), axis=0)
    first_2_best_100 = np.concatenate((first_2_best_100, np.expand_dims(first_best_100_result, axis=0)), axis=0)

    #calculating data by comparing the best generated against the best real fixations (15 and 100)
    best_15 = first_best_15_result
    best_100 = first_best_100_result
    for fix_idx, fixation in enumerate(pred_list):
      if fix_idx == 0:
        continue

      first_best_15_result = _get_best_from_fixation(fixation, real_list, shape, 15)
      first_best_100_result = _get_best_from_fixation(fixation, real_list, shape, 100)

      if best_15[1] > first_best_15_result[1]:
        best_15 = first_best_15_result
      
      if best_100[1] > first_best_100_result[1]:
        best_100 = first_best_100_result
    
    best_2_best_15 = np.concatenate((best_2_best_15, np.expand_dims(best_15, axis=0)), axis=0)
    best_2_best_100 = np.concatenate((best_2_best_100, np.expand_dims(best_100, axis=0)), axis=0)

    print(first_2_best_15)
    print(best_2_best_15)
    print(first_2_best_100)
    print(best_2_best_100)
  
    save_json(first_2_best_15, "COCOSearch-18", csv_int, first_fix, real_list[:15], csv_name, shape, "SaltiNet", first_2_best_100=first_2_best_100[0][1], best_2_best_15=best_2_best_15[0][1], best_2_best_100=best_2_best_100[0][1])


  
def create_pathgan_csv():
  pathgan_dataset_route = "Datasets/ftp.ivc.polytech.univ-nantes.fr/Images/Stimuli"
  pathgan_results_route = "Results/PathGAN_csv/"

  p_predict.predict_and_save(pathgan_dataset_route, pathgan_results_route)

def _create_circle_on_image(image, loc, r):
  start_height = int(loc[0] - r)
  end_height = int(loc[0] + r)
  start_width = int(loc[1] - r)
  end_width = int(loc[1] + r)
  for h in range(int(-r), int(r)):
    for w in range(int(-r), int(r)):
      try:
        if math.sqrt(h**2 + w**2) <= r:
          h_ind = loc[0] + h
          w_ind = loc[1] + w
          if h_ind > image.shape[0] or h_ind < 0 or w_ind > image.shape[1] or w_ind < 0:
            continue
          image[h_ind, w_ind] = 0
      except IndexError: 
        continue
  return image


def test_itti_koch(seq, stim, stim_names, dataset):
  iteration_no = 0
  for image, curr_seq, curr_name in zip(stim, seq, stim_names):
    n_fix= [4,8,12]
    max_fix = n_fix[-1]
    pred = []
    img_width  = image.shape[0]
    img_height = image.shape[1]
    sm = ik.pySaliencyMap(img_width, img_height)
    sal_map = sm.SMGetSM(image) * 255

    r = math.sqrt((0.065 * img_width * img_height)/math.pi)
    for i in range(max_fix):
      #finding max index of sal map
      maxidx = sal_map.argmax()
      maxidx = np.unravel_index(maxidx, sal_map.shape)

      pred.append(maxidx)

      sal_map = _create_circle_on_image(sal_map, maxidx, r)

      if i + 1 == n_fix[0]:
        fix = i + 1
        n_fix.pop(0)
        fix_to_json(np.array(pred), curr_seq, image = image, name = curr_name, directory = "IttiKoch", dataset = dataset)

    iteration_no += 1
    print(f"Iteration {iteration_no} for {dataset}")

