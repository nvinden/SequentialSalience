import os

import Eval.dataset as ds
import Eval.metrics.metrics as met

from Models import *
from Models.Eymol import eymol

import warnings
warnings.filterwarnings('ignore')

import numpy as np

import json

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
        



def save_json(results, test, index, P, Q, pic_path, pic_shape, model_name):
  #results: (float64 2d matrix) numpy matrix size 15xnumber_of_tests with results to each of the tests
  #test: (string) which dataset used
  #index: (int) number used to identify which image it is in the dataset
  #model_name: (string) name of the model used to get the results
  #P: Array of predicted eye values for all 15.
  #Q: Actual ground truth of image
  #pic path: path to the pic used.
  #model_name: what model was used to achieve these results
  out = {}
  path = os.path.join(results_folder, model_name, str(1000 + index) + ".json")
  out["best_score"] = results.min(axis=0).tolist()
  out["all_scores"] = results.tolist()
  out["metrics"] = metrics
  out["dataset"] = test
  min_idx = np.argmin(results, axis = 0)
  predicted = P[min_idx[0]]
  actual = P[min_idx[1]]
  out["predicted"] = predicted.tolist()
  out["actual"] = actual.tolist()
  out["len_predicted"] = predicted.shape[0]
  out["len_actual"] = actual.shape[0]
  out["image_size"] = pic_shape
  out["p_bubbles"] = _create_close_values(predicted).tolist()
  out["q_bubbles"] = _create_close_values(actual).tolist()

  if not os.path.exists(os.path.join(results_folder, model_name)):
    os.makedirs(os.path.join(results_folder, model_name))

  with open(path, 'w') as f:
    json.dump(out, f)


def test_eymol():
  #testing on OSIE, SALICON
  tests = ["OSIE", "SUN09"]
  dataset = ds.SaliencyDataset(config=CONFIG)
  for test in tests:
    i = 0
    dataset.load(test)
    fixations = dataset.get('sequence')
    pictures = dataset.get('stimuli')
    pictures_path = dataset.get('stimuli_path')

    #result arrays for metric results, P's and Q's in total
    results = np.empty((0, len(tests)), dtype=np.float64)
    P = []
    Q = []

    for fix_batch, samp_pic, samp_pic_path in zip(fixations, pictures, pictures_path):
      i += 1
      for fix1 in fix_batch:
        gaze_positions = eymol.compute_simulated_scanpath(samp_pic, seconds=5)
        fix2 = eymol.get_fixations(gaze_positions)

        P_curr = fix1[:, 0:2]
        Q_curr = fix2[:, 0:2]

        out = run_diagnostics(P_curr, Q_curr, metrics)

        #concatenating results to result matix
        results = np.concatenate((results, np.expand_dims(out, axis=0)), axis=0)
        P.append(P_curr)
        Q.append(Q_curr)

      print(f"Run number: {i}")
      save_json(results, test, i, P, Q, samp_pic_path, samp_pic.shape, "Eymol")
    