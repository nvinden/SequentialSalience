import os

import Metrics.dataset as ds
import Metrics.metrics.metrics as met

from Models import *
from Models.Eymol import eymol

import warnings
warnings.filterwarnings('ignore')

CONFIG = {
		'data_path' : 'Datasets/',
		'dataset_json': 'Metrics/data/dataset.json',
		'auto_download' : True,
}

metrics = ["eyenalysis", "DTW"]

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

def test_eymol():
  #testing on OSIE, SALICON
  tests = ["OSIE", "SUN09"]
  dataset = ds.SaliencyDataset(config=CONFIG)
  for test in tests:
    dataset.load(test)
    fixations = dataset.get('sequence')
    pictures = dataset.get('stimuli')
    for fix_batch, samp_pic in zip(fixations, pictures):
      for fix1 in fix_batch:
        gaze_positions = eymol.compute_simulated_scanpath(samp_pic, seconds=5)
        fix2 = eymol.get_fixations(gaze_positions)
        results = run_diagnostics(fix1[:, 0:2], fix2[:, 0:2], metrics)
        print(results)
        