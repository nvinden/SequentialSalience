import utils

import os

import Eval.dataset as ds
import Eval.metrics.metrics as met

from Models import *
from Models.Eymol import eymol

import warnings
warnings.filterwarnings('ignore')


def main():
    utils.test_eymol()

if __name__ == '__main__':
    main()