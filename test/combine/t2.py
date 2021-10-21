import random
from random import shuffle
import numpy as np
from datetime import datetime
import time
import queue
import threading
import logging
from PIL import Image
import itertools
import re
import os
import glob
import shutil
import sys
import copy
import h5py
from typing import Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor

target_city_list = ['NEWYORK',  'VIENNA']
file_last_line = '_test_spatiotemporal.h5'
save_model_id = 't2'
load_model_id_list = ['t2m1', 't2m2', 't2m3', 't2m4', ]
assert len(load_model_id_list) > 1

def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, dtype='uint8', compression='gzip', compression_opts=6)
    f.close()

for target_city in target_city_list:
  test_save_folder_path = '../test_runs/' + save_model_id + '/' + target_city
  test_save_filepath = test_save_folder_path + '/' + target_city + file_last_line
  
  try:
    if not os.path.exists(test_save_folder_path):
      os.makedirs(test_save_folder_path)
  except Exception:
    print('output path not made\t', test_save_folder_path)
    exit(-1)  
  
  test_data_sum = None
  for i, load_model_id in enumerate(load_model_id_list):
    input_test_data_filepath = '../test_runs/' + load_model_id + '/' + target_city + '/' + target_city + file_last_line           
    test_data = None
    if 1:
      fr = h5py.File(input_test_data_filepath, 'r')
      a_group_key = list(fr.keys())[0]
      data = fr[a_group_key]         
      test_data = np.array(data, np.float32)        
      if i == 0:
        test_data_sum = test_data.copy()
      else:
        test_data_sum += test_data
  prediction_list = np.rint(test_data_sum/float(len(load_model_id_list)))
  prediction_list = np.clip(prediction_list, 0.0, 255.0).astype(np.uint8)
  write_data(prediction_list, test_save_filepath)
