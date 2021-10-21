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

target_city   = 'MELBOURNE'        # 'BERLIN' 'CHICAGO' 'ISTANBUL' 'MELBOURNE'             'NEWYORK'  'VIENNA'                          
load_model_id = 't1m1'                                                                 
load_model_path  = '../trained_models/' + load_model_id + '_' + target_city +'.pth'
input_test_data_filepath = '../../0_data/' + target_city + '/' + target_city + '_test_temporal.h5'           
test_save_folder_path = '../test_runs/' + load_model_id + '/' + target_city
test_save_filepath = test_save_folder_path + '/' + target_city + '_test_temporal.h5'
input_static_data_path   = '../../0_data/' + target_city + '/' + target_city + "_static.h5" 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SEED = 0
num_train_file = 180
num_frame_per_day = 288
num_frame_before = 12
num_frame_sequence = 24
num_frame_out = 6   
num_sequence_per_day = num_frame_per_day - num_frame_sequence + 1
height=495
width =436
num_channel=8
num_channel_out=8
num_channel_static = 9
visual_input_channels=105
visual_output_channels=48
NUM_INPUT_CHANNEL  = visual_input_channels
NUM_OUTPUT_CHANNEL = visual_output_channels
num_groups = 8
EPS = 1e-12
np.set_printoptions(precision=8)

def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data = data, dtype='uint8', compression='gzip', compression_opts=6)  # 9
    f.close()

class Deconv3x3Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Deconv3x3Block, self).__init__()
        self.add_module('deconv', nn.ConvTranspose2d(in_size, h_size, kernel_size=3, stride=2, padding=1, bias=True))
        self.add_module('elu',  nn.ELU(inplace=True))                                        
        self.add_module('norm', nn.GroupNorm(num_groups=num_groups, num_channels=h_size))    

class Conv1x1Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Conv1x1Block, self).__init__()
        self.add_module('conv', nn.Conv2d(in_size, h_size, kernel_size=1, stride=1, padding=0, bias=True))

class Conv3x3Block(nn.Sequential):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, ) -> None:
        super(Conv3x3Block, self).__init__()
        self.add_module('conv', nn.Conv2d(in_size, h_size, kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('elu',  nn.ELU(inplace=True))                                        
        self.add_module('norm', nn.GroupNorm(num_groups=num_groups, num_channels=h_size))    

class AvgBlock(nn.Sequential):
    def __init__(self, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int) -> None:
        super(AvgBlock, self).__init__()
        self.add_module('pool', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))    
        
class MaxBlock(nn.Sequential):
    def __init__(self, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int) -> None:
        super(MaxBlock, self).__init__()
        self.add_module('pool', nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))    

class DownBlock(nn.Module):
    def __init__(self, 
                 in_size: int, 
                 h_size: int, 
                 out_size: int, 
                 do_pool: int = True):
        
        super(DownBlock, self).__init__()     

        self.do_pool = do_pool

        in_size_cum = in_size  
        
        self.conv_1 = Conv3x3Block( in_size=in_size_cum, h_size=h_size)
        in_size_cum += h_size
        
        self.conv_3 = Conv3x3Block( in_size=in_size_cum, h_size=h_size)
        in_size_cum += h_size
        
        self.conv_2 = Conv1x1Block( in_size=in_size_cum,  h_size=out_size)

    def forward(self, x):
        
        batch_size = len(x)

        if self.do_pool:
          x = F.interpolate(x, scale_factor=0.7, mode='bilinear', align_corners=False, recompute_scale_factor=None)

        x_list = []
        x_list.append(x)
        
        x = self.conv_1(x)
        x_list.append(x)
        x = torch.cat(x_list, 1)
        
        x = self.conv_3(x)
        x_list.append(x)
        x = torch.cat(x_list, 1)
        
        x = self.conv_2(x)

        return x

    def cuda(self, ):
        super(DownBlock, self).cuda()
        self.conv_1.cuda()
        self.conv_3.cuda()
        self.conv_2.cuda()
        return self

class UpBlock(nn.Module):
  

    def __init__(self, 
                 in_size:   int, 
                 in_size_2: int, 
                 h_size:    int, 
                 out_size:  int, 
                 ):
        
        super(UpBlock, self).__init__()     
        self.deconv   = Conv3x3Block( in_size=in_size, h_size=h_size)
        self.out_conv = Conv3x3Block( in_size=h_size + in_size_2, h_size=out_size)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        x1 = F.interpolate(x1, size=x2.size()[2:4], scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None)
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)

    def cuda(self, ):
        super(UpBlock, self).cuda()   
        self.deconv.cuda()
        self.out_conv.cuda()
        return self

class NetA(nn.Module):

    def __init__(self,):
        super(NetA, self).__init__()

        self.block0 = DownBlock(in_size=NUM_INPUT_CHANNEL, h_size=128, out_size=128, do_pool=False)
        self.block1 = DownBlock(in_size=128, h_size=128, out_size=128,)
        self.block2 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block3 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block4 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block5 = DownBlock(in_size=128, h_size=128, out_size=128, )
        self.block6 = DownBlock(in_size=128, h_size=128, out_size=128,)
        self.block7 = DownBlock(in_size=128, h_size=128, out_size=128,)
        
        self.block20 = Conv3x3Block(in_size=128, h_size=128)
        
        self.block16 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block15 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block14 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block13 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block12 = UpBlock(in_size=128, in_size_2=128,  h_size=128,  out_size=128,)
        self.block11 = UpBlock(in_size=128, in_size_2=128 , h_size=128,  out_size=128,)
        self.block10 = UpBlock(in_size=128, in_size_2=128 , h_size=128,  out_size=128,)
        
        self.out_conv  = nn.Sequential(
           nn.Conv2d(128*1, NUM_OUTPUT_CHANNEL, kernel_size=3, stride=1, padding=1, bias=True)
        )
        
        if 1:
          for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                  nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                  nn.init.constant_(m.bias, 0)


    def forward(self, x):
      
        batch_size = len(x)
        
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)
        
        x  = self.block20(x7)
        
        x  = self.block16(x, x6)
        x  = self.block15(x, x5)
        x  = self.block14(x, x4)
        x  = self.block13(x, x3)
        x  = self.block12(x, x2)
        x  = self.block11(x, x1)
        x  = self.block10(x, x0)

        x  = self.out_conv(x)
        x  = torch.sigmoid(x)
        return x

    def cuda(self, ):
        super(NetA, self).cuda()
        
        self.block0.cuda()
        self.block1.cuda()
        self.block2.cuda()
        self.block3.cuda()
        self.block4.cuda()
        self.block5.cuda()
        self.block6.cuda()
        self.block7.cuda()
        
        self.block20.cuda()
        
        self.block16.cuda()
        self.block15.cuda()
        self.block14.cuda()
        self.block13.cuda()
        self.block12.cuda()
        self.block11.cuda()
        self.block10.cuda()
        
        self.out_conv.cuda()
        return self  



if __name__ == '__main__':

  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  
  torch.backends.cudnn.enabled       = True
  torch.backends.cudnn.benchmark     = True 
  torch.backends.cudnn.deterministic = True
  
  net = NetA().cuda()
  if 1:
    print('Loading ', load_model_path)
    state_dict = torch.load(load_model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict, strict=True)

  static_data = None                     
  if 1:
          file_path = input_static_data_path
          fr = h5py.File(file_path, 'r')
          a_group_key = list(fr.keys())[0]
          data = np.asarray(fr[a_group_key], np.uint8)  
          static_data = data[np.newaxis,:,:,:]
          static_data = static_data.astype(np.float32)
          static_data = static_data / 255.0

  try:
    if not os.path.exists(test_save_folder_path):
      os.makedirs(test_save_folder_path)
  except Exception:
    print('output path not made\t', test_save_folder_path)
    exit(-1)  

  test_data = None
  if 1:
    fr = h5py.File(input_test_data_filepath, 'r')
    a_group_key = list(fr.keys())[0]
    data = fr[a_group_key]
    test_data = np.array(data, np.float32)        
        
  prediction_list = list()
  batch_size_val = 1
  net.eval()    
  for i in range(len(test_data)):
    input_data = test_data[i,:,:,:,:][np.newaxis,:,:,:,:]
    input_data /= 255.0
    input_data = np.moveaxis(input_data, -1, 2).reshape((batch_size_val, -1, height, width))  
    input_data = np.concatenate(( input_data, np.repeat(static_data, batch_size_val, axis=0)), axis=1)
    input  = torch.from_numpy(input_data).float().cuda() 
    prediction = net(input)
    prediction_list.append(prediction.cpu().detach().numpy()) 

  prediction_list = np.concatenate(prediction_list, axis=0)        
  prediction_list = np.moveaxis(np.reshape(prediction_list, [-1, num_frame_out, num_channel_out, height, width]), 2, -1)  
  prediction_list *= 255.0
  prediction_list = np.rint(prediction_list)
  prediction_list = np.clip(prediction_list, 0.0, 255.0).astype(np.uint8)
  write_data(prediction_list, test_save_filepath)

