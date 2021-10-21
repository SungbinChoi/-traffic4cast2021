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



target_city = 'ANTWERP'
other_city_list = ['ANTWERP', 'BANGKOK', 'BARCELONA', 'MOSCOW', 'BERLIN', 'CHICAGO', 'ISTANBUL', 'MELBOURNE', ]


input_train_data_folder_path = '../../0_data/' + target_city + '/'  + 'training'
input_static_data_path       = '../../0_data/' + target_city + '/'  + target_city + "_static.h5"
out_dir                      = 'output'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SEED = int(time.time())
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
vector_input_channels=1     
num_epoch_to_train = 100000000
save_per_iteration = 5000
global_step_start  = 0        
initial_checkpoint = None
initial_checkpoint_optimizer = None
LEARNING_RATE  = 3e-4
batch_size = 2
batch_size_val = 1   
num_thread=2
num_groups = 8
EPS = 1e-12
np.set_printoptions(precision=8)
NUM_INPUT_CHANNEL  = visual_input_channels
NUM_OUTPUT_CHANNEL = visual_output_channels

def get_data_filepath_list_by_year(input_data_folder_path):
  data_filepath_list_1 = []
  data_filepath_list_2 = []
  for filename in os.listdir(input_data_folder_path):
    if filename.split('.')[-1] != 'h5':     
      continue
    if filename.startswith('2019'):
      data_filepath_list_1.append(os.path.join(input_data_folder_path, filename))
    elif filename.startswith('2020'):
      data_filepath_list_2.append(os.path.join(input_data_folder_path, filename))
    else:
      print('Error - Unknown data year\t', filename)
      exit(-1)
  data_filepath_list_1 = sorted(data_filepath_list_1)
  data_filepath_list_2 = sorted(data_filepath_list_2)
  return data_filepath_list_1, data_filepath_list_2

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

        self.out_conv  = nn.Sequential(nn.Conv2d(128*1, NUM_OUTPUT_CHANNEL, kernel_size=3, stride=1, padding=1, bias=True))
        
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
  
  if initial_checkpoint == None:
    assert global_step_start == 0 
  else:
    assert global_step_start > 0 

  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)
  torch.backends.cudnn.enabled       = True
  torch.backends.cudnn.benchmark     = True 
  torch.backends.cudnn.deterministic = False 

  try:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
  except Exception:
            print('out_dir not made')

  net = NetA().cuda()
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=LEARNING_RATE)
  loss_func2 = nn.MSELoss()           
  if initial_checkpoint is not None:
    print('Loading ', initial_checkpoint)
    state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict, strict=True)
    optimizer_state_dict_ = torch.load(initial_checkpoint_optimizer, map_location=lambda storage, loc: storage)
    optimizer_state_dict = optimizer_state_dict_['optimizer']
    optimizer.load_state_dict(optimizer_state_dict)

  static_data = None
  if 1:
          file_path = input_static_data_path
          fr = h5py.File(file_path, 'r')
          a_group_key = list(fr.keys())[0]
          data = np.asarray(fr[a_group_key], np.uint8)
          static_data = data[np.newaxis,:,:,:]
          static_data = static_data.astype(np.float32)
          static_data = static_data / 255.0
  static_data_list = []
  if 1:
        for other_city in other_city_list:
          file_path = '../../0_data/' + other_city + '/'  + other_city + "_static.h5"
          fr = h5py.File(file_path, 'r')
          a_group_key = list(fr.keys())[0]
          data = np.asarray(fr[a_group_key], np.uint8)
          static_data_ = data[np.newaxis,:,:,:]
          static_data_ = static_data_.astype(np.float32)
          static_data_ = static_data_ / 255.0
          static_data_list.append(static_data_)

  train_static_data_index_list = []
  train_data_filepath_list, val_data_filepath_list = get_data_filepath_list_by_year(input_train_data_folder_path)
  target_city_i = other_city_list.index(target_city)
  for _ in range(len(train_data_filepath_list)):
    train_static_data_index_list.append(target_city_i)
  for o, other_city in enumerate(other_city_list):
    if o == target_city_i:
      continue
    train_data_filepath_list_one, _ = get_data_filepath_list_by_year('../../0_data/' + other_city + '/'  + 'training')
    for _ in range(len(train_data_filepath_list_one)):
      train_static_data_index_list.append(o)
    train_data_filepath_list += train_data_filepath_list_one

  train_set = []       
  for i in range(len(train_data_filepath_list)):
    for j in range(num_sequence_per_day):
      train_set.append( (i,j) )
  num_iteration_per_epoch = int(len(train_set) / batch_size)
  print('num_iteration_per_epoch:', num_iteration_per_epoch)
  assert num_iteration_per_epoch > 10
  val_set = []
  val_skip_k = 0
  val_skip_ratio = 5
  for i in range(len(val_data_filepath_list)):
    for j in range(0, num_sequence_per_day, num_frame_sequence):
      val_skip_k += 1
      if val_skip_k % val_skip_ratio == 0:
        val_set.append( (i,j) )
  num_val_iteration_per_epoch = int(len(val_set) / batch_size_val)    
  print('num_val_iteration_per_epoch:', num_val_iteration_per_epoch)

          
  train_input_queue  = queue.Queue()
  train_output_queue = queue.Queue()
  def load_train_multithread():
    while True:
      if train_input_queue.empty() or train_output_queue.qsize() > 8:
        time.sleep(0.1)
        continue
      i_j_list = train_input_queue.get()
      train_orig_data_batch_list  = []
      train_data_batch_list = []  
      train_data_mask_list = [] 
      train_stat_batch_list = [] 
      train_static_data_batch_list = []
      for train_i_j in i_j_list:
          (i,j) = train_i_j
          file_path = train_data_filepath_list[i]
          train_static_data_batch_list.append(static_data_list[train_static_data_index_list[i]])
          fr = h5py.File(file_path, 'r')
          a_group_key = list(fr.keys())[0]
          data = fr[a_group_key]                               
          train_data_batch_list.append(data[j:j+num_frame_sequence,:,:,:][np.newaxis,:,:,:,:])                            
      train_data_batch = np.concatenate(train_data_batch_list, axis=0)
      train_static_data_batch = np.concatenate(train_static_data_batch_list,axis=0)
      input_data = train_data_batch[:,:num_frame_before ,:,:,:]                
      orig_label = train_data_batch[:, num_frame_before:,:,:,:num_channel_out] 
      true_label = np.concatenate((orig_label[:, 0:3, :,:,:],  orig_label[:, 5::3,:,:,:] ), axis=1)
      input_data = input_data.astype(np.float32)
      true_label = true_label.astype(np.float32)
      input_data = input_data / 255.0
      true_label = true_label / 255.0
      
      flip_dr = np.random.randint(0,2)
      if flip_dr == 1:
        input_data_flipped = copy.deepcopy(input_data) 
        input_data_flipped[:,:,:,:,4:8] = input_data[:,:,:,:,0:4]
        input_data_flipped[:,:,:,:,0:4] = input_data[:,:,:,:,4:8]
        input_data = input_data_flipped[:,:,::-1,::-1,:]
        true_label_flipped = copy.deepcopy(true_label)
        true_label_flipped[:,:,:,:,4:8] = true_label[:,:,:,:,0:4]
        true_label_flipped[:,:,:,:,0:4] = true_label[:,:,:,:,4:8]
        true_label = true_label_flipped[:,:,::-1,::-1,:]        
        train_static_data_batch_flipped = copy.deepcopy(train_static_data_batch)
        train_static_data_batch_flipped[:,5:9,:,:] = train_static_data_batch[:,1:5,:,:]
        train_static_data_batch_flipped[:,1:5,:,:] = train_static_data_batch[:,5:9,:,:]
        train_static_data_batch = train_static_data_batch_flipped[:,:,::-1,::-1]

      input_data = np.moveaxis(input_data, -1, 2).reshape((batch_size, -1, height, width))  
      true_label = np.moveaxis(true_label, -1, 2).reshape((batch_size, -1, height, width))  
      input_data = np.concatenate((input_data, train_static_data_batch), axis=1)
      train_output_queue.put( (input_data, true_label) )
  thread_list = []
  assert num_thread > 0
  for i in range(num_thread):
    t = threading.Thread(target=load_train_multithread)
    t.start()
                    
  
  net.train()   
  sum_train_loss = 0.0
  sum_train_iter = 0
  global_step = global_step_start
  for epoch in range(num_epoch_to_train):
    np.random.shuffle(train_set)
    for a in range(num_iteration_per_epoch):
      i_j_list = []      
      for train_i_j in train_set[a * batch_size : (a+1) * batch_size]:
        i_j_list.append(train_i_j)
      train_input_queue.put(i_j_list)
      
    for a in range(num_iteration_per_epoch):

      if global_step % save_per_iteration == 0:
        net.eval()
        state_dict_0 = copy.deepcopy(net.state_dict())
        torch.save(state_dict_0, out_dir + '/%09d_model.pth' % (global_step))
        torch.save(
              {
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
              }, 
              out_dir + '/%09d_optimizer.pth' % (global_step))  
        
        eval_loss_list = list()
        eval_loss_list = [0]
        with torch.no_grad():
         for a in range(num_val_iteration_per_epoch):
          val_orig_data_batch_list  = []
          val_data_batch_list = []   
          val_data_mask_list = [] 
          val_stat_batch_list = []   
          for i_j in val_set[a * batch_size_val : (a+1) * batch_size_val]:
            (i,j) = i_j
            file_path = val_data_filepath_list[i]
            fr = h5py.File(file_path, 'r')
            a_group_key = list(fr.keys())[0]
            data = fr[a_group_key]
            val_data_batch_list.append(data[j:j+num_frame_sequence,:,:,:][np.newaxis,:,:,:,:])
          val_data_batch = np.concatenate(val_data_batch_list, axis=0)
          input_data = val_data_batch[:,:num_frame_before ,:,:,:]                 
          orig_label = val_data_batch[:, num_frame_before:,:,:,:num_channel_out]  
          true_label = np.concatenate((orig_label[:, 0:3, :,:,:], orig_label[:, 5::3,:,:,:]), axis=1)
          input_data = input_data.astype(np.float32)
          true_label = true_label.astype(np.float32)
          input_data = input_data / 255.0
          true_label = true_label / 255.0
          input_data = np.moveaxis(input_data, -1, 2).reshape((batch_size_val, -1, height, width))
          true_label = np.moveaxis(true_label, -1, 2).reshape((batch_size_val, -1, height, width))
          input_data = np.concatenate((input_data,np.repeat(static_data, batch_size_val, axis=0)), axis=1)
          input  = torch.from_numpy(input_data).float().cuda() 
          target = torch.from_numpy(true_label).float().cuda() 
          prediction = net(input)
          loss = loss_func2(prediction, target)  
          eval_loss_list.append(loss.item())
        avg_train_loss = sum_train_loss / (float(sum_train_iter)+EPS)
        sum_train_loss = 0.0
        sum_train_iter = 0

        print('global_step:', global_step, '\t', 'epoch:', epoch, \
            '\t', 'train_loss:', avg_train_loss, \
            '\t', 'eval_loss:', np.mean(eval_loss_list), \
            '\t', datetime.now(), )
        debug_out = open('res.txt', 'a')
        debug_out.write(str(global_step))
        debug_out.write('\t')
        debug_out.write('%.8f' % float(avg_train_loss))
        debug_out.write('\t')
        debug_out.write('%.8f' % float(np.mean(eval_loss_list)))
        debug_out.write('\n')
        debug_out.close()
        net.train()

      while train_output_queue.empty():
        time.sleep(0.1)
      (input_data, true_label) = train_output_queue.get()
      optimizer.zero_grad()
      input  = torch.from_numpy(input_data).float().cuda() 
      target = torch.from_numpy(true_label).float().cuda() 
      prediction = net(input)
      loss = loss_func2(prediction, target)  
      sum_train_iter += 1
      sum_train_loss += loss.item()
      loss.backward()
      optimizer.step()
      global_step += 1

