#!/usr/bin/env python

import cv2
import argparse
import time, os, sys
from torch import nn
import torch
import numpy as np

def weight_inits(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def print_log(global_step, epoch, local_cnt, cnt_inter, dataset_sz, loss, time_inter):
    print('Step: {:>5} Training Epoch: {:>3} [{:>4}/{:>4}] ({:3.1f}%)]  '
          'Loss: {:.6f}[{:.2f}s every {:>4} data]'.format(
              global_step, epoch, local_cnt, dataset_sz, 100.*local_cnt/dataset_sz,
              loss.data, time_inter, cnt_inter
          ))

def save_ckpt(model_dir, model, optimizer, global_step, epoch, local_cnt, num_train, loss):
    epoch_float = epoch + (local_cnt / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(model_dir, model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))

def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)
        