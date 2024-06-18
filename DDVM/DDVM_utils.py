import torch
import torch.nn.functional as F  
import os, time, sys
import numpy as np
import torch.nn as nn

class forwardDiffusion():
    def __init__(self, T=200):
        self.T = T
        self.betas = self.linear_beta_schedule(time_steps=self.T)
        self.alphas = 1. - self.betas
        self.sqrt_one_alphas = torch.sqrt(1.0 / (1. - self.betas))
        self.sqrt_alpha_bar, self.sqrt_one_minus_alpha_bar, self.var = self.get_alpha(self.T)

    def linear_beta_schedule(self, time_steps, start=0.0001, end=0.02):
        return torch.linspace(start, end, time_steps)

    def get_alpha(self, T=200):
        alphas_cumprod = torch.cumprod(self.alphas, axis = 0)
        sqrt_alpha_bar = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - alphas_cumprod)

        var = self.betas * (1. - F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0)) / (1. - alphas_cumprod)
        return sqrt_alpha_bar, sqrt_one_minus_alpha_bar, var

    def get_index_from_list(self, val, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = val.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    ####################################################
    # output    x_{t}, noise_{t}
    def forward_diffusion_sample(self, depth, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(depth)
        sqrt_alpha_bar_t = self.get_index_from_list(self.sqrt_alpha_bar, t, depth.shape)
        sqrt_one_minus_alpha_bar_t = self.get_index_from_list(self.sqrt_one_minus_alpha_bar, t, depth.shape)
        # mean + variance
        return sqrt_alpha_bar_t.to(device) * depth.to(device) \
        + sqrt_one_minus_alpha_bar_t.to(device) * noise.to(device), noise.to(device)

def get_loss(model, input, epsilon, mask, t):
    epsilon_pred = model(input, t)
    # compare with gt original (without filling) masked epsilon
    return F.l1_loss(epsilon * mask, epsilon_pred * mask)

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
        