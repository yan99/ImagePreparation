import numpy as np
import scipy.io
import imageio
import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch
import pandas as pd
from train import img_w, img_h

default_path = '../middlebury/data.pickle'
default_train_path = '../middlebury/'
default_test_path = '../middlebury/'

def dataset_creation(load_dataset, data_path, training_size_portion = 0.8):
    if load_dataset:
        train = pd.read_pickle(default_train_path+"train.pickle")
        test = pd.read_pickle(default_train_path+"test.pickle")
    else:
        if data_path is None:
            data_path = default_path
        data = pd.read_pickle(data_path)
        train = data.sample(frac = training_size_portion, random_state = 200)
        test = data.drop(train.index)

        train.to_pickle(default_train_path+"train.pickle")
        test.to_pickle(default_test_path+"test.pickle")
    return train, test

class middleburyDataset(Dataset):
    def __init__(self, dataframe, transform = None):
        self.transform = transform
        self.dataframe = dataframe
    
    def __getitem__(self, index):
        row = self.dataframe[index]
        data = {'rgb': row[0], 'depth': row[1], 'gt': row[2]}
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.dataframe)

class scaleNorm(object):
    def __call__(self, sample):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']

        # Bi-linear
        rgb = skimage.transform.resize(rgb, (img_h, img_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (img_h, img_w), order=0,
                                         mode='reflect', preserve_range=True)
        gt = skimage.transform.resize(gt, (img_h, img_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'rgb': rgb, 'depth': depth, 'gt': gt}

class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * rgb.shape[0]))
        target_width = int(round(target_scale * rgb.shape[1]))
        # Bi-linear
        rgb = skimage.transform.resize(rgb, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        gt = skimage.transform.resize(gt, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'rgb': rgb, 'depth': depth, 'gt': gt}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']
        h = rgb.shape[0]
        w = rgb.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'rgb': rgb[i:i + img_h, j:j + img_w, :],
                'depth': depth[i:i + img_h, j:j + img_w],
                'gt': gt[i:i + img_h, j:j + img_w]}


class RandomFlip(object):
    def __call__(self, sample):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']
        if random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
            gt = np.fliplr(gt).copy()

        return {'rgb': rgb, 'depth': depth, 'gt': gt}

# To tensor
class Normalize(object):
    def __call__(self, sample):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']
        rgb = rgb / 255
        rgb = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(rgb)
        depth = torchvision.transforms.Normalize(mean=[19050/5000],
                                                 std=[9650/5000])(depth)
        gt = torchvision.transforms.Normalize(mean=[19050/5000],
                                                 std=[9650/5000])(gt)
        sample['rgb'] = rgb
        sample['depth'] = depth
        sample['gt'] = gt

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']
        # numpy image: H x W x C
        # torch image: C X H X W
        rgb = rgb.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float32)
        gt = np.expand_dims(gt, 0).astype(np.float32)
        return {'rgb': torch.from_numpy(rgb).float(),
                'depth': torch.from_numpy(depth).float(),
                'gt': torch.from_numpy(gt).float(),}
                