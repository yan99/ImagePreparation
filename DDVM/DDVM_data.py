import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import random
import torchvision
import torch
import pandas as pd

default_path = 'D:/ImagePreparation/dataset/RGBD_1200.pkl'
default_train_path = 'D:/ImagePreparation/dataset/'
default_test_path = 'D:/ImagePreparation/dataset/'

img_w = 640 # 1296
img_h = 480 # 968

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

class Dataset_RGBD(Dataset):
    def __init__(self, dataframe, transform = None):
        self.transform = transform
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe[index]
        data = {'rgb': row[0], 'depth': row[1], 'gt_mask': row[1]}
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.dataframe)

# Interpolation to fill the holes in GT images to generate depth image
# The filled images as part of the inputs
class interpolate_mask_x0(object):
    def __call__(self, sample):
        depth, gt_mask = sample['depth'], sample['gt_mask']
        # get mask on valid depth image
        gt_mask = (gt_mask>0)
        gt_mask = gt_mask.astype(np.uint8) # to 0s and 1s
        sample['gt_mask'] = gt_mask

        # interpolate with nearest-neighbor
        invalid_mask = np.logical_or(np.isnan(depth), (depth == 0))
        dilation_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dilation_mask = cv2.dilate(invalid_mask.astype(np.uint8), dilation_disc, 
                                   borderType = cv2.BORDER_CONSTANT, borderValue=int(0))
        # Neighbor mask
        nei_mask = (dilation_mask * (1. - invalid_mask)).astype(np.uint8)

        # For each neighbor masked connected component
        nei_CC = cv2.connectedComponentsWithStats(nei_mask, 8) 
        (total_N, _, values, _) = nei_CC
        for i in range(1, total_N):
            x1 = values[i, cv2.CC_STAT_LEFT] 
            y1 = values[i, cv2.CC_STAT_TOP] 
            w = values[i, cv2.CC_STAT_WIDTH] 
            h = values[i, cv2.CC_STAT_HEIGHT]

            x1_b = np.max((0, x1 - 20))
            y1_b = np.max((0, y1 - 20))
            x2_b = np.min((x1 + w + 20, img_w - 1))
            y2_b = np.min((y1 + h + 20, img_h - 1))
            # calculate mean
            temp_depth = depth[x1_b:x2_b, y1_b:y2_b]
            region_to_mean = (temp_depth * (nei_mask[x1_b:x2_b, y1_b:y2_b] > 0).astype(np.uint16))
            vals = region_to_mean[np.logical_and(np.logical_not(region_to_mean==0), np.logical_not(np.isnan(region_to_mean)))]
            mean_val = np.mean(vals)
            # print(mean_val)
            # filling the area inside the neighbor mask
            temp_depth[invalid_mask[x1_b:x2_b, y1_b:y2_b]>0] = mean_val
            depth[x1_b:x2_b, y1_b:y2_b] = temp_depth

        sample['depth'] = depth
        return sample

class scaleNorm(object):
    def __call__(self, sample):
        rgb, depth, gt_mask = sample['rgb'], sample['depth'], sample['gt_mask']

        # Bi-linear
        rgb = cv2.resize(rgb, (img_h, img_w), interpolation=cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (img_h, img_w), interpolation=cv2.INTER_AREA)
        gt_mask = cv2.resize(gt_mask, (img_h, img_w), interpolation=cv2.INTER_AREA)

        return {'rgb': rgb, 'depth': depth, 'gt_mask': gt_mask}

class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        rgb, depth, gt_mask = sample['rgb'], sample['depth'], sample['gt_mask']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * rgb.shape[0]))
        target_width = int(round(target_scale * rgb.shape[1]))
        # Bi-linear
        rgb = cv2.resize(rgb, (target_height, target_width), interpolation=cv2.INTER_LINEAR)
        # Nearest-neighbor
        depth = cv2.resize(depth, (target_height, target_width), interpolation=cv2.INTER_AREA)
        gt_mask = cv2.resize(gt_mask, (target_height, target_width), interpolation=cv2.INTER_AREA)

        return {'rgb': rgb, 'depth': depth, 'gt_mask': gt_mask}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        rgb, depth, gt_mask = sample['rgb'], sample['depth'], sample['gt_mask']
        h = rgb.shape[0]
        w = rgb.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'rgb': rgb[i:i + img_h, j:j + img_w, :],
                'depth': depth[i:i + img_h, j:j + img_w],
                'gt_mask': gt_mask[i:i + img_h, j:j + img_w]}


class RandomFlip(object):
    def __call__(self, sample):
        rgb, depth, gt_mask = sample['rgb'], sample['depth'], sample['gt_mask']
        if random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy()
            gt_mask = np.fliplr(gt_mask).copy()

        return {'rgb': rgb, 'depth': depth, 'gt_mask': gt_mask}

# To tensor
class Normalize(object):
    def __call__(self, sample):
        rgb = sample['rgb']
        rgb = rgb / 255
        rgb = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(rgb)

        sample['rgb'] = rgb

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, depth, gt_mask = sample['rgb'], sample['depth'], sample['gt_mask']
        # numpy image: H x W x C
        # torch image: C X H X W
        rgb = rgb.transpose((2, 0, 1)).astype(np.float32)
        # 1 channel image

        # According to DDVM paper
        # clip at 10 meter
        # depth/1000 = meter
        depth[depth>10*1000] = 10*1000
        depth = cv2.normalize(depth,depth,0,1.0,cv2.NORM_MINMAX).astype(np.float32)
        # 1 x H x W
        depth = np.expand_dims(depth, 0).astype(np.float32)
        gt_mask = np.expand_dims(gt_mask, 0).astype(np.uint8)
        return {'rgb': torch.from_numpy(np.array(rgb)).float(),
                'depth': torch.from_numpy(np.array(depth)).float(),
                'gt_mask': torch.from_numpy(np.array(gt_mask)).float()}