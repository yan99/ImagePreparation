import argparse
import torch
import imageio
import skimage.transform
import torchvision
import pandas as pd
import cv2
import numpy as np
import pickle
import os
from torch import nn

import torch.optim
import MSMF_model
import MSMF_data
import utils

image_w = 1024 # 1024
image_h = 768 # 768
default_test_path = '../WiFi_SP_example/NPZs/'
default_output_path = 'WiFi_SP_clean_depth/'

def WiFi_SP_test():
    parser = argparse.ArgumentParser(description='Depth Denoise')
    parser.add_argument("--cuda_idx", default=0, type=int, help="cuda id")
    parser.add_argument('-i', '--input', default=default_test_path, metavar='DIR',
                        help='path to input')
    parser.add_argument('-o', '--output', default=default_output_path, metavar='DIR',
                        help='path to output')
    parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()
    device = torch.device("cuda:0" if args.cuda_idx and torch.cuda.is_available() else "cpu")

    test_loss = []
    running_loss = 0.0
    model = MSMF_model.MSMF_CNN(pretrain = True)
    utils.load_ckpt(model, None, args.last_ckpt, device)
    criterion = nn.MSELoss(reduction='sum')  
    model.eval()
    model.to(device)

    NPZs_list = os.listdir(args.input)

    for i in range(np.shape(NPZs_list)[0]):
        print(NPZs_list[i])
        try:
            npz_data = np.load(args.input+NPZs_list[i])
            rgb = npz_data.f.image
            depth = npz_data.f.depth

            # Bi-linear
            rgb = skimage.transform.resize(rgb, (image_h, image_w), order=1,
                                            mode='reflect', preserve_range=True)
            # Nearest-neighbor
            depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                            mode='reflect', preserve_range=True)

            rgb = rgb / 255
            rgb = torch.from_numpy(rgb).float()
            depth = torch.from_numpy(depth).float()
            rgb = rgb.permute(2, 0, 1)
            depth.unsqueeze_(0)

            rgb = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])(rgb)
            depth = torchvision.transforms.Normalize(mean=[19050/5000],
                                                    std=[9650/5000])(depth)

            rgb = rgb.to(device).unsqueeze_(0)
            depth = depth.to(device).unsqueeze_(0)

            pred = model(rgb, depth, pretrain = True)

            depth_out = pred

            depth_out = depth[0].cpu().detach().numpy()
            pred_out = pred[0].cpu().detach().numpy()
            
            depth_out = depth_out.transpose((1,2,0))
            pred_out = pred_out.transpose((1,2,0))
            
            depth_out = cv2.normalize(depth_out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            pred_out = cv2.normalize(pred_out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(args.output+'test_{}_depth.png'.format(i),depth_out)
            cv2.imwrite(args.output+'test_{}_pred.png'.format(i),pred_out)
        except:
            print("Unable to generate denoise image {}\n".format(i))
    
    print("Test on WiFi_SP job finish!\n")

if __name__ == '__main__':
    WiFi_SP_test()