import argparse
import torch
import imageio
import skimage.transform
import torchvision
import pandas as pd
import cv2
import numpy as np
import pickle

from torch import nn
import torch.optim
import MSMF_model
import MSMF_data
import utils

image_w = 256
image_h = 256
default_test_path = '../middlebury/'



def test():
    parser = argparse.ArgumentParser(description='Depth Denoise')
    parser.add_argument("--cuda_idx", default=0, type=int, help="cuda id")
    parser.add_argument('-o', '--output', default=None, metavar='DIR',
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

    test = pd.read_pickle(default_test_path+"test.pickle")
    test = test.to_numpy()
    for i in range(np.shape(test)[0]):
        rgb = test[i,0]
        depth = test[i,1]
        gt = test[i,2]

        # Bi-linear
        rgb = skimage.transform.resize(rgb, (image_h, image_w), order=1,
                                        mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                        mode='reflect', preserve_range=True)
        gt = skimage.transform.resize(gt, (image_h, image_w), order=0,
                                        mode='reflect', preserve_range=True)

        rgb = rgb / 255
        rgb = torch.from_numpy(rgb).float()
        depth = torch.from_numpy(depth).float()
        gt = torch.from_numpy(gt).float()
        rgb = rgb.permute(2, 0, 1)
        depth.unsqueeze_(0)
        gt.unsqueeze_(0)

        rgb = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])(rgb)
        depth = torchvision.transforms.Normalize(mean=[19050/5000],
                                                std=[9650/5000])(depth)
        gt = torchvision.transforms.Normalize(mean=[19050/5000],
                                            std=[9650/5000])(gt)

        rgb = rgb.to(device).unsqueeze_(0)
        depth = depth.to(device).unsqueeze_(0)
        gt = gt.to(device).unsqueeze_(0)

        pred = model(rgb, depth, pretrain = True)
        loss = criterion(pred, gt)

        # pred = model(rgb, depth, pretrain = False)
        # loss = criterion(pred, gt)

        test_loss.append(loss)

        depth_out = depth[0].cpu().detach().numpy()
        pred_out = pred[0].cpu().detach().numpy()
        gt_out = gt[0].cpu().detach().numpy()

        # print(np.shape(depth_out))
        
        depth_out = depth_out.transpose((1,2,0))
        pred_out = pred_out.transpose((1,2,0))
        gt_out = gt_out.transpose((1,2,0))
        
        depth_out = cv2.normalize(depth_out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        pred_out = cv2.normalize(pred_out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        gt_out = cv2.normalize(gt_out,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite('checkpts/test_img/test_{}_depth.png'.format(i),depth_out)
        cv2.imwrite('checkpts/test_img/test_{}_pred.png'.format(i),pred_out)
        cv2.imwrite('checkpts/test_img/test_{}_gt.png'.format(i),gt_out)
    
    with open('checkpts/test_img/loss_test.pickle',"wb") as loss_out:
        pickle.dump(test_loss, loss_out)
    print("Test job finish!\n")

if __name__ == '__main__':
    test()