import argparse
import torch
import imageio
import skimage.transform
import torchvision
import pandas as pd
import cv2

import torch.optim
import MSMF_model
import MSMF_data
import utils

image_w = 256
image_h = 256
default_test_path = '../middlebury/'

parser = argparse.ArgumentParser(description='Depth Denoise')
parser.add_argument('-r', '--rgb', default=None, metavar='DIR',
                    help='path to image')
parser.add_argument('-d', '--depth', default=None, metavar='DIR',
                    help='path to depth')
parser.add_argument('-o', '--output', default=None, metavar='DIR',
                    help='path to output')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

def test():

    model = MSMF_model.MSMF_CNN(pretrain = False)
    utils.load_ckpt(model, None, args.last_ckpt, device)
    model.eval()
    model.to(device)

    test = pd.read_pickle(default_test_path+"test.pickle")
    test.to_numpy()
    rgb = test[:][0]
    depth = test[:][1]

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

    pred = model(rgb, depth)

    output = pred
    # output = utils.color_label(torch.max(pred, 1)[1] + 1)[0]

    output = output.numpy().transpose((1,2,0))
    cv2.imwrite(args.output+"/test_{}.png".format(i), output)

if __name__ == '__main__':
    test()