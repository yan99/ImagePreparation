#!/usr/bin/env python

import cv2
import argparse
import time, os, sys
import os.path
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Show Images')
    parser.add_argument('--filename', dest='filename',
                        default=0, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # construct the filenames
    root = 'WiFi_SP_example/NPZs/'
    out = 'WiFi_SP_example/demo/'

    npz_data = np.load(root+args.filename)

    rgb_filename = out+args.filename[:-4]+'_rgb.tif'
    print(rgb_filename)
    cv2.imwrite(rgb_filename, npz_data.f.image)
    
    depth_filename = out+args.filename[:-4]+'_depth.tif'
    depth = npz_data.f.depth
    depth = (depth/(np.amax(depth)-np.amin(depth)))*255
    cv2.imwrite(depth_filename, depth.astype(np.uint8))
    
    seg_filename = out+args.filename[:-4]+'_seg.tif'
    seg = npz_data.f.segmentation
    seg = (seg/(np.amax(seg)-np.amin(seg)))*255
    cv2.imwrite(seg_filename, seg.astype(np.uint8))