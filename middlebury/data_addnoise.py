import pandas as pd	
import numpy as np 
import cv2 

from scipy.interpolate import griddata
from scipy import ndimage

'''
Noise Models References:
@article{Barron:etal:2013A,
  author  = {Jonathan T. Barron and Jitendra Malik},
  title   = {Intrinsic Scene Properties from a Single RGB-D Image},
  journal = {CVPR},
  year    = {2013},
}

@article{Bohg:etal:2014,
  title   = {Robot arm pose estimation through pixel-wise part classification},
  author  = {Bohg, Jeannette and Romero, Javier and Herzog, Alexander and Schaal, Stefan},
  journal = {ICRA},
  year    = {2014},
}
'''

'''
Function add_gaussian_shifts referred to https://github.com/ankurhanda/simkinect/blob/master/add_noise.py
'''
def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp

'''
Function filterDisp modified according to https://github.com/ankurhanda/simkinect/blob/master/add_noise.py
'''    

def filterDisp(disp, dot_pattern_, invalid_disp_):
    
    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape
    
    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        if(accu>=invalid_disp_):
                            accu = mean
                            window[center,center] = accu
                        # assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp


if __name__ == "__main__":

    # reading the image directly in gray with 0 as input 
    dot_pattern_ = cv2.imread("kinect-pattern_3x3.png", 0)

    # various variables to handle the noise modelling
    scale_factor  = 100     # converting depth from m to cm 
    focal_length  = 480.0   # focal length of the camera used 
    baseline_m    = 0.075   # baseline in m 
    invalid_disp_ = 99999999.9

    rgb_pkl_dir = 'rgb.pickle'
    gt_pkl_dir = 'depth.pickle'   

    rgb_imgs = pd.read_pickle(rgb_pkl_dir)
    gt_depth = pd.read_pickle(gt_pkl_dir)

    depth_imgs = []
    gt_imgs = []
    print(len(rgb_imgs))
    for i in range(len(rgb_imgs)):
        print(i)
        gt = gt_depth[i]
        h, w = gt.shape 
        depth = gt.astype('float') / 5000.0

        depth_interp = add_gaussian_shifts(depth)
        disp_= focal_length * baseline_m / (depth_interp + 1e-10)
        depth_f = np.round(disp_ * 8.0)/8.0

        dot_h, dot_w = dot_pattern_.shape
        if(h>dot_h or w>dot_w):
            num_h = np.ceil(h/dot_h)
            num_w = np.ceil(w/dot_w)
            dot_pattern_ = np.tile(dot_pattern_,(num_h.astype(int),num_w.astype(int)))

        out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

        depth = focal_length * baseline_m / out_disp
        depth[out_disp == invalid_disp_] = 0 
        
        # The depth here needs to converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        noisy_depth = (35130/np.round((35130/(np.round(depth*scale_factor)+1e-10)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor

        noisy_depth = noisy_depth
        noisy_depth = noisy_depth.astype('float')
        
        '''
        noisy_depth = noisy_depth * 5000.0 
        noisy_depth = noisy_depth.astype('uint16')
        '''
        
        depth2save = np.hstack((gt.astype(np.uint16), (noisy_depth*5000).astype(np.uint16)))
        cv2.imwrite('noise_depth/depth_noised_{}.png'.format(i), depth2save)

        depth_imgs.append(noisy_depth)
        gt_imgs.append((gt/5000.0).astype('float'))
        
    raw_df = pd.DataFrame({'rgb':rgb_imgs,
                            'depth':depth_imgs,
                            'gt':gt_imgs})
    raw_df.to_pickle("data.pkl")