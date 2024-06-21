# Depth Image Denoise 
This implementations include three parts, the middlebury dataset together with noise models, an implementation of a CNN denoising model MSMF, and an implementation of a diffusion model DDVM (with a simple Unet, will implement with efficient Unet and the patchwise training method in the near future).


# On linux
In `.bash_profile` file in home directory: `module load cuda/12.1.1`

In bash file, load with log file which is actually the bash profile file defined above by using `-l`: `#!/bin/sh -l`

Sample bash file is included.

Sample commend line for submitting bash job with memories of 8 CPUs:

```
sbash -t 0:30:00 --nodes=1 --gpus-per-node=1 -n8 train1.sh
```
Tip: issue caused by tab, in vim, use `/\t` to find tabs and hit `n` to go to next.

Install required packages:
```
pip install -r requirements.txt
```


# Citations

If you use this code, please consider citing the following papers


Noise Models References:
```
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
```
MSMF, the CNN model:
```
@inproceedings{liao2017multi,
  title={Multi-scale mutual feature convolutional neural network for depth image denoise and enhancement},
  author={Liao, Xuan and Zhang, Xin},
  booktitle={2017 IEEE Visual Communications and Image Processing (VCIP)},
  pages={1--4},
  year={2017},
  organization={IEEE}
}
```
DDVM, the diffusion model
```
@article{saxena2024surprising,
  title={The surprising effectiveness of diffusion models for optical flow and monocular depth estimation},
  author={Saxena, Saurabh and Herrmann, Charles and Hur, Junhwa and Kar, Abhishek and Norouzi, Mohammad and Sun, Deqing and Fleet, David J},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```