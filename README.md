# Depth Image Denoise 

Pytorch Implementation of Multi-Scale Mutual Feature Convolutional Neural Network for Depth Image Denoise and Enhancement

# On linux
In `.bash_profile` file in home directory: `module load cuda/12.1.1`

In bash file, load with log file which is actually the bash profile file defined above by using `-l`: `#!/bin/sh -l`

Sample bash file is included.

Sample commend line for submitting bash job with memories of 8 CPUs:

```
sbash -t 0:30:00 --nodes=1 --gpus-per-node=1 -n8 train1.sh
```
Install required packages:
```
pip install -r requirements.txt
```


# Citations

If you use this code, please consider citing the following papers

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