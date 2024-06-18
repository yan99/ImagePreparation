#!/bin/sh -l
# FILENAME: train1.sh 

module load anaconda
conda activate py311_new
python train.py --T 200 --epochs 100 --print-freq 5 --save-epoch-freq 10 --ckpt-dir D:/ImagePreparation/checkpts