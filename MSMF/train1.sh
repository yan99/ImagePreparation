#!/bin/sh -l
# FILENAME: train1.sh 

module load anaconda
conda activate py311_new
python train.py --data-dir ../middlebury --epochs 100 --print-freq 5 --save-epoch-freq 10