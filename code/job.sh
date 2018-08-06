#!/bin/sh

CUDA_VISIBLE_DEVICE=1 python hsi_cnn_train.py --data /brazos/mayerich/berisha/data/hd/left-pca16/ --masks /brazos/mayerich/berisha/data/hd/masks/com-left/ --checkpoint /brazos/mayerich/berisha/results/chp/orig-fc128-64ep/ --crop 33 --classes 6 --epochs 64 --batch 128 --balance --samples 100000
