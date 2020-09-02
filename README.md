# Pneumothorax detection with U-Net/U-Net++ in Tensorflow 2.x
An image classification and segmentation pipeline using U-Net/U-Net++. I've used it to train a pneumothorax detector here on the data from the SIIM-ACR pneumothorax segmentation contest at https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/

Features:
 - Classifier builds on TF2 Keras pretrained models, segmentation model can be chosen as U-Net or U-Net++ 
 - Custom generators for reading dicom files and RLE encoded masks support fully customizable image and mask augmentation 
 - Customizable U-Net++ depth, from 1 up to 4 levels of encoding + decoding  
 - Performance finder class for finding and plotting precision-recall curves of classifier model, as well as mean dice calculator to evaluate overall performance of model as per SIIM-ACR contest
 - Prediction class for plotting side-by-side model predictions and radiologist ground truths 
 
L=3 depth U-Net++ model pretrained on ImageNet avaiable at (url)

## Demonstrations 

## Training

## Prediction 

## performance.py 

## References 



