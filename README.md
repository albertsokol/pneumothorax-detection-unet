# Pneumothorax detection with U-Net/U-Net++ in Tensorflow 2.x
An image classification and segmentation pipeline using U-Net/U-Net++. I've used it to train a pneumothorax detector here on the data from the SIIM-ACR pneumothorax segmentation contest at https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/

Features:
 - Classifier builds on TF2 Keras pretrained models, segmentation model can be chosen as U-Net or U-Net++ 
 - Custom generators for reading dicom files and RLE encoded masks support fully customizable image and mask augmentation 
 - Customizable U-Net++ depth, from 1 up to 4 levels of encoding + decoding  
 - Performance finder class for finding and plotting precision-recall curves of classifier model, as well as mean dice calculator to evaluate overall performance of model as per SIIM-ACR contest
 - Prediction class for plotting side-by-side model predictions and radiologist ground truths 
 
L=3 depth U-Net++ model pretrained on ImageNet available at: https://drive.google.com/drive/folders/1Xrf77veOoGOegThvm7a-za0WgO1j7Su8

## Demo 

## Training
There are 3 different training files: 
 - pretrain_unet.py
 - train_classifier.py
 - train_seg.py


### pretrain_unet.py (optional)
Use this to pretrain a **segmentation** model on ImageNet/other data before training. Can give a modest boost in dice score. For classifier models, TF already offers pretrained image classifiers. 


### train_classifier.py
Train the classifier model. 
 - **Required input format**: X-ray images should be in **.dicom** format, of any dimensions.
 - **Required label format**: **.csv** file with two headers: **'ImageId'** and **'Class'**. ImageId should be the name of the image without the .dcm extension. Class should be 0 for negative samples, and 1 for positive samples. See train_classifier.csv for an example. 

Set training parameters:
 - Edit the config.ini file to set batch_size, resize_to, and train_prop
 - resize_to = length and width that the image will be resized to; this is therefore also the input size to the model
 - train_prop = %age of the data in image folder to use for training; remainder will be used for validation 
 - mode can be **'lrf'** or **'train'**. See below for more info on learning rate finder. Using **'train'** mode will automatically save the best model to save_path

Further options:
 - you can adjust the augmentation parameters by passing in different values for the arguments in the ClassifierGenerator constructor
 - you can choose a different backbone by adjusting the **'bb'** parameter of the create_classification_model function: 'DenseNet121', 'DenseNet169', 'DenseNet201' are supported already but more can easily be added in the models.py file

### train_seg.py


### learning rate finder


## Prediction 

## performance.py 

## References 



