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
 - **Required label format**: **.csv** file with two headers: **'ImageId'** and **'Class'**. ImageId should be the name of the image without the .dcm extension. Class should be 0 for negative samples, and 1 for positive samples. See train_classifier_example.csv for an example. 

Set training parameters:
 - Edit the config.ini file to set `batch_size`, `resize_to`, and `train_prop`
 - `resize_to` = length and width that the image will be resized to; this is therefore also the input size to the model
 - `train_prop` = %age of the data in image folder to use for training; remainder will be used for validation 
 - `mode` can be **'lrf'** or **'train'**. See below for more info on learning rate finder. Using **'train'** mode will automatically save the best model to save_path

Further options:
 - you can adjust the augmentation parameters by passing in different values for the arguments in the ClassifierGenerator constructor
 - you can choose a different backbone by adjusting the **'bb'** parameter of the create_classification_model function: `'DenseNet121', 'DenseNet169', 'DenseNet201'` are supported already but more can easily be added in the models.py file

### train_seg.py
Train the segmentation model. Both U-Net and U-Net++ are available. 
 - **Required input format**: X-ray images should be in **.dicom** format, of any dimensions.
 - **Required label format**: **.csv** file with two headers: **'ImageId'** and **'EncodedPixels'**. ImageId should be the name of the image without the .dcm extension. EncodedPixels should be the RLE-format encoded ground truth segmentation map. If there is no pneumothorax in the image, ie a negative sample, the value for EncodedPixels should be -1. See train_seg_example.csv for an example. 
 
 Set training parameters:
  - mostly the same as for train_classifier
  - note `beta_pixel_weighting`: this is the average percentage of label = 1 pixels in an image in the training set, used for the weighted pixel binary cross-entropy loss function. 
  
  Further options:
   - choice of loss: dice loss (1 - dice coefficient), weighted pixel BCE loss, and combined loss (default is 2x dice loss + 1x weighted pixel loss).
   - same augmentation options as classifier - augments images and segmentation maps together. 
   - can change depth of U-Net++ in create_segmentation_model function. Eg., `l=3` will create a U-Net++ model with 3 downsampling and 3 upsampling steps. Plain U-Net can be constructed using `architecture='unet'`. 

### learning rate finder
The learning rate finder can be activated by setting `mode='lrf'`. 

This mode cycles through all feasible learning rates, and plots the loss against the learning rate. Using this, you can find the optimal learning rate for your configuration. 

Use at least 1000 training steps for best results. 

![Image of LRF plot](https://github.com/albertsokol/pneumothorax-detection-unet/blob/master/readme%20images/lrf_labelled.png)

## Prediction 
This file will run the prediction pipeline.

Images are fed to the classification model. If the output is higher than `classifier_threshold`, they are also fed to the segmentation model. Note that as the classifier model uses RGB and the segmentation model uses Grayscale, the image is converted during the process. The prediction file uses the `train_prop` value to select only validation set images for displaying predictions. If you had a test set, you could re-configure this. 

The predict function then plots ground truth and predicted images side by side.

On the left is the ground truth label, and if there is a ground truth segmentation map, it is displayed in red on the image.

On the right is the predicted label, the confidence in the prediction `(classifier output * 100)` and the predicted segmentation map if appropriate. Brighter areas represent higher confidence by the segmentation model. 

![Image of prediction plot](https://github.com/albertsokol/pneumothorax-detection-unet/blob/master/readme%20images/predict.png)

## performance.py 
This file can be used to plot precision-recall curves or find the mean dice score of the prediction pipeline. 

### Plotting precision-recall
The output of the classifier model is a float between 0 and 1. The classifier threshold can be changed to affect the precision and recall of the model.

For example, if set to 0.8, only X-rays which generate an output of >0.8 will be passed to the segmentation model.

### Mean dice score
This follows the Kaggle contest linked above, and calculates the mean dice score of the classifier and segmentation pipeline at a single classifier threshold. 

You can try testing the mean dice score at many different classifier thresholds to choose the one with best performance.

## References 
U-Net https://arxiv.org/pdf/1505.04597.pdf

U-Net++ https://arxiv.org/pdf/1912.05074.pdf


