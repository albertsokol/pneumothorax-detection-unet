import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras.models as models
from matplotlib.colors import ListedColormap
from PIL import Image
from pydicom import dcmread

import utils
from losses import combined_dice_wpce_loss
from metrics import dice_coefficient_wrapper


class PneumothoraxPredictor:
    """
    Predictor class that takes in a chest X-ray image. Classifies as pneumothorax (pt) or non-pneumothorax. If pt
    predicted, carries out segmentation prediction on the X-ray for pt localization.

    ...
    Methods
    -------
    predict(fname=None):
        carry out classification on the chest X-ray and perform segmentation if pt is predicted. Note the optional
        fname string parameter can be passed if prediction on a specific file is required. If an fname is not given,
        a random image in the given folder is chosen. Calls plot method, which shows:
        > cxr with radiologist classification, + segmentation map if ground truth was a pt
        > cxr with model classification, + segmentation map if predicted a pt, + confidence in prediction

    ...
    Attributes
    ----------
    fpath: string:
        file path to the directory containing the dicom images
    csv_path: string:
        file path to the csv file containing rle mask encodings and corresponding filenames
    classifier_path: string:
        file path to the classifier model in tensorflow SavedModel format
    seg_path: string:
        file path to the segmentation model in tensorflow SavedModel format
    classifier_threshold: float:
        threshold for classifying image as a pt, between 0 and 1. Eg., if classifier predicts 0.74 and the threshold is
        0.5, take this as a predicted pt. Precision-recall tradeoff curves can be plotted in performance.py

    read from the config file:
    resize_to: int:
        resize images to this size for input to the models
    train_prop: float:
        train prop is the proportion of the csv file used for training. If train_prop = 0.8, this means the first
        80% of entries in the csv file were used for training, so we will only use the last 20% for predicting here
        - meaning that in this file, we are predicting on the validation set only, to get a less biased overview

    """

    def __init__(self, fpath, csv_path, classifier_path, seg_path, classifier_threshold):
        """
        Constructor for the PneumothoraxPredictor class, which predicts whether a chest x-ray contains a pneumothorax,
        predicts a segmentation map if the classification prediction is above the threshold, and displays the
        results.

        ...
        Parameters
        ----------
        fpath: string:
            file path to the directory containing the dicom images
        csv_path: string:
            file path to the csv file containing rle mask encodings and corresponding filenames
        classifier_path: string:
            file path to the classifier model in tensorflow SavedModel format
        seg_path: string:
            file path to the segmentation model in tensorflow SavedModel format
        classifier_threshold: float:
            threshold for classifying image as a pt, between 0 and 1. Eg., if classifier predicts 0.74 and the threshold
            is 0.5, take this as a predicted pt. Precision-recall tradeoff curves can be plotted in performance.py

        read from the config file:
        resize_to: int:
            resize images to this size for input to the models
        train_prop: float:
            train prop is the proportion of the csv file used for training. If train_prop = 0.8, this means the first
            80% of entries in the csv file were used for training, so we will only use the last 20% for predicting here
            - meaning that in this file, we are predicting on the validation set only, to get a less biased overview
        """
        # Set up dataframe and directories
        self.fpath = fpath
        self.csv_path = csv_path
        self.classifier_path = classifier_path
        self.seg_path = seg_path

        _, self.resize_to, train_prop = utils.read_config_file()
        self.df = self.__create_df(train_prop)

        # Set up model parameters and load models
        self.classifier_threshold = classifier_threshold
        self.classifier, self.seg = self.__load_models()

        # Prepare variables for storing images and masks
        self.img_rgb = None
        self.img_grayscale = None
        self.seg_gt = None
        self.seg_pred = None

    def __create_df(self, train_prop):
        # Loads the csv file containing ground truth mask information
        main_df = pd.read_csv(self.csv_path, index_col='ImageId')
        df = main_df[int(len(main_df) * train_prop):]

        return df

    def __load_models(self):
        # Pre-loads the classifier and segmentation models. Note loss and metric passed in as custom objects only to
        # re-construct the model from SavedModel format, these are not used in prediction
        print('Loading classifier and segmentation models - this may take a couple of minutes ... ')
        chosen_loss = combined_dice_wpce_loss(0.010753784, 1)
        dice_coefficient = dice_coefficient_wrapper()

        classifier = models.load_model(self.classifier_path)
        seg = models.load_model(self.seg_path,
                                custom_objects={'compute_loss': chosen_loss, 'dice_coefficient': dice_coefficient})

        return classifier, seg

    def __load_dicom(self, fname):
        # Loads a dicom file by filename and updates the img rgb and grayscale predictor attributes
        if fname is None:
            # Get random file, otherwise if fname is defined, the pre-determined file will be opened
            rng = np.random.default_rng()
            i = rng.choice(len(self.df))
            fname = self.df.index[i]

        # Get the image mask if it exists, otherwise get a zero, save result in self.seg_gt
        self.__fname_to_mask(fname)

        dcm_file = dcmread(self.fpath + fname + '.dcm')
        dcm_pixel_data = dcm_file.pixel_array
        pil_data = Image.fromarray(dcm_pixel_data)

        if dcm_pixel_data.shape[0] != self.resize_to:
            pil_data = pil_data.resize((self.resize_to, self.resize_to))

        # Get the image in RGB and Grayscale formats with correct dimensions
        pil_data = pil_data.convert('RGB')
        rgb_data = np.array(pil_data)
        rgb_data = np.expand_dims(rgb_data, axis=0)
        l_data = pil_data.convert('L')
        l_data = np.array(l_data)
        l_data = np.expand_dims(l_data, axis=0)
        l_data = np.expand_dims(l_data, axis=-1)

        # Normalise pixels
        rgb_data = rgb_data.astype(np.float32)
        self.img_rgb = rgb_data / 255.
        l_data = l_data.astype(np.float32)
        self.img_grayscale = l_data / 255.

    def __fname_to_mask(self, fname):
        # If there is a ground truth radiologist mask associated with the chosen image, we want to get it for plotting
        rle_mask = self.df.loc[fname].values[0]
        print('rle mask:', rle_mask)

        # If there is no ground truth mask, return zero
        if rle_mask == '-1':
            self.seg_gt = None
            return

        # Convert from rle format to a pixel mask
        try:
            s = rle_mask.split()
        except AttributeError:
            rle_mask = rle_mask[0]
            s = rle_mask.split()

        starts, lengths = [np.asarray(x, dtype=int) for x in (s[::2], s[1::2])]
        int_mask = np.zeros(1024 * 1024, dtype=np.uint8)
        current_pos = 0

        for start, length in zip(starts, lengths):
            current_pos += start
            int_mask[current_pos:current_pos + length] = 1
            current_pos += length

        int_mask = int_mask.reshape(1024, 1024).T

        # Resize to the correct cnn input size
        if 1024 != self.resize_to:
            pil_data = Image.fromarray(int_mask)
            pil_data = pil_data.resize((self.resize_to, self.resize_to), resample=Image.NEAREST)
            int_mask = np.array(pil_data)

        self.seg_gt = np.expand_dims(int_mask, axis=-1)

    def __plot(self, confidence):
        # Plot ground truth segmentation alongside predicted segmentation with classifications and confidences
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18.5, 10.5)

        if self.seg_gt is not None:
            ax1.set_title('GROUND TRUTH: PT present, radiologist label:')

            if np.all(self.seg_pred == np.zeros([self.resize_to, self.resize_to, 1])):
                ax2.set_title(f'PREDICTION: no PT present, {float(confidence) * 100:.1f}% confidence')

            else:
                ax2.set_title(f'PREDICTION: predicted PT present, {float(confidence) * 100:.1f}% confidence')

        else:
            ax1.set_title('GROUND TRUTH: no PT present')

            if np.all(self.seg_pred == np.zeros([self.resize_to, self.resize_to, 1])):
                ax2.set_title(f'PREDICTION: no PT present, {float(confidence) * 100:.1f}% confidence')

            else:
                ax2.set_title(f'PREDICTION: predicted PT present, {float(confidence) * 100:.1f}% confidence')

        # Set up the colourmap for the predicted mask
        cmap = plt.cm.get_cmap('viridis')
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0, 0.7, cmap.N)
        my_cmap = ListedColormap(my_cmap)

        # Plot the images and their masks
        plt.gray()
        ax1.imshow(self.img_grayscale[0, :, :, 0])
        if self.seg_gt is not None:
            ax1.imshow(self.seg_gt[..., 0], cmap='Reds', alpha=0.1)
        ax2.imshow(self.img_grayscale[0, :, :, 0])
        ax2.imshow(self.seg_pred[0, :, :, 0], cmap=my_cmap, alpha=None)

        plt.show()

    def predict(self, fname=None):
        """
        Carry out classification on the chest X-ray and perform segmentation if pt is predicted. Calls plot method,
        which shows:
        > cxr with radiologist classification, + segmentation map if ground truth was a pt
        > cxr with model classification, + segmentation map if predicted a pt, + confidence in prediction

        Parameters:
        fname: string: filename to predict on. If not given, a random image in the folder will be selected
        """
        self.__load_dicom(fname)
        start = time.time()

        # Run predict (image) to get a predicted class
        classifier_pred = self.classifier.predict(self.img_rgb)
        confidence = classifier_pred
        print('classifier pred:', classifier_pred)

        # If prediction < confidence threshold, very likely there is no pneumothorax
        if classifier_pred < self.classifier_threshold:
            confidence = 1 - classifier_pred
            self.seg_pred = np.zeros([1, self.resize_to, self.resize_to, 1])

        # Else if prediction > threshold found at prec/recall testing: classify as PT and segment
        else:
            self.seg_pred = self.seg.predict(self.img_grayscale)

        print('Time to predict:', time.time() - start, 'seconds.')
        self.__plot(confidence)


if __name__ == '__main__':

    _, RESIZE_TO, TRAIN_PROP = utils.read_config_file()
    pp = PneumothoraxPredictor(fpath='/path/to/image/folder/',
                               csv_path='/path/to/labels/csv/file',
                               classifier_path='/path/to/trained/classifier/model',
                               seg_path='/path/to/trained/segmentation/model',
                               classifier_threshold=0.5)
    print('Predictor loaded. Use pp.predict() to predict on a random image. Use pp.predict(fname=) to predict a'
          ' specific file.')

    # To run in the Python Console:
    # exec(open('predict.py').read())
