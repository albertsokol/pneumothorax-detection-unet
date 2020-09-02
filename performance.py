import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import models
from tqdm import tqdm

import utils
from generators import ClassifierGenerator, SegGenerator
from losses import combined_dice_wpce_loss
from metrics import dice_coefficient_wrapper


def np_dice_coefficient(seg_gt, seg_pred):
    """ Computes the dice coefficient between gt and predicted segmentations using numpy rather than Keras. """
    numerator = 2 * np.sum(seg_gt * seg_pred)
    denominator = np.sum(seg_gt ** 2) + np.sum(seg_pred ** 2)

    _dice = numerator / denominator

    return _dice


class PerformanceFinder:
    """
    Class for assessing the precision-recall tradeoff of various classification thresholds, or for calculating the mean
    dice coefficient of the trained networks. The mean dice coefficient calculation follows the same conventions as
    the pneumothorax segmentation Kaggle contest.

    ...
    Methods
    -------
    find_mean_dice(classifier_threshold=0.5):
        finds the mean dice coefficient of the classification and segmentation pipeline. classifier_threshold is a float
        between 0 and 1. If the classifier model outputs a prediction greater than this, the model will be predicted
        to be a pt
    plot_pr():
        finds the precision and recall values for a range of classification thresholds and plots them against each
        other. Useful for determining a good classification threshold

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
    thresholds: int:
        the number of different thresholds to test precision and recall at, these will then be plotted when running
        plot_pr

    """

    def __init__(self,
                 fpath,
                 csv_path,
                 classifier_path,
                 seg_path,
                 thresholds=31):

        # Read the config file to get the resize dimensions and the training data proportion
        _, self.resize_to, train_prop = utils.read_config_file()

        # Set up dataframe
        self.df = self.__create_df(csv_path, train_prop)
        self.steps = len(self.df)
        print(f'Calculating metrics on {self.steps} validation samples')

        # Load generators and models
        self.classifier_gen = ClassifierGenerator(self.df, fpath, 1, self.resize_to, aug=False, shuffle=False)
        self.seg_gen = SegGenerator(self.df, fpath, 1, resize_to=self.resize_to, aug=False, shuffle=False)
        self.classifier_model, self.seg_model = self.__load_models(classifier_path, seg_path)

        # Set up thresholds and initial lists for precision and recall calculations
        self.thresholds = np.linspace(0, 1, thresholds, dtype=np.float32)
        self.precisions = []
        self.recalls = []

        # Set up parameters of mean dice testing
        self.dices = []

    @staticmethod
    def __load_models(classifier_model_path, seg_model_path):
        # Pre-loads the classifier and generator models for use in performance calculations
        print('Loading models - this may take a couple of minutes ... ')
        # The losses and metrics here are required to be passed in as custom objects to load the segmentation model
        # but they are not used in the calculation
        chosen_loss = combined_dice_wpce_loss(0.010753784, 1)
        dice_coefficient = dice_coefficient_wrapper()

        _class = models.load_model(classifier_model_path)
        _seg = models.load_model(seg_model_path,
                                 custom_objects={'compute_loss': chosen_loss, 'dice_coefficient': dice_coefficient})

        return _class, _seg

    @staticmethod
    def __create_df(csv_path, train_prop):
        # Create pandas dataframe from csv file
        main_df = pd.read_csv(csv_path, index_col='ImageId')
        df = main_df[int(len(main_df) * train_prop):]

        return df

    def __find_prec_recall(self):
        # File uses the saved models for prediction, and then finds the false positives, etc., for precision and recall
        # calculations. Updates the object attributes once complete
        tp = {}
        fp = {}
        fn = {}

        for threshold in self.thresholds:
            tp[threshold] = []
            fp[threshold] = []
            fn[threshold] = []

        print('Beginning precision and recall calculations...')

        for i in tqdm(range(self.steps)):
            # get an image and label from the classifier generator and output a probability prediction
            x, label = self.classifier_gen.__getitem__(i)
            pred = self.classifier_model.predict(x)

            for threshold in self.thresholds:
                thresh_pred = pred >= threshold
                tp[threshold].append((thresh_pred == 1. and label == 1.).astype(np.uint8))
                fp[threshold].append((thresh_pred == 1. and label == 0.).astype(np.uint8))
                fn[threshold].append((thresh_pred == 0. and label == 1.).astype(np.uint8))

        for threshold in self.thresholds:
            tps = np.sum(tp[threshold])
            fps = np.sum(fp[threshold])
            fns = np.sum(fn[threshold])
            self.precisions.append(tps / (tps + fps))
            self.recalls.append(tps / (tps + fns))

    def plot_pr(self):
        """
        Finds the precision and recall values for a range of classification thresholds and plots them against each
        other.
        """
        self.__find_prec_recall()

        # Set up the plots
        fig, axs = plt.subplots(1)
        axs.plot(self.recalls, self.precisions, 'b-', marker='.')
        axs.set(xlabel='recall', ylabel='precision')
        axs.set_xlim(left=0., right=1.)
        axs.set_ylim(bottom=0., top=1.)

        # Annotate the points
        for i, threshold in enumerate(self.thresholds):
            axs.annotate(f'{threshold:.3f}', xy=(self.recalls[i], self.precisions[i]))

        plt.show()

    def find_mean_dice(self, classifier_threshold=0.5):
        """
        For evaluation of model performance on the dice coefficient metric, as per the original Kaggle contest.

        Parameters:
        classifier_threshold: float:
            float value between 0 and 1. If the classifier model outputs a prediction greater than this, the model will
            be predicted to be a pt
        """
        self.dices = []

        for i in tqdm(range(self.steps)):

            # Run the classifier model first to predict pt or not
            x, label = self.classifier_gen.__getitem__(i)
            class_pred = self.classifier_model.predict(x)
            label_pred = (class_pred > classifier_threshold).astype(int)

            # If correctly identifies non-PT as non-PT then dice coefficient is 1
            if label[0] == 0 and label_pred[0] == 0:
                self.dices.append(1.)
                continue

            # If model incorrectly predicts PT, but no PT is present, the dice coefficient is 0
            if label[0] == 0 and label_pred[0] == 1:
                self.dices.append(0.)
                continue

            # If model incorrectly predicts non-PT, but PT is present, the dice coefficient is 0
            if label[0] == 1 and label_pred[0] == 0:
                self.dices.append(0.)
                continue

            # Generate the ground truth segmentation for this X-ray
            # Note the x is different; the classifier uses RGB while the segmentation model uses grayscale
            x, seg_gt = self.seg_gen.__getitem__(i)

            # Otherwise predict the segmentation
            seg_pred = self.seg_model.predict(x)

            # Get the dice coefficient between the ground truth segmentation and the predicted segmentation
            _dice = np_dice_coefficient(seg_gt, seg_pred)
            self.dices.append(_dice)

        # Take the mean and compare it to the competition metric performance
        mean_dice = np.sum(self.dices) / self.steps
        # print('Mean dice coefficient:', mean_dice)

        return mean_dice


if __name__ == '__main__':
    """
    Can be used for testing the precision and recall of the classifier model, or for calculating the mean dice
    coefficient as per the original Kaggle contest. 
    """

    # Create performance finder object
    pf = PerformanceFinder(csv_path='/path/to/labels/csv/file',
                           fpath='/path/to/image/folder/',
                           classifier_path='/path/to/trained/classifier/model',
                           seg_path='/path/to/trained/segmentation/model')

    print('Use pf.plot_pr() to assess the precision-recall tradeoff of various classification thresholds. '
          'Otherwise, use pf.find_mean_dice() to return the mean dice coefficient.')

    # pf.plot_pr()

    tx = np.linspace(0, 1, 31, dtype=np.float32)
    for t in tx:
        dice = pf.find_mean_dice(classifier_threshold=t)
        print(f'Threshold: {t:.3f}, mean dice found: {dice:.4f}')
