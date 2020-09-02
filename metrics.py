import tensorflow.keras.backend as K


def dice_coefficient_wrapper():
    """
    Compute dice coefficient and return as a metric, used for assessing segmentation model performance.
    """

    def dice_coefficient(y_true, y_pred):
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)

        dice = numerator / denominator

        return dice

    return dice_coefficient
