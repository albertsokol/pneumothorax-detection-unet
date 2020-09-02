import tensorflow.keras.backend as K


def clip(z):
    """ Clip all values in a tensor to prevent divide by 0 errors. """
    z = K.clip(z, 1e-7, 1)
    return z


def weighted_pixel_bce_loss(beta, batch_size):
    """
    Weighted pixel-wise binary cross entropy loss function, averaged over the batch.
    Parameters:
    beta: weighting factor. Set this to be the average proportion of class = 1 pixels in a training image
    batch_size: number of training examples in the batch
    """

    def compute_loss(y_true, y_pred):
        # Assign a greater loss to false negative predictions to prevent model always predicting y = 0 for all pixels
        px_wt = 1. / beta
        # Find the number of total pixels in an image
        num_pix = K.int_shape(y_pred)[1] * K.int_shape(y_pred)[2]
        # Calculate the loss
        bce = - ((px_wt * y_true * K.log(clip(y_pred))) + (1 - y_true) * K.log(clip(1 - y_pred)))

        # Sum and average the loss by the number of pixels and by the batch size
        loss = (K.sum(bce) / num_pix) / batch_size

        return loss

    return compute_loss


def dice_loss():
    """
    Computes the dice loss for a predicted segmentation.
    """

    def compute_loss(y_true, y_pred):
        # Compute dice coefficient and return loss
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)
        dice_coefficient = numerator / denominator
        loss = 1 - dice_coefficient

        return loss

    return compute_loss


def combined_dice_wpce_loss(beta, batch_size):
    """
    A combination of 2x dice loss and 1x weighted pixel-wise binary cross entropy loss for a prediction.
    Tends to result in better performance than wpce or dice alone.
    Parameters:
    beta: weighting factor. Set this to be the average proportion of class = 1 pixels in a training image
    batch_size: number of training examples in the batch
    """

    def compute_loss(y_true, y_pred):
        # Get weighted pixel cross entropy loss
        # Assign a greater loss to false negative predictions to prevent model always predicting y = 0 for all pixels
        px_wt = 1. / beta
        # Find the number of total pixels in an image
        num_pix = K.int_shape(y_pred)[1] * K.int_shape(y_pred)[2]
        bce = - ((px_wt * y_true * K.log(clip(y_pred))) + (1 - y_true) * K.log(clip(1 - y_pred)))

        # Sum and average the loss by the number of pixels and by the batch size
        wpce_loss = (K.sum(bce) / num_pix) / batch_size

        # Compute dice coefficient and return loss
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)
        dice_coefficient = numerator / denominator
        _dice_loss = 1 - dice_coefficient

        # The final loss value is a mix of both loss functions
        loss = 2 * _dice_loss + wpce_loss

        return loss

    return compute_loss
