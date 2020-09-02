import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class LearningRateFinder(Callback):
    """ Callback function that graphs learning rate against loss to help find optimal learning rate.

    Callback function that inherits from Keras Callback module. Tries a range of learning rates from min_lr to max_lr
    along a logarithmic scale, split up into total_steps steps. Saves learning rate and loss after each batch update in
    the history attribute. The plot can then be used to find the best learning rates to use for training.

    ...
    Methods
    -------
    plot(): plot learning rate against loss. Use this graph to find optimal lr: the part where the loss stops
            decreasing. Any point on the loss' downward slope can be used. Note you might need to zoom in a fair
            bit on the graph if the loss explodes at the end

    ...
    Attributes
    ----------
    total_steps: int:
        the total number of batch updates to perform. This is number of epochs * training steps per epoch
        the more total_steps, the more stable the output graph will be, around 1000 steps is usually a good amount
    min_lr: float:
        base learning rate to start from
    max_lr: float:
        maximum possible learning rate to test

    ...
    Public attributes
    -----------------
    history: the history dictionary contains keys 'lr' for learning rate values and 'loss' for losses
    """

    def __init__(self, total_steps, min_lr=1e-9, max_lr=1.):
        """
        Constructor for callback function that graphs learning rate against loss to help find optimal learning rate.

        ...
        Parameters
        ----------
        total_steps: int:
            the total number of batch updates to perform. This is number of epochs * training steps per epoch
            the more total_steps, the more stable the output graph will be, around 1000 steps is usually a good amount
        min_lr: float:
            base learning rate to start from
        max_lr: float:
            maximum possible learning rate to test

        ...
        Public attributes
        -----------------
        history: the history dictionary contains keys 'lr' for learning rate values and 'loss' for losses
        """
        super(LearningRateFinder, self).__init__()
        self._history = {'lr': [], 'loss': []}
        self.__min_lr = min_lr
        self.__max_lr = max_lr
        self.__total_steps = total_steps
        self.__batch_number = 0.
        self.__k = np.log(max_lr / min_lr) / self.__total_steps

    @property
    def history(self):
        """ Getter for the history dictionary containing keys 'lr' for learning rates and 'loss' for losses. """
        return self._history

    def __lr(self):
        """ Returns learning rate on an exponentially increasing scale from min_lr to max_lr. """
        return self.__min_lr * np.exp(self.__k * self.__batch_number)

    def on_train_begin(self, logs=None):
        """ Set learning rate to base, minimum value when training begins. """
        K.set_value(self.model.optimizer.lr, self.__min_lr)

    def on_batch_end(self, batch, logs=None):
        """ Update learning rate, append loss and learning rate to history. """
        self.__batch_number += 1

        K.set_value(self.model.optimizer.lr, self.__lr())

        self.history['lr'].append(K.get_value(self.model.optimizer.lr))
        current_loss = logs.get('loss')
        self.history['loss'].append(current_loss)

    def plot(self):
        """ Plots learning rate against loss.

        Recommended to pick the point where the loss just starts to drop as the lower bound for cyclical learning rate,
        pick point where loss stops decreasing or becomes ragged as upper bound for cyclical learning rate.
        Alternatively, pick the minimum loss as static learning rate, or starting rate for a decreasing regimen.
        """
        lr_history = self.history['lr']
        smooth_loss_history = self.__smooth_losses()

        _fig, _axs = plt.subplots(2)
        _axs[0].plot(lr_history, smooth_loss_history, 'r-')
        _axs[0].set(xlabel='learning rate', ylabel='loss')
        _axs[0].set_xscale('log')

        _axs[1].plot(range(len(lr_history)), lr_history, 'b-')
        _axs[1].set(xlabel='iterations', ylabel='learning rate')
        plt.show()

    def __smooth_losses(self, beta=0.96):
        """ Returns an exponential moving average of the losses for smoother plotting.

        By default, beta is set to 0.96 which will result in moving average of around 25 data points. Decrease beta
        to reduce smoothing. Also applies bias correction to improve plotting at start of data.
        """
        avg_loss = 0
        coarse_loss = self.history['loss']
        smooth_loss = []

        for i in range(len(coarse_loss)):
            avg_loss = (beta * avg_loss) + ((1 - beta) * coarse_loss[i])
            corr_avg_loss = avg_loss / (1 - (beta ** (i + 1)))
            smooth_loss.append(corr_avg_loss)

        return smooth_loss
