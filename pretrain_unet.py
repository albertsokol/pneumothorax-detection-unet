import os

import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from callbacks import LearningRateFinder
from models import unet_pp_pretrain_model

if __name__ == '__main__':
    """ 
    This file was used for pretraining the segmentation UNet++ on ImageNet. You can use it to train other depth
    UNets, or try pretraining using different data. 
    """

    # =================== ADJUST THESE PARAMETERS AS REQUIRED ===================

    # Set the paths where to save trained model, and where to find training and validation data
    save_path = '/path/where/model/will/be/saved/'
    train_path = '/path/to/imgnet/training/data'
    val_path = '/path/to/imgnet/validation/data'

    # Training parameters for pretraining on ImageNet - image size does NOT need to match seg model input size
    BATCH_SIZE = 16
    RESIZE_TO = 224
    NUM_EPOCHS = 50

    # Choose mode as 'lrf' or 'train'; 'lrf' will draw graph for choosing learning rate. 'train' will use
    # checkpoint callback, and will save best model to save_path
    mode = 'train'

    # If training, set the learning rate
    lr = 3e-4

    # ===========================================================================

    assert mode == 'lrf' or mode == 'train', f'mode \'{mode}\' does not exist, please use \'lrf\' or \'train\''

    # Loading data and setting up training constants
    n_train = sum([len(files) for r, d, files in os.walk(train_path)])
    n_val = sum([len(files) for r, d, files in os.walk(val_path)])
    print('Number of training samples:', n_train, 'Number of validation samples:', n_val)
    TRAIN_STEPS = n_train // BATCH_SIZE
    VAL_STEPS = n_val // BATCH_SIZE

    # Set up image augmentors - note pretrained on grayscale images
    train_aug = ImageDataGenerator(rotation_range=12.,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   brightness_range=(0.7, 1.3),
                                   shear_range=6.,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rescale=1./255)
    val_aug = ImageDataGenerator(rescale=1./255)

    # Set up generators; I used flow from directory for the ImageNet dataset as it's in folders
    train_gen = train_aug.flow_from_directory(train_path,
                                              target_size=(RESIZE_TO, RESIZE_TO),
                                              color_mode="grayscale",
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    val_gen = val_aug.flow_from_directory(val_path,
                                          target_size=(RESIZE_TO, RESIZE_TO),
                                          color_mode="grayscale",
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

    # Set up callbacks
    if mode == 'lrf':
        lrf = LearningRateFinder(NUM_EPOCHS * TRAIN_STEPS)
        cbs = [lrf]
    if mode == 'train':
        checkpoint = callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
        cbs = [checkpoint]

    # Create the model and compile with Adam optimizer; can change l to change UNet depth (see model plots)
    model = unet_pp_pretrain_model(RESIZE_TO, l=3)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Begin training
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=TRAIN_STEPS,
                        validation_steps=VAL_STEPS,
                        callbacks=cbs)

    # Plot callback graphs if used
    if mode == 'lrf':
        lrf.plot()

    # Plot the losses and accuracy for the model
    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']
    acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']

    fig, axs = plt.subplots(2)
    fig.set_size_inches(8, 12)

    axs[0].plot(range(1, 1 + len(loss_history)), loss_history, 'r-', label='train loss')
    axs[0].plot(range(1, 1 + len(val_loss_history)), val_loss_history, 'b-', label='val loss')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[0].legend(loc="upper right")

    axs[1].plot(range(1, 1 + len(acc_history)), acc_history, 'g-', label='train accuracy')
    axs[1].plot(range(1, 1 + len(val_acc_history)), val_acc_history, 'm-', label='val accuracy')
    axs[1].set(xlabel='epochs', ylabel='classification accuracy')
    axs[1].legend(loc="upper right")

    plt.show()
