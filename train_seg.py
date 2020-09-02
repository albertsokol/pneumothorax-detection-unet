import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

import utils
from callbacks import LearningRateFinder
from generators import SegGenerator
from losses import weighted_pixel_bce_loss, dice_loss, combined_dice_wpce_loss
from metrics import dice_coefficient_wrapper
from models import create_segmentation_model, unet_pp_pretrain_model

if __name__ == '__main__':
    """
    Train a segmentation model based on the UNet++ / UNet papers. 
    Images classified as PT by classifier model will then be fed to this model during prediction. 
    Model depth can be changed by modifying the 'l' parameter. 
    """

    # =================== ADJUST THESE PARAMETERS AS REQUIRED ===================

    # Set the paths where to save trained model, and where to find training and validation data
    save_path = '/path/where/model/will/be/saved/'
    image_path = '/path/to/training/data/'
    csv_path = '/path/to/labels/csv/file'
    imgnet_pretrain_path = '/path/to/pretrained/image_net/model/'

    # Training parameters - read from config.ini file. Beta is the average % of PT pixels across training set only
    BATCH_SIZE, RESIZE_TO, TRAIN_PROP = utils.read_config_file()
    NUM_EPOCHS = 80
    beta_pixel_weighting = 0.010753784

    # Choose mode as 'lrf' or 'train'; 'lrf' will draw graph for choosing learning rate. 'train' will use
    # checkpoint callback, and will save best model to save_path
    mode = 'train'

    # If training, set the learning rate
    lr = 3e-4

    # ===========================================================================

    assert mode == 'lrf' or mode == 'train', f'mode \'{mode}\' does not exist, please use \'lrf\' or \'train\''

    # Loading data and training constants
    overall_df = pd.read_csv(csv_path, index_col='ImageId')
    overall_df_size = len(overall_df)
    train_num = int(overall_df_size * TRAIN_PROP)
    print('Number of training samples:', train_num, 'Number of validation samples:', overall_df_size - train_num)
    TRAIN_STEPS = train_num // BATCH_SIZE
    VAL_STEPS = (overall_df_size - train_num) // BATCH_SIZE

    # Set up dataframes and generators
    train_df = overall_df[:train_num]
    val_df = overall_df[train_num:]
    train_generator = SegGenerator(train_df, image_path, BATCH_SIZE, resize_to=RESIZE_TO)
    val_generator = SegGenerator(val_df, image_path, BATCH_SIZE, resize_to=RESIZE_TO, aug=False)

    # Set up loss functions and metrics - by default, combined_loss is used
    weighted_bce_loss = weighted_pixel_bce_loss(beta_pixel_weighting, BATCH_SIZE)
    dice_loss = dice_loss()
    combined_loss = combined_dice_wpce_loss(beta_pixel_weighting, BATCH_SIZE)
    dice_coefficient = dice_coefficient_wrapper()

    # Set up callbacks - if using 'train' mode, model with best val dice coefficient will be saved to save_path
    if mode == 'lrf':
        lrf = LearningRateFinder(NUM_EPOCHS * TRAIN_STEPS)
        cbs = [lrf]
    if mode == 'train':
        checkpoint = callbacks.ModelCheckpoint(save_path, monitor='val_dice_coefficient', save_best_only=True,
                                               verbose=1, mode='max')
        cbs = [checkpoint]

    # Load classification UNet++ trained on ImageNet (224x224 input)
    imgnet_model = unet_pp_pretrain_model(224)
    imgnet_model.load_weights(imgnet_pretrain_path)

    # Create a new segmentation UNet++/UNet (512x512 input)
    model = create_segmentation_model(RESIZE_TO)

    # Because the input sizes and output heads are different, transfer weights like this
    print('Loading following layer weights from ImageNet pretrained model to new segmentation model...')
    for i in range(1, len(imgnet_model.layers) - 2):
        wts = imgnet_model.layers[i].get_weights()
        model.layers[i].set_weights(wts)
        print('imgnet:', imgnet_model.layers[i].name, '--------> seg:', model.layers[i].name)

    """
    Change and uncomment this if you want to freeze layers during training. 
    
    for l in model.layers:
        if l.name == 'conv2d_14' or l.name == 'conv2d_15':
            l.trainable = False
    """

    # Set up Adam optimizer and compile the model - can change the loss for experimenting
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=combined_loss, metrics=[dice_coefficient])

    # Begin training
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=TRAIN_STEPS,
                        validation_steps=VAL_STEPS,
                        callbacks=cbs)

    # Plot callback graphs if used
    if mode == 'lrf':
        lrf.plot()

    # Plot the losses and dice coefficients for the model
    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']
    dice_coefficient_history = history.history['dice_coefficient']
    val_dice_coefficient_history = history.history['val_dice_coefficient']

    fig, axs = plt.subplots(2)
    fig.set_size_inches(8, 12)

    axs[0].plot(range(1, 1 + len(loss_history)), loss_history, 'r-', label='train loss')
    axs[0].plot(range(1, 1 + len(val_loss_history)), val_loss_history, 'b-', label='val loss')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[0].legend(loc="upper right")

    axs[1].plot(range(1, 1 + len(dice_coefficient_history)), dice_coefficient_history, 'g-', label='train dice')
    axs[1].plot(range(1, 1 + len(val_dice_coefficient_history)), val_dice_coefficient_history, 'm-', label='val dice')
    axs[1].set(xlabel='epochs', ylabel='dice coefficient')
    axs[1].legend(loc="upper right")

    plt.show()
