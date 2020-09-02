import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

import utils
from callbacks import LearningRateFinder
from generators import ClassifierGenerator
from models import create_classification_model

if __name__ == '__main__':
    """
    Train a simple image classifier which assigns 0 (no PT) or 1 (PT) to images. 
    Images classified as PT are sent to the segmentation model during the final prediction step. 
    """

    # =================== ADJUST THESE PARAMETERS AS REQUIRED ===================

    # Set the paths where to save trained model, and where to find training and validation data
    save_path = '/path/where/model/will/be/saved/'
    image_path = '/path/to/training/data/'
    csv_path = '/path/to/labels/csv/file'

    # Training parameters - read from config.ini file
    BATCH_SIZE, RESIZE_TO, TRAIN_PROP = utils.read_config_file()
    NUM_EPOCHS = 70

    # Choose mode as 'lrf' or 'train'; 'lrf' will draw graph for choosing learning rate. 'train' will use
    # checkpoint callback, and will save best model to save_path
    mode = 'train'

    # If training, set the learning rate
    lr = 1e-4

    # ===========================================================================

    assert mode == 'lrf' or mode == 'train', f'mode \'{mode}\' does not exist, please use \'lrf\' or \'train\''

    # Loading data and setting up training constants
    overall_df = pd.read_csv(csv_path, index_col='ImageId')
    overall_df_size = len(overall_df)
    train_num = int(overall_df_size * TRAIN_PROP)
    print('Number of training samples:', train_num, 'Number of validation samples:', overall_df_size - train_num)
    TRAIN_STEPS = train_num // BATCH_SIZE
    VAL_STEPS = (overall_df_size - train_num) // BATCH_SIZE

    # Set up dataframes and generators
    train_df = overall_df[:train_num]
    val_df = overall_df[train_num:]
    train_generator = ClassifierGenerator(train_df, image_path, BATCH_SIZE, resize_to=RESIZE_TO)
    val_generator = ClassifierGenerator(val_df, image_path, BATCH_SIZE, resize_to=RESIZE_TO, aug=False)

    # Set up callbacks
    if mode == 'lrf':
        lrf = LearningRateFinder(NUM_EPOCHS * TRAIN_STEPS)
        cbs = [lrf]
    if mode == 'train':
        checkpoint = callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
        cbs = [checkpoint]

    # Create the model and compile with Adam optimizer; can change bb to change the backbone used
    # bb can be any of: 'DenseNet121', 'DenseNet169', 'DenseNet201'
    model = create_classification_model(RESIZE_TO, bb='DenseNet169')
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy'])

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
