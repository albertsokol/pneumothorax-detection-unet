from configparser import ConfigParser


def write_config_file(batch_size, resize_to, train_prop):
    """
    Create a config file specifying model hyper parameters. This will be used in training the classification and
    segmentation models and in prediction.

    Parameters:
    batch_size: batch size to be used during training
    resize_to: will resize images to the given dimensions for inputting to CNN
    train_prop: proportion of data to use for training; I have used 0.8 here for an 80/20 train/val split
    """
    config = ConfigParser()
    config.read('config.ini')

    # Add model parameters; will add new section and create config file if it does not already exist
    if not config.has_section('model params'):
        config.add_section('model params')

    config.set('model params', 'batch_size', str(batch_size))
    config.set('model params', 'resize_to', str(resize_to))
    config.set('model params', 'train_prop', str(train_prop))

    # Write the file
    with open('config.ini', 'w') as f:
        config.write(f)

    return


def read_config_file():
    """ Read batch size, resize dimensions, train proportion from config file and apply them to model. """
    config = ConfigParser()
    config.read('config.ini')

    # Get the model parameters as correct type
    batch_size = config.getint('model params', 'batch_size')
    resize_to = config.getint('model params', 'resize_to')
    train_prop = config.getfloat('model params', 'train_prop')

    return batch_size, resize_to, train_prop
