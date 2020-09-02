import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from pydicom import dcmread
from tensorflow.keras.utils import Sequence


class SegGenerator(Sequence):
    """
    Generator that returns a batch of images and their ground truth segmentation maps as pixels. Converts segmentations
    from RLE encoding in the given dataset to 0, 1 pixel maps of same size as the input images.

    ...
    Methods
    _______
    The generator has no public methods and interacts directly with Keras to automatically return images and
    ground truth tensors. Note the required methods __len__ and __getitem__:
    __len__: the same thing as TRAIN_STEPS or TEST_STEPS; this is how many times the generator needs to return data
             in an epoch
    __getitem__: returns the images and the segmentation maps as numpy arrays

    ...
    Attributes
    ----------
    dataframe:
        the pandas dataframe object containing the image and rle mask data
    image_path: str:
        path to the folder containing images

    batch_size: int:
        number of training/validation examples in a batch
    resize_to: int:
        the input size of the network; images will be resized to this size
    shuffle: bool:
        if True, data will be randomly shuffled at the end of each epoch

    rotate: tuple:
        (counter-clockwise, clockwise) degree amounts to augment image rotation
    horizontal_flip: float:
        probability of flipping an image horizontally; 0.5 is 50% chance
    zoom: tuple:
        (out, in) proportions to zoom in on the image; range 0. - 1. zooms out, 1.+ zooms in
    brightness: tuple:
        (low, high) amounts of augment brightness; range < 0 decreases, > 0 increases
    contrast: tuple:
        (low, high) amounts to augment contrast; range 0. - 1. decreases, 1.+ increases
    shear: tuple:
        maximum pixel amount to shear image in any direction
    aug: bool:
        if True, data augmentation will be applied to images and segmentation maps (applied simultaneously)
    """

    def __init__(self, dataframe, image_path, batch_size, resize_to=1024, shuffle=True, rotate=8,
                 horizontal_flip=0.5, zoom=0.15, brightness=10, contrast=0.2, shear=6, aug=True):
        """
        Constructs an instance of the SegGenerator class.

        ...
        Parameters
        ----------
        dataframe:
            the pandas dataframe object containing the image and rle mask data
        image_path: str:
            path to the folder containing images

        batch_size: int:
            number of training/validation examples in a batch
        resize_to: int:
            the input size of the network; images will be resized to this size
        shuffle: bool:
            if True, data will be randomly shuffled at the end of each epoch

        rotate: tuple:
            (counter-clockwise, clockwise) degree amounts to augment image rotation
        horizontal_flip: float:
            probability of flipping an image horizontally; 0.5 is 50% chance
        zoom: tuple:
            (out, in) proportions to zoom in on the image; range 0. - 1. zooms out, 1.+ zooms in
        brightness: tuple:
            (low, high) amounts of augment brightness; range < 0 decreases, > 0 increases
        contrast: tuple:
            (low, high) amounts to augment contrast; range 0. - 1. decreases, 1.+ increases
        shear: tuple:
            maximum pixel amount to shear image in any direction
        aug: bool:
            if True, data augmentation will be applied to images and segmentation maps (applied simultaneously)
        """

        # Data parameters
        self.df = dataframe
        self.image_path = image_path
        self.image_filenames = self.df.index.to_list()
        self.index = np.arange(len(self.image_filenames))

        # Model parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize_to = resize_to

        # Augmentation parameters
        # Any augmentor settings can be set to 0 in order to turn off that augmentation mode
        self.rotate = rotate
        self.horizontal_flip = horizontal_flip
        self.zoom = zoom
        self.brightness = brightness
        self.contrast = contrast
        self.shear = shear
        self.aug = aug

        # Create the Sequential image augmentation object
        self.seq = self.__create_seq()

        # Shuffle the data before starting if shuffling has been turned on
        self.on_epoch_end()

    def on_epoch_end(self):
        # Optionally shuffle the data at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.index)

    def __len__(self):
        # Number of gradient descent steps that will be taken per epoch
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, index):
        # Create a list of batch_size numerical indices
        indices = self.index[self.batch_size * index:self.batch_size * (index + 1)]
        if indices.size == 0:
            raise IndexError('Index not within possible range (0 to number of training steps)')
        # Generate the data
        x, y = self.__get_data(indices)
        return x, y

    def __get_data(self, batch_indices):
        # Initialise empty arrays for the training data and labels
        x = np.empty([self.batch_size, self.resize_to, self.resize_to, 1], dtype=np.uint8)
        y = np.empty([self.batch_size, self.resize_to, self.resize_to, 1], dtype=np.uint8)

        # Get the training data
        for i, index in enumerate(batch_indices):
            x[i, :, :, :] = self.__fname_to_px(index)
            y[i, :, :, :] = self.__fname_to_mask(index)

        # Apply data augmentation if option is turned on
        if self.aug:
            x, y = self.__aug(x, y)

        # Normalise pixels
        x = x.astype(np.float32)
        x /= 255.

        return x, y

    def __fname_to_mask(self, index):
        # Takes an image filename, gets the RLE mask and converts it into a pixel segmentation map with 1 and 0 values
        rle_mask = self.df.iloc[index].values[0]

        # If there is no ground truth mask, return an array of zeroes
        if rle_mask == '-1':
            return np.zeros([self.resize_to, self.resize_to, 1])

        # Convert from rle format to a pixel mask
        s = rle_mask.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[::2], s[1::2])]
        int_mask = np.zeros(1024 * 1024, dtype=np.uint8)
        current_pos = 0

        for start, length in zip(starts, lengths):
            current_pos += start
            int_mask[current_pos:current_pos + length] = 1
            current_pos += length

        int_mask = int_mask.reshape(1024, 1024).T

        # resize the pixel mask to be the same size as the image inputs
        if 1024 != self.resize_to:
            pil_data = Image.fromarray(int_mask)
            pil_data = pil_data.resize((self.resize_to, self.resize_to), resample=Image.NEAREST)
            int_mask = np.array(pil_data)

        int_mask = np.expand_dims(int_mask, axis=-1)

        return int_mask

    def __fname_to_px(self, index):
        # Loads an image in the dicom format, resizes it and converts it into a numpy array
        filename = self.image_filenames[index]
        dcm_file = dcmread(self.image_path + filename + '.dcm')
        dcm_pixel_data = dcm_file.pixel_array

        if dcm_pixel_data.shape[0] != self.resize_to:
            pil_data = Image.fromarray(dcm_pixel_data)
            pil_data = pil_data.resize((self.resize_to, self.resize_to))
            dcm_pixel_data = np.array(pil_data)

        dcm_pixel_data = np.expand_dims(dcm_pixel_data, axis=-1)

        return dcm_pixel_data

    def __create_seq(self):
        # This is the augmentation sequence; applied simultaneously to the images and the segmentation maps
        seq = iaa.Sequential([
            # Horizontal flip
            iaa.Fliplr(self.horizontal_flip),

            # Contrast
            iaa.LinearContrast((1 - self.contrast, 1 + self.contrast)),

            # Brightness
            iaa.Add((-self.brightness, self.brightness)),

            # Zoom, rotate, shear
            iaa.Affine(scale={'x': (1 - self.zoom, 1 + self.zoom), 'y': (1 - self.zoom, 1 + self.zoom)},
                       rotate=(-self.rotate, self.rotate),
                       shear=(-self.shear, self.shear))
        ])

        return seq

    def __aug(self, x, y):
        # The augmentation function that augments each image in a batch alongside its segmentation map
        x_aug = np.empty(x.shape, dtype=np.uint8)
        y_aug = np.empty(y.shape, dtype=np.uint8)

        for i in range(self.batch_size):
            # Move the axes to be in the imgaug format for segmentation masks
            current_seg = np.moveaxis(y[i, ...], -1, 0)
            current_seg = np.expand_dims(current_seg, axis=-1)
            x_aug[i, ...], y_aug[i, ...] = self.seq(image=x[i, ...], segmentation_maps=current_seg)

        return x_aug, y_aug


class ClassifierGenerator(Sequence):
    """
    Generator that returns a batch of images and their ground truth labels: 1 if pneumothorax is present in the image,
    and 0 if not.

    ...
    Methods
    _______
    The generator has no public methods and interacts directly with Keras to automatically return images and
    ground truth tensors. Note the required methods __len__ and __getitem__:
    __len__: the same thing as TRAIN_STEPS or TEST_STEPS; this is how many times the generator needs to return data
             in an epoch
    __getitem__: returns the images and labels as numpy arrays

    ...
    Attributes
    ----------
    dataframe:
        the pandas dataframe object containing the image and rle mask data
    image_path: str:
        path to the folder containing images

    batch_size: int:
        number of training/validation examples in a batch
    resize_to: int:
        the input size of the network; images will be resized to this size
    shuffle: bool:
        if True, data will be randomly shuffled at the end of each epoch

    rotate: tuple:
        (counter-clockwise, clockwise) degree amounts to augment image rotation
    horizontal_flip: float:
        probability of flipping an image horizontally; 0.5 is 50% chance
    zoom: tuple:
        (out, in) proportions to zoom in on the image; range 0. - 1. zooms out, 1.+ zooms in
    brightness: tuple:
        (low, high) amounts of augment brightness; range < 0 decreases, > 0 increases
    contrast: tuple:
        (low, high) amounts to augment contrast; range 0. - 1. decreases, 1.+ increases
    shear: tuple:
        maximum pixel amount to shear image in any direction
    aug: bool:
        if True, data augmentation will be applied to images
    """

    def __init__(self, dataframe, image_path, batch_size, resize_to=1024, shuffle=True, rotate=12,
                 horizontal_flip=0.5, zoom=0.20, brightness=15, contrast=0.2, shear=8, aug=True):
        """
        Constructs an instance of the ClassifierGenerator class.

        ...
        Parameters
        ----------
        dataframe:
            the pandas dataframe object containing the image and rle mask data
        image_path: str:
            path to the folder containing images

        batch_size: int:
            number of training/validation examples in a batch
        resize_to: int:
            the input size of the network; images will be resized to this size
        shuffle: bool:
            if True, data will be randomly shuffled at the end of each epoch

        rotate: tuple:
            (counter-clockwise, clockwise) degree amounts to augment image rotation
        horizontal_flip: float:
            probability of flipping an image horizontally; 0.5 is 50% chance
        zoom: tuple:
            (out, in) proportions to zoom in on the image; range 0. - 1. zooms out, 1.+ zooms in
        brightness: tuple:
            (low, high) amounts of augment brightness; range < 0 decreases, > 0 increases
        contrast: tuple:
            (low, high) amounts to augment contrast; range 0. - 1. decreases, 1.+ increases
        shear: tuple:
            maximum pixel amount to shear image in any direction
        aug: bool:
            if True, data augmentation will be applied to images
        """

        # Data parameters
        self.df = dataframe
        self.image_path = image_path
        self.image_filenames = self.df.index.to_list()
        self.index = np.arange(len(self.image_filenames))

        # Model parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize_to = resize_to

        # Augmentation parameters
        # Any augmentor settings can be set to 0 in order to turn off that augmentation mode
        self.rotate = rotate
        self.horizontal_flip = horizontal_flip
        self.zoom = zoom
        self.brightness = brightness
        self.contrast = contrast
        self.shear = shear
        self.aug = aug

        # Create the Sequential image augmentation object
        self.seq = self.__create_seq()

        # Shuffle the data before starting if shuffling has been turned on
        self.on_epoch_end()

    def on_epoch_end(self):
        # Optionally shuffle the data at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.index)

    def __len__(self):
        # Number of gradient descent steps that will be taken per epoch
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, index):
        # Create a list of batch_size numerical indices
        indices = self.index[self.batch_size * index:self.batch_size * (index + 1)]
        if indices.size == 0:
            raise IndexError('Index not within possible range (0 to number of training steps)')
        # Generate the data
        x, y = self.__get_data(indices)
        return x, y

    def __get_data(self, batch_indices):
        # Initialise empty arrays for the training data and labels
        x = np.empty([self.batch_size, self.resize_to, self.resize_to, 3], dtype=np.uint8)
        y = np.empty([self.batch_size, 1], dtype=np.uint8)

        # Get the training data
        for i, index in enumerate(batch_indices):
            x[i, :, :, :] = self.__fname_to_px(index)
            y[i, :] = self.__fname_to_label(index)

        # Apply data augmentation if option is turned on
        if self.aug:
            x = self.__aug(x)

        # Normalise pixels
        x = x.astype(np.float32)
        x /= 255.

        return x, y

    def __fname_to_px(self, index):
        # Loads an image in the dicom format, resizes it and converts it into a numpy array
        filename = self.image_filenames[index]
        dcm_file = dcmread(self.image_path + filename + '.dcm')
        dcm_pixel_data = dcm_file.pixel_array
        pil_data = Image.fromarray(dcm_pixel_data)

        if dcm_pixel_data.shape[0] != self.resize_to:
            pil_data = pil_data.resize((self.resize_to, self.resize_to))

        # Note that since DenseNet etc are pretrained on RGB images, they expect 3 channels and therefore we need
        # to convert the grayscale dicom images to RGB in order to maintain compatibility
        pil_data = pil_data.convert('RGB')
        dcm_pixel_data = np.array(pil_data)

        return dcm_pixel_data

    def __fname_to_label(self, index):
        # Get the ground truth label of an image
        try:
            # The classifier csv file was pre-processed to have 1 for pt, 0 for no pt
            label = self.df.iloc[index]['Class'].astype(np.uint8)
        except KeyError:
            # If a different csv file is used, this portion will check the rle value and assign non-pt if the rle is -1
            if self.df.iloc[index].values[0] == '-1':
                return 0
            else:
                return 1

        return label

    def __create_seq(self):
        # This is the augmentation sequence
        seq = iaa.Sequential([
            # Horizontal flip
            iaa.Fliplr(self.horizontal_flip),

            # Contrast
            iaa.LinearContrast((1 - self.contrast, 1 + self.contrast)),

            # Brightness
            iaa.Add((-self.brightness, self.brightness)),

            # Zoom, rotate, shear
            iaa.Affine(scale={'x': (1 - self.zoom, 1 + self.zoom), 'y': (1 - self.zoom, 1 + self.zoom)},
                       rotate=(-self.rotate, self.rotate),
                       shear=(-self.shear, self.shear))
        ])

        return seq

    def __aug(self, x):
        # The augmentation function that augments each image in a batch
        x_aug = np.empty(x.shape, dtype=np.uint8)

        for i in range(self.batch_size):
            # Carry out augmentation: images only
            x_aug[i, ...] = self.seq(image=x[i, ...])

        return x_aug
