import h5py
from skimage import io
import numpy as np
import os

from .labeling import get_frame_gt, gen_map_stack
from DEFCoN import augmentors

#%% Dataset class

class TrainingSet(h5py.File):
    """A TrainingSet is a h5py.File containing two groups "input" and "target", the
    input images and density map targets for the FCN respectively

    """

    def __init__(self, name, mode=None):
        h5py.File.__init__(self, name, mode=mode)
        for group in ['input', 'target']:
            if group not in self:
                self.create_group(group)

    def add_input(self, input_file, name=None, input_dataset=None):
        """Add a new dataset to the h5 from a tiff stack or a HDF5 dataset"""
        filename, ext = os.path.splitext(input_file)
        if name is None:
            name=filename

        if ext.lower() in ['.tif', '.tiff']:
            data = tiff_to_array(input_file)
        elif ext.lower() in ['.h5', '.hdf5', '.he5']:
            file = h5py.File(input_file, 'r')
            data = np.array(file[input_dataset])
        else:
            raise DatasetError('Invalid file type. Supported formats: TIFF, HDF5')

        self['input'].create_dataset(name=name, data=data)

    def add_target(self, h5_path, dataset, name=None):
        """Add a new target set to the h5 from a h5 dataset"""
        if name is None:
            filename = os.path.splitext(h5_path)[0]
            name=filename
        file = h5py.File(h5_path, 'r')
        self['target'].create_dataset(name=name, data=file[dataset])
        file.close()

    def create_target(self, input_set, frame_logger, threshold=250, sigma=1):
        """Create a density map stack corresponding to input_set given the
        SASS position and state loggers

        """
        # If input_set is 'input/my_dataset', returns only 'my_dataset'
        input_set = input_set.split('/')[-1]
        # Check that the dataset exists
        if 'input/'+input_set not in self:
            raise DatasetError('Dataset: ' + input_set
                                +'is not in "/input/" inside ' + self.filename)

        (n_frames, x_max, y_max) = self['input/'+input_set].shape[:3]

        density = gen_map_stack(frame_logger,
                                n_frames,
                                x_max, y_max,
                                threshold=threshold,
                                sigma=sigma)
        density = density[:, :, :, np.newaxis]
        self['target'].create_dataset(name=input_set, data=density)

    def create_seg_map(self, input_set, threshold=0.03):
        """Add a segmentation map to the TrainingSet for the set 'input_set'.

        This method creates a segmentation mask for the set 'input_set' using
        the density map as a basis. The density map is thresholded at the value
        defined by the 'threshold' argument (default 0.03), with every value
        superior set to 1 and every value inferior set to 0. This mask is saved
        as a numpy array of int, under 'seg_map'+input_set.
        """
        if 'seg_map' not in self:
            self.create_group('seg_map')

        # If input_set is 'input/my_dataset', returns only 'my_dataset'
        input_set = input_set.split('/')[-1]
        # Check that the dataset exists
        if 'input/'+input_set not in self:
            raise DatasetError('Dataset: ' + input_set
                                +'is not in "/input/" inside ' + self.filename)

        y = np.array(self['target/' + input_set])
        segmap = np.zeros(shape=(y.shape), dtype='int')
        segmap[y > threshold] = 1
        segmap[y <= threshold] = 0

        ## Uncomment these 2 lines if using the original softmax layer
        #segmap_inv = 1-segmap
        #segmap = np.concatenate((segmap_inv, segmap), axis=3)
        self['seg_map'].create_dataset(name=input_set, data=segmap)

    def add_dataset(self, name,
                    input_file,
                    frame_logger,
                    brightness_threshold=250,
                    sigma=1,
                    seg_threshold=0.03):
        """From a tiff stack and a frame logger, add an input set, a
        segmentation map stack and a density map stack to a TrainingSet.

        """

        self.add_input(input_file, name=name)
        self.create_target(input_set=name,
                           frame_logger=frame_logger,
                           threshold=brightness_threshold,
                           sigma=sigma)
        self.create_seg_map(input_set=name, threshold=seg_threshold)

    def summary(self):
        """Print a summary of every dataset in self."""
        print('\n' + self.filename)
        print(self.name)
        tab = '    '
        for group in self:
            print(tab + group + '/')
            for dataset in self[group]:
                data = self[group + '/' + dataset]
                print(2*tab + dataset + ': shape = '+ str(data.shape)
                      + ', dtype = ' + str(data.dtype))
        print('')

    def lengths(self):
        """"Returns the lenghts of every input set in length in an array of int."""
        l = []
        for i, dataset in enumerate(self['input']):
            l.append(len(self['input/'+dataset]))
        return l

#%%
def compact_set(training_set, output_file, shuffle=False, augment=False, temporal=False):
    """Create a compact TrainingSet from a normal TrainingSet.

    In a TrainingSet build normally, the datasets are organized as follows:

    input/
        set_1
        set_2
        set_3
    seg_map/
        set_1
        set_2
        set_3
    target/
        set_1
        set_2
        set_3

    For faster training, i can be converted to a compact set:
    input/
        input
    seg_map/
        seg_map
    target/
        target

    This requires that every dataset is composed of images of the same dimensions.
    output_file is the h5 file that will contain the compact TrainingSet. If
    'shuffle' is True, The different sets are shuffled together. If 'augment' is
    True, the images in 'input' are augmented by randomly changing the brightness.
    If 'temporal' is True, the training set is added another dimension, that
    accounts for the time distribution.
    """

    output = TrainingSet(output_file, 'w')

    for n_group, group in enumerate(training_set):
        # If there is a seg_map group, creates it in the output file
        if group == 'seg_map':
            output.create_group('seg_map')

        print('Compacting ' + group + ' data...')
        for i, dataset in enumerate(training_set[group]):
            print('    ' + dataset + '...')
            path = group + '/' + dataset
            data_array = np.array(training_set[path])

            if temporal is True:
                # Make sure that data_array is divisible by 3
                (dim_n, dim_x, dim_y, dim_c) = data_array.shape
                data_array = data_array[dim_n%3:,...]
                # Convert to RGB stacks
                data_array = np.reshape(data_array, (-1, 3, dim_x, dim_y, 1))

            if i == 0:
                compact_set = data_array
            else:
                compact_set = np.concatenate((compact_set, data_array))

        if shuffle is True:
            if n_group == 0:
                index = np.arange(compact_set.shape[0])
                np.random.shuffle(index)
            compact_set = compact_set[index]
        if augment is True:
            if group == 'input':
                print('Augmenting data...')
                compact_set = augmentors.brightness(compact_set)
        output.create_dataset(group + '/' + group, data=compact_set)
    output.close()

#%%
def tiff_to_array(inputTiff):
    """Transform a tiff image/stack into a (n, x, y, 1) numpy array of floats.

    This function is used to convert tiff images to input vectors. The inputs
    are converted to floats and normalized between 0 and 1."""
    data = io.imread(inputTiff)
    if data.ndim == 3:
        data = data[:, :, :, np.newaxis]
    elif data.ndim == 2:
        data = data[np.newaxis, :, :, np.newaxis]
    else:
        raise ImageDimError('Tiff image should be grayscale, and 2D (3D if stack)')
    # Converting from uint to float
    if data.dtype == 'uint8':
        max_uint = 255
    elif data.dtype == 'uint16':
        max_uint = 2**16-1
    else:
        raise ImageTypeError('Tiff image type should be uint8 or uint16')
    data = data.astype('float')
    data /= max_uint
    return data

#%% Exceptions
class ImageTypeError(Exception):
    pass

class ImageDimError(Exception):
    pass

class DatasetError(Exception):
    pass