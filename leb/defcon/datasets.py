# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics, 2018
# See the LICENSE.txt file for more details.

import os

import h5py
import numpy as np
from skimage import io

from leb.defcon import augmentors
from .labeling import gen_map_stack


class TrainingSet(h5py.File):
    """Creates an interface to HDF files that contain training data for DEFCoN.

    A TrainingSet is a subclass of h5py.File and contains two groups: "input"
    and "target." These are, respectively, the input images and density map
    targets for the FCN.

    All arguments to the constructor are passed directly to the h5py.File
    constructor.

    Examples
    --------
    Create an empty TrainingSet and print its summary, overwriting any
    existing HDF file.

    >>> from leb.defcon import datasets
    >>> with datasets.TrainingSet('training_data.h5', mode='w') as f:
    >>>     f.summary()

    """
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        for group in ['input', 'target']:
            if group not in self:
                self.create_group(group)

    def __repr__(self):
        r = '{0}(\'{1}\', mode=\'{2}\')'
        return r.format(self.__class__.__name__, self.filename, self.mode)


    def add_input(self, input_file, name=None, input_dataset=None):
        """Adds a new raw dataset to the HDF file.

         The input data most originate either from a TIF stack or another
         HDF5 dataset.

         Parameters
         ----------
         input_file : str
            The name of the file containing the dataset. The file must end in
            one of the following extensions: tif, tiff, h5, hdf5, or he5.
         name : str
            The name to give to the dataset. If None, the filename of the
            input_file is used.
         input_dataset : str
            The name of the dataset inside the original HDF file to add to this
            TrainingSet. This is ignored if adding files from a TIF stack.

         """
        filename, ext = os.path.splitext(input_file)
        if name is None:
            name = filename

        if ext.lower() in ['.tif', '.tiff']:
            data = tiff_to_array(input_file)
        elif ext.lower() in ['.h5', '.hdf5', '.he5']:
            with h5py.File(input_file, 'r') as file:
                data = np.array(file[input_dataset])
        else:
            raise DatasetError('Invalid file type. Supported formats: '
                               'TIFF, HDF5')

        self['input'].create_dataset(name=name, data=data)

    def add_target(self, input_file, input_dataset, name=None):
        """Adds a new target dataset to the TrainingSet file from another HDF.

        Parameters
        ----------
        input_file : str
            The name of the file containing the dataset. The file must end in
            one of the following extensions: tif, tiff, h5, hdf5, or he5.
        input_dataset : str
            The name of the dataset inside the original HDF file to add to this
            TrainingSet.
        name : str
            The name to give to the dataset. If None, the filename of the
            input_file is used.

        """
        if name is None:
            filename = os.path.splitext(input_file)[0]
            name = filename
        with h5py.File(input_file, 'r') as file:
            self['target'].create_dataset(name=name, data=file[input_dataset])

    def _create_target(self,
                       input_dataset,
                       frame_logger,
                       threshold=250,
                       sigma=1):
        """Creates a set of target density maps from ground truth data.
        
        Parameters
        ----------
        input_dataset
        frame_logger
        threshold
        sigma

        """
        # If input_set is 'input/my_dataset', returns only 'my_dataset'
        input_dataset = input_dataset.split('/')[-1]
        # Check that the dataset exists
        if 'input/' + input_dataset not in self:
            raise DatasetError('Dataset: ' + input_dataset
                               + 'is not in "/input/" inside ' + self.filename)

        (n_frames, x_max, y_max) = self['input/' + input_dataset].shape[:3]

        density = gen_map_stack(frame_logger,
                                n_frames,
                                x_max, y_max,
                                threshold=threshold,
                                sigma=sigma)
        density = density[:, :, :, np.newaxis]
        self['target'].create_dataset(name=input_dataset, data=density)

    def _create_seg_map(self, input_set, threshold=0.03):
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
        if 'input/' + input_set not in self:
            raise DatasetError('Dataset: ' + input_set
                               + 'is not in "/input/" inside ' + self.filename)

        y = np.array(self['target/' + input_set])
        segmap = np.zeros(shape=(y.shape), dtype='int')
        segmap[y > threshold] = 1
        segmap[y <= threshold] = 0

        ## Uncomment these 2 lines if using the original softmax layer
        # segmap_inv = 1-segmap
        # segmap = np.concatenate((segmap_inv, segmap), axis=3)
        self['seg_map'].create_dataset(name=input_set, data=segmap)

    def add_dataset(self,
                    name,
                    input_file,
                    ground_truth,
                    brightness_threshold=250,
                    sigma=1,
                    seg_threshold=0.03):
        """Adds a stack of images to the TrainingSet and builds the targets.

        After adding the raw data to the TrainingSet, the corresponding
        segmentation and target maps are built and added as well.

        Parameters
        ----------
        name : str
            The name to give to the dataset inside the HDF5 file.
        input_file : str
            The name of the file containing the input training data.
        ground_truth: str or pandas.DataFrame
            Path to a CSV file or a pandas.DataFrame containing the ground
            truth data. The column names must include frame, x, and y. An
            optional column may be included with the name brightness that
            specifies the number of photons emitted by a fluorophore in a
            given frame. Fluorophore positions should be specified in pixels.
        brightness_threshold : float
            Fluorophores with a ground truth brightness smaller than this value
            will be excluded from the segmentation and target density maps.
        sigma : float
            The standard deviation of the Gaussian kernels from which the target
            density maps will be constructed.
        seg_threshold: float
            The threshold of the binary transform to apply to the target density
            map when building the segmentation maps.

        """

        self.add_input(input_file, name=name)
        self._create_target(input_dataset=name,
                            frame_logger=ground_truth,
                            threshold=brightness_threshold,
                            sigma=sigma)
        self._create_seg_map(input_set=name, threshold=seg_threshold)

    def summary(self):
        """Prints a summary of every dataset in the TrainingSet.

        """
        print('\n' + self.filename)
        print(self.name)
        tab = '    '
        for group in self:
            print(tab + group + '/')
            for dataset in self[group]:
                data = self[group + '/' + dataset]
                print(2 * tab + dataset + ': shape = ' + str(data.shape)
                      + ', dtype = ' + str(data.dtype))
        print('')

    def lengths(self):
        """"Returns the lengths of every input dataset in the TrainingSet.

        Returns
        -------
        l : array_like
            A list of the lengths of the input datasets.

        """
        l = []
        for i, dataset in enumerate(self['input']):
            l.append(len(self['input/' + dataset]))
        return l


def compact_set(
        training_set,
        output_file,
        shuffle=False,
        augment=False,
        temporal=False):
    """Creates a compact TrainingSet from a normal TrainingSet.

    This requires that every dataset is composed of images of the same
    dimensions. A new HDF file will be created.

    In a TrainingSet, the datasets are organized as follows:

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

    For faster training, it can be converted to a compact set:
    input/
        input
    seg_map/
        seg_map
    target/
        target

    Parameters
    ----------
    training_set : leb.defcon.TrainingSet
        An open TrainingSet.
    output_file : str
        The filename for the new HDF file that will containing the compacted
        training data.
    shuffle : bool
        If 'shuffle' is True, the different sets are shuffled together.
    augment : bool
        If True, the images in 'input' are augmented by randomly changing the
        brightness.
    temporal : bool
        If True, then the another dimension is added to the training set that
        accounts for correlated changes in the data over time.

    """
    with TrainingSet(output_file, 'w') as output:

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
                    data_array = data_array[dim_n % 3:, ...]
                    # Convert to RGB stacks
                    data_array = np.reshape(data_array,
                                            (-1, 3, dim_x, dim_y, 1))

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


def tiff_to_array(input_file):
    """Transforms a tiff image/stack into a (n, x, y, 1) numpy array of floats.

    This function is used to convert tiff images to input arrays. The inputs
    are converted to floats and normalized between 0 and 1.

    Parameters
    ----------
    input_file : str
        Path and filename to the TIF image/stack.

    Returns
    -------
    data : array_like
        The normalized array of images.

    """
    # TODO Add unit test for this method.
    data = io.imread(input_file)
    if data.ndim == 3:
        data = data[:, :, :, np.newaxis]
    elif data.ndim == 2:
        data = data[np.newaxis, :, :, np.newaxis]
    else:
        raise ImageDimError('Tiff image should be grayscale, and 2D '
                            '(3D if stack)')
    # Converting from uint to float
    if data.dtype == 'uint8':
        max_uint = 255
    elif data.dtype == 'uint16':
        max_uint = 2 ** 16 - 1
    else:
        raise ImageTypeError('Tiff image type should be uint8 or uint16')
    data = data.astype('float')
    data /= max_uint
    return data


# Exceptions
class ImageTypeError(Exception):
    pass


class ImageDimError(Exception):
    pass


class DatasetError(Exception):
    pass
