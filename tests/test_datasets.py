# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics, 2018
# See the LICENSE.txt file for more details.

import tempfile
import unittest

import h5py
import numpy as np
import numpy.testing as npt
import tifffile

import leb.defcon.datasets as datasets


class TestTrainingSetMethods(unittest.TestCase):
    """Test case for TrainingSet method that do not require file I/O.

    """
    def setUp(self):
        self.tmp_training_set = tempfile.mkstemp()[1]

    def test_init(self):
        """Tests the creation of a new TrainingSet."""
        with datasets.TrainingSet(self.tmp_training_set, mode='w') as test_set:
            pass


class TestTrainingSetMethods_IO(unittest.TestCase):
    """Test case for tests requiring file input/output.

    """
    def setUp(self):
        self.tmp_training_set = tempfile.mkstemp()[1]

        self.tmp_tiff_stack = tempfile.mkstemp(suffix='.tif')[1]
        tifffile.imsave(self.tmp_tiff_stack, shape=(2, 2, 1), dtype='uint8')

        self.tmp_training_set_with_data = tempfile.mkstemp()[1]
        with h5py.File(self.tmp_training_set_with_data, 'w') as f:
            f['input/dset1'] = np.random.randint(0, 255, (16, 16, 1))

    def test_add_input_tiff(self):
        """Tests the addition of a tif image to the dataset."""
        with datasets.TrainingSet(self.tmp_training_set, mode='w') as t:
            t.add_input(self.tmp_tiff_stack, 'test_tif')

        with h5py.File(self.tmp_training_set, 'r') as f:
            assert 'input/test_tif' in f.keys(), \
                'Error: Test image not found in training set!'

    def test_lengths(self):
        """Are the lengths of the input datasets correctly returned?"""
        with datasets.TrainingSet(self.tmp_training_set_with_data, 'r+') as t:
            lengths = t.lengths()

        assert (lengths[0], 1)

class TestTiffToArray(unittest.TestCase):
    """Test case for the tiff_to_array method.

    """

    def setUp(self):
        self.tmp_tiff_stack = tempfile.mkstemp(suffix='.tif')[1]
        self.ground_truth = np.arange(0, 8, dtype='uint8').reshape(2, 2, 2)
        tifffile.imsave(
            self.tmp_tiff_stack,
            self.ground_truth,
            dtype='uint8')

        self.ground_truth = self.ground_truth[:, :, :, np.newaxis]

    def test_tiff_to_array(self):
        """Tests the import of tif stacks."""
        data = datasets.tiff_to_array(self.tmp_tiff_stack)
        npt.assert_array_equal(data, self.ground_truth / 255)

    def test_tiff_to_array_unnormalized(self):
        """Tests the import of tif stacks without normalizing the values."""
        data_unnormalized = datasets.tiff_to_array(
            self.tmp_tiff_stack, normalize=False)
        npt.assert_array_equal(data_unnormalized, self.ground_truth)


if __name__ == '__main__':
    unittest.main()
