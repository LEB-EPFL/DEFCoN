# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics, 2018
# See the LICENSE.txt file for more details.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pkg_resources
import tifffile

import defcon.networks as networks


class TestFCNMethods(unittest.TestCase):
    """Test case for methods of the FCN class.

    """
    def setUp(self):
        """Obtains the path to the DEFCoN model file and loads it.

        """
        # The DEFCoN model
        defcon_filename = pkg_resources.resource_filename('defcon.resources', 'defcon_tf13.h5')
        self.model = networks.FCN.from_file(defcon_filename)

    def test_init(self):
        """Can a FCN be initialized from scratch?

        """
        model = networks.FCN()

    def test_density_to_max_count(self):
        """Checks that a FCN may be correctly converted to a max count network.

        """
        model = self.model
        assert (not model._max_count_layers), 'Error: Expected FCN max_count_layers to be False.'
        old_num_layers = len(model.model.layers)

        model.density_to_max_count()
        assert model._max_count_layers, 'Error: Expected FCN max_count_layers to be True.'
        new_num_layers = len(model.model.layers)
        assert new_num_layers - old_num_layers == 3, 'Error: Incorrect number of layers.'


class TestFCNMethodsIO(unittest.TestCase):
    """Test case for FCN methods requiring input/output.

    """
    def setUp(self):
        """Creates a DEFCoN model and a test image file with a single fluorophore.

        """
        # The DEFCoN model
        defcon_filename = pkg_resources.resource_filename('defcon.resources', 'defcon_tf13.h5')
        self.model = networks.FCN.from_file(defcon_filename)

        # Creates an image with a single Gaussian kernel in the center.
        self.tmp_tiff_stack = tempfile.mkstemp(suffix='.tif')[1]
        x = np.linspace(-10, 10, 32)
        y = np.linspace(-10, 10, 32)
        x, y = np.meshgrid(x, y)
        sigma_x = 1
        sigma_y = 1
        self.ground_truth = 1 / 2 / sigma_x / sigma_y / np.pi * np.exp(
            -x ** 2 / 2 / sigma_x ** 2 - y ** 2 / 2 / sigma_y ** 2)
        self.ground_truth = np.uint16(self.ground_truth * 500).reshape(1, 32, 32) + 100
        self.ground_truth += np.random.randint(0, 5, size=(32, 32), dtype='uint16')
        tifffile.imsave(
            self.tmp_tiff_stack,
            self.ground_truth,
            dtype='uint16')

        # Output tensor flow graph.
        self.output_dir = tempfile.mkdtemp()
        self.output_dir = str(Path(self.output_dir) / Path('tf_output'))

    def test_predict_tiff(self):
        """Verify that prediction from an image in a TIF file is approximately correct.

        """
        y_pred = self.model.predict_tiff(self.tmp_tiff_stack)
        dens = np.sum(y_pred.squeeze())
        np.testing.assert_approx_equal(1, dens, significant=1)

    def test_save_tf_model(self):
        """Does the TensorFlow graph output without error?

        """
        self.model.save_tf_model(self.output_dir)

        saved_model = Path(self.output_dir) / Path('saved_model.pb')
        variables_data = Path(self.output_dir) / Path('variables/variables.data-00000-of-00001')
        variables_index = Path(self.output_dir) / Path('variables/variables.index')

        assert saved_model.exists(), 'Error: Saved model file does not exist.'
        assert variables_data.exists(), 'Error: Variables data file does not exist.'
        assert variables_index.exists(), 'Error: Variables index file does not exist.'
