# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics, 2018
# See the LICENSE.txt file for more details.

import unittest

import pkg_resources

import defcon.networks as networks


class TestFCNMethods(unittest.TestCase):
    """Test case for methods of the FCN class.

    """

    def setUp(self):
        pass

    def test_init(self):
        """Can a FCN be initialized from scratch?

        """
        model = networks.FCN()


class TestFCNMethodsIO(unittest.TestCase):
    """Test case for FCN methods requiring input/output.

    """

    def setUp(self):
        """Obtains the full path to the DEFCoN model file.

        """
        self.defcon_filename = pkg_resources.resource_filename('defcon.resources', 'defcon_tf13.h5')

    def test_defcon_from_file(self):
        """Can the DEFCoN FCN model be retrieved from a HDF file?

        """
        model = networks.FCN.from_file(self.defcon_filename)
