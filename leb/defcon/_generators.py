# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics, 2018
# See the LICENSE.txt file for more details.

import math
import os.path
import random

import numpy as np
from keras.utils import HDF5Matrix

from leb.defcon import _augmentors


#%%

def get_matrices(training_set_path):
    """Return three Keras HDF5Matrix instances for the input, ground-truth
    density map and ground-truth segmentation mask in a compact TrainingSet

    """
    if os.path.isfile(training_set_path):
        X = HDF5Matrix(training_set_path, 'input/input')
        y = HDF5Matrix(training_set_path, 'target/target')
        y_seg = HDF5Matrix(training_set_path, 'seg_map/seg_map')
        return X, y, y_seg
    else:
        raise Exception('Training set file not found.')

#%%
class DataIterator():
    """Build a generator that operates on a TrainingSet object"""

    def __init__(self, training_set, random_state=None):
        self.training_set = training_set
        np.random.seed(random_state)

    def augment(self, batch):
        #batch = augmentors.intensity_curves(batch)
        #batch = augmentors.contrast(batch)
        batch = _augmentors.brightness(batch)
        #batch = augmentors.gaussnoise(batch)
        return batch

    def flow(self, batch_size=32, output='both', crops=0):
        """Generate outputs from a non-compact TrainingSet to use with Keras'
        'fit_generator' function.

        If 'n_crops' is non-zero, the Iterator crops
        n_crops 20x20 regions from each image before feeding them.

        """
        while True:
            for dataset in self.input_sets:
                X = self.training_set['input/'+dataset]
                y = self.training_set['target/'+dataset]
                y_seg = self.training_set['seg_map/'+dataset]

                for i in range(int(math.ceil(X.shape[0]/2000))):
                    index = list(range(0,X.shape[0]))
                    sample = random.sample(index, batch_size)
                    sample.sort()
                    X_batch = X[sample, ...]
                    y_batch = y[sample, ...]
                    y_seg_batch = y_seg[sample, ...]
                    X_batch = self.augment(X_batch)

                    if crops > 0:
                        (X_batch, y_batch,
                         y_seg_batch) = _augmentors.random_crops(
                                X_batch, y_batch, y_seg_batch, n_crops=crops, crop_dim=20)

                    if output=='both':
                        yield (X_batch, [y_batch, y_seg_batch])
                    elif output=='seg':
                        yield (X_batch, y_seg)
                    elif output=='density':
                        yield (X_batch, y_batch)
                    else:
                        raise Exception('output must be "density", "seg" or "both"')

    def close(self):
        self.training_set.close()

#%%
class CompactIterator(DataIterator):
    """Builds an iterator that operates on a compact TrainingSet

    Builds an iterator that operates on a compact TrainingSet (FCNN.datasets).
    A TrainingSet is compacted using its "compact" method.

    """

    def flow(self, output='both', batch_size=32):
        X = np.array(self.training_set['input/input'])
        y = np.array(self.training_set['target/target'])
        y_seg = np.array(self.training_set['seg_map/seg_map'])

        while True:
            print('Shuffling batches')
            index = np.arange(X.shape[0])
            np.random.shuffle(index)
            X, y, y_seg = X[index], y[index], y_seg[index]

            i = 0
            while i < (X.shape[0] - batch_size):
            #while True:
                X_batch = X[i:i+batch_size, ...]
                y_batch = y[i:i+batch_size, ...]
                y_seg_batch = y_seg[i:i+batch_size, ...]
                X_batch = self.augment(X_batch)

                if output=='both':
                    yield (X_batch, [y_batch, y_seg_batch])
                elif output=='seg':
                    yield (X_batch, y_seg)
                elif output=='density':
                    yield (X_batch, y_batch)
                else:
                    raise Exception('output must be "density", "seg" or "both"')
                i += batch_size