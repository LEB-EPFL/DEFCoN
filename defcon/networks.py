# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics, 2018
# See the LICENSE.txt file for more details.

import warnings

import keras.backend as K
import tensorflow as tf
from keras.layers import (AveragePooling2D, GlobalMaxPooling2D,
                          Lambda,
                          Input,
                          InputLayer)
from keras.models import Sequential
from keras.models import load_model

from .datasets import tiff_to_array
from .models import DEFCoN


class FCN():
    # TODO Fix Tensorflow Runtime error for closed sessions that occurs after running either from_file or save_tf_model.
    """Interface to a fully convolutional network with additional methods.

    A FCN wraps a Keras model and provides additional functionality.

    Parameters
    ----------
    model : keras.models.Model
        The model's representation implemented in Keras.

    """
    def __init__(self, model=DEFCoN()):
        self.model = model
        # Does the model predict the max_count?
        if model.outputs[0].shape.ndims == 2:
            self._max_count_layers = True
        else:
            self._max_count_layers = False

    def __getattr__(self, attr):
        """FCN has all the attributes and functions of a Keras Model.

        """
        return getattr(self.model, attr)

    @classmethod
    def from_file(cls, input_file):
        """Creates a new FCN from a Keras model stored in a HDF file.

        The data in the HDF file would have been saved with Keras model.save().

        Parameters
        ----------
        input_file : str
            The path to the HDF file containing the model data.


        """
        # compile must be False to prevent Keras from raising an error.
        model = load_model(input_file, compile=False)
        return cls(model=model)

    def predict_tiff(self, input_file, index=None, **kwargs):
        """Load a tiff image/stack, transform it and make a prediction.

        Parameters
        ----------
        input_file : str
            The path to the TIF file.
        index
            The index of the image in the TIF stack to run a prediction on. If
            this is None, prediction will be performed on all images.

        All kwargs are passed to the predict method from keras.models.Model.

        """
        data = tiff_to_array(input_file)
        if index is not None:
            y_pred = self.predict(data[index], **kwargs)
        else:
            y_pred = self.predict(data, **kwargs)
        return y_pred

    def density_to_max_count(self, pool=(7, 7), strides=(1, 1)):
        """Add layers to predict the maximum local count from the density map.

        The pool is the size of the regions to consider. The sliding window
        horizontal and vertical strides are given by 'strides'.

        Parameters
        ----------
        pool : tuple
            A 2-tuple of integers defining the size of the region over which
            the maximum local count will be computed.
        strides : tuple
            A 2-tuple of integers defining the strides of the sliding window in
            the horizontal and vertical directions.

        """
        if self._max_count_layers == False:
            new_model = Sequential()
            new_model.add(InputLayer(input_shape=(None, None, 1), name="input"))
            for layer in self.model.layers[1:]:
                new_model.add(layer)

            new_model.add(AveragePooling2D(pool_size=pool,
                                           strides=strides,
                                           padding='valid',
                                           name='av_pool'))
            new_model.add(Lambda(lambda x: x*pool[0]*pool[1], name='mult'))
            new_model.add(GlobalMaxPooling2D(name='max_count_output'))
            new_model.build()
            self.model = new_model
            self._max_count_layers = True
        else:
            warnings.warn('Model is already in "max_count" configuration')

    def max_count_to_density(self):
        """Removes max count layers to revert the FCN to a density predictor.

        """
        if self._max_count_layers == True:
            self.model.pop().pop().pop()
            self.model.build()
        else:
            warnings.warn('Model is already in "density" configuration')

    def save_tf_model(self, export_dir='tf_model', division=None):
        """Saves the FCN as a TensorFlow graph.

        This function may be used to output the underlying TensorFlow
        representation of the model so that it may imported into other
        applications.

        Parameters
        ----------
        export_dir : str
            The path to the directory where the model will be exported.
        division : func
            # TODO Add docstring for this parameter.


        """
        K.set_learning_phase(0)
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        input_tensor = tf.placeholder(
            tf.float32,
            shape=(None, None, None, 1),
            name='input_tensor')
        input_layer = Input(tensor=input_tensor, name='input')
        x = self.model(input_layer)

        if division is not None:
            x = Lambda(lambda x: x/division)(x)

        x = tf.identity(x, name='output_tensor')

        with K.get_session() as sess:
            builder.add_meta_graph_and_variables(
                sess=sess,
                tags=[tf.saved_model.tag_constants.SERVING])
            builder.save()
