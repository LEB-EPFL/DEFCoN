# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics
# See the LICENSE.txt file for more details.

import warnings

from keras.models import load_model
from keras.layers import (AveragePooling2D, GlobalMaxPooling2D,
                          Lambda,
                          Input,
                          InputLayer)
from keras.models import Sequential
import keras.backend as K

import tensorflow as tf

from .models import DEFCoN
from .datasets import tiff_to_array

class FCN():
    # TODO Fix Tensorflow Runtime error for closed sessions that occurs after running either from_file or save_tf_model.
    """A class with the attributes of a Keras Model, and additional methods."""
    def __init__(self, model=DEFCoN()):
        """An FCN is initiated from a Keras Model object.

        It has a _max_count_layers boolean attributes, that indicates if it
        possesses maximum local count layers.
        """
        self.model = model
        # Does the model predict the max_count?
        if model.outputs[0].shape.ndims == 2:
            self._max_count_layers = True
        else:
            self._max_count_layers = False

    def __getattr__(self, attr):
        """FCN has all the attributes and functions of a Keras Model."""
        return getattr(self.model, attr)

    @classmethod
    def from_file(cls, h5_file, compile=False):
        """Create a FCN from a h5 file saved with Keras model.save()."""
        model = load_model(h5_file, compile=compile)
        return cls(model=model)

    def predict_tiff(self, input_tiff, batch_size=None, index=None):
        """Load a tiff image/stack, transform it and make a prediction."""
        data = tiff_to_array(input_tiff)
        if index is not None:
            y_pred = self.predict(data[index], batch_size=batch_size)
        else:
            y_pred = self.predict(data, batch_size=batch_size)
        return y_pred

    def density_to_max_count(self, pool=(7,7), strides=(1,1)):
        """Add layers in the end of the model to predict the maximum local
        count from the density map.

        The pool is the size of the regions to
        consider. The sliding window horizontal and vertical strides are given
        by 'strides'.

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
        """Remove the last layers, to predict density"""
        if self._max_count_layers == True:
            self.model.pop().pop().pop()
            self.model.build()
        else:
            warnings.warn('Model is already in "density" configuration')

    def save_tf_model(self, dir_name='tf_model', division=None):
        """Saves the graph and variables of the model as a tensoflow model
        directory.

        This function is used to output the tensorflow variables used for
        tensorflow serving in Java.

        """
        K.set_learning_phase(0)
        builder = tf.saved_model.builder.SavedModelBuilder(dir_name)

        input_tensor = tf.placeholder(tf.float32, shape=(None,None,None,1), name='input_tensor')
        input_layer = Input(tensor=input_tensor, name='input')
        x = self.model(input_layer)

        if division is not None:
            x = Lambda(lambda x: x/division)(x)

        x = tf.identity(x, name='output_tensor')

        with K.get_session() as sess:
            builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tf.saved_model.tag_constants.SERVING])
            builder.save()
