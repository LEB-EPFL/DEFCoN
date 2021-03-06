# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics, 2018.
# See the LICENSE.txt file for more details.

import configparser
import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from defcon import models
from defcon._generators import get_matrices
from defcon._losses import pixel_count_loss
from defcon.networks import FCN


def _tf_init(seed=42, gpu_fraction=0.4):
    """Initializes the random number generator and GPU configuration.

    Seeds every random number generator and configures the amount of graphics
    card memory to use (default 0.4, the right number depends on the GPU but
    should be between 0.3 and 0.8).

    This method is called to ensure reproducible results.

    """
    os.environ['PYTHONHASHSEED'] = '0'

    # Set seed for numpy RNG
    np.random.seed(seed)

    # Set seed for Python RNG
    random.seed(seed)

    # Force TensorFlow to use single thread, and only a fraction of gpu memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1,
                                  gpu_options=gpu_options)

    # Set the random seed for Tensorflow and Keras
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def train(config_file, architecture=models.DEFCoN):
    """Convenience function for training the DEFCoN network.

    The training configuration is stored in the ini file supplied as input. The
    segmentation network is trained first, then frozen, and then the full
    network (segmentation + density networks) is trained. The model is saved
    as a Keras model file.

    Parameters
    ----------
    config_file : str
        Path to the configuration file for training.
    architecture : func
        A function from leb.defcon.models that constructs and returns a
        Keras Model.

    See Also
    --------
    leb.defcon.models : Pre-defined DEFCoN architectures

    """
    _tf_init()

    # TODO Create the trained_models and weights directories specified in the config file if they don't exist.
    # Otherwise, we get a totally uncool error if they don't exist.
    # Configure the parser
    config = configparser.ConfigParser()
    config.read(config_file)

    # Change working directory to the config file directory
    initial_workdir = os.getcwd()
    configdir = os.path.split(config_file)[0]

    # TODO Change dir back if there is an error; otherwise we're stuck here
    os.chdir(configdir)

    # Configure the training outputs
    model_name = Path(config['General']['ModelName'])
    weight_dir = Path(config['General']['WeightDir'])
    weight_file = str(weight_dir / model_name)

    training_set = config['General']['TrainingSetPath']
    X, y, y_seg = get_matrices(training_set)

    # Build and compile the model for segmentation
    model = FCN(architecture(output='seg'))
    model.summary()
    model.compile(optimizer=Adam(lr=config['SegNet'].getfloat('AdamLR')),
                  loss='binary_crossentropy')

    # Train the segmentation network
    # iter_seg = SegIterator(TrainingSet(training_set))
    batch_size = config['SegNet'].getint('BatchSize')
    nEpochs = config['SegNet'].getint('NumEpochs')

    model.fit(X, y_seg,
              batch_size=batch_size,
              shuffle='batch',
              epochs=nEpochs,
              verbose=1,
              validation_split=config['SegNet'].getfloat('ValidationSplit'),
              callbacks=[ModelCheckpoint(weight_file,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         monitor='val_loss'),
                         EarlyStopping(patience=1)])

    # Train the full network
    model = FCN(architecture(output='density'))
    model.load_weights(weight_file, by_name=True)

    lambda_factor = config['DensityNet'].getfloat('LambdaFactor')
    model.compile(optimizer=Adam(lr=config['DensityNet'].getfloat('AdamLR')),
                  loss=pixel_count_loss(lambda_factor=lambda_factor))
    model.summary()

    batch_size = config['DensityNet'].getint('BatchSize')
    nEpochs = config['DensityNet'].getint('NumEpochs')
    output_dir = Path(config['General']['OutputDir'])
    model_file = str(output_dir / model_name)

    model.fit(X, y,
              batch_size=batch_size,
              shuffle='batch',
              epochs=nEpochs,
              verbose=1,
              validation_split=config['DensityNet'].getfloat('ValidationSplit'),
              callbacks=[ModelCheckpoint(model_file,
                                         save_best_only=True,
                                         save_weights_only=False,
                                         monitor='val_loss'),
                         EarlyStopping(patience=1)])
    # Back to the initial working directory
    os.chdir(initial_workdir)


if __name__ == '__main__':
    train(sys.argv[1])
