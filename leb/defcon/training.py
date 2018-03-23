# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics
# See the LICENSE.txt file for more details.

import numpy as np
import configparser
import os
import random
import sys

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf

from leb.defcon import models
from leb.defcon.losses import pixel_count_loss
from leb.defcon.networks import FCN
from leb.defcon.generators import get_matrices

# Reproducible results
def tf_init(seed=42, gpu_fraction=0.4):
    """Initialize RNG and GPU configuration.

    Seed every random number generator, and configure the amount of graphic
    memory to use (default 0.4, the right number depends on the GPU but should
    be between 0.3 and 0.8).
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

#%% Training leb
def train(config_file):
    """Train leb in two times.

    The training configuration is stored in the ini file config_file. The
    segmentation network is trained first, then frozen, and the density
    network is trained alone. The model is saved as a Keras h5 model file.

    """
    tf_init()

    #%% Config parser
    config = configparser.ConfigParser()
    config.read(config_file)

    # Change working directory to the config file directory
    initial_workdir = os.getcwd()
    configdir = os.path.split(config_file)[0]

    # TODO Change dir back if there is an error; otherwise we're stuck here
    os.chdir(configdir)

    #%% Sets
    model_name = config['General']['ModelName']
    weight_dir = config['General']['WeightDir']
    if (weight_dir[-1] == '/'):
        weight_dir = weight_dir[:-1]
    weight_file = weight_dir + '/' + model_name

    training_set = config['General']['TrainingSetPath']
    X, y, y_seg = get_matrices(training_set)

    architecture = models.DEFCoN

    #%% Model
    model = FCN(architecture(output='seg'))
    model.summary()

    model.compile(optimizer=Adam(lr=config['SegNet'].getfloat('AdamLR')),
                  loss='binary_crossentropy')

    #%% Training segmentation

    #iter_seg = SegIterator(TrainingSet(training_set))
    batch_size = config['SegNet'].getint('BatchSize')
    nEpochs = config['SegNet'].getint('NumEpochs')

    model.fit(X, y_seg,
              batch_size=batch_size,
              shuffle='batch',
              epochs=nEpochs,
              verbose=1,
              validation_split = config['SegNet'].getfloat('ValidationSplit'),
              callbacks=[ModelCheckpoint(weight_file,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         monitor='val_loss'),
                         EarlyStopping(patience=1)])

    #%% Training density
    model = FCN(architecture(output='density'))
    model.load_weights(weight_file, by_name=True)

    lambda_factor = config['DensityNet'].getfloat('LambdaFactor')
    model.compile(optimizer=Adam(lr=config['DensityNet'].getfloat('AdamLR')),
                loss=pixel_count_loss(lambda_factor=lambda_factor))
    model.summary()

    batch_size = config['DensityNet'].getint('BatchSize')
    nEpochs = config['DensityNet'].getint('NumEpochs')
    output_dir = config['General']['OutputDir']
    if (output_dir[-1] == '/'):
        output_dir = output_dir[:-1]
    model_file = output_dir + '/' + model_name

    model.fit(X, y,
              batch_size=batch_size,
              shuffle='batch',
              epochs=nEpochs,
              verbose=1,
              validation_split = config['DensityNet'].getfloat('ValidationSplit'),
              callbacks=[ModelCheckpoint(model_file,
                                         save_best_only=True,
                                         save_weights_only=False,
                                         monitor='val_loss'),
                         EarlyStopping(patience=1)])
    # Back to the initial working directory
    os.chdir(initial_workdir)

if __name__ == '__main__':
  train(sys.argv[1])