# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics
# See the LICENSE.txt file for more details.

from keras.layers import (Conv2D,
                          Conv2DTranspose,
                          Conv3D,
                          MaxPooling2D)

#%% convReLU
def convReLU(filters, kernel=(3,3), strides=(1,1), **kwargs):
    conv = Conv2D(filters, kernel_size=kernel, strides=strides,
                   padding='same',
                   activation='relu',
                   kernel_initializer='orthogonal',
                   **kwargs)
    return conv

def deconvReLU(filters, kernel=(3,3), strides=(2,2), **kwargs):
    deconv = Conv2DTranspose(filters,
                             kernel_size=kernel,
                             strides=strides,
                             padding='same',
                             kernel_initializer='orthogonal',
                             activation='relu',
                             **kwargs)
    return deconv

def convReLU_3D(filters, kernel=(3,3,3), strides=(1,1,3), **kwargs):
    conv_3D = Conv3D(filters, kernel_size=kernel, strides=strides,
                   padding='same',
                   kernel_initializer='orthogonal',
                   activation='relu',
                   **kwargs)
    return conv_3D