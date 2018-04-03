# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics
# See the LICENSE.txt file for more details.


import keras.backend as K
from keras.layers import (Conv2D,
                          Conv3D,
                          Input,
                          Dropout,
                          Lambda,
                          TimeDistributed,
                          UpSampling3D)
from keras.models import Model

from ._layers import convReLU, deconvReLU, convReLU_3D


def DEFCoN(input_size=(None,None,1), output='density'):
    """Keras Model architecture for leb.

    The inputs are histogram-normalized by a Lambda layer. output='seg' builds only
    the segmentation network. output='density' builds the entire architecture with
    just the denstiy map output. 'both' builds the architecture with both outputs
    for simultaneous training (not recommended).

    """
    def normalization(x):
        x = (x - K.min(x, axis=(1,2,3), keepdims=True))/(K.max(x, axis=(1,2,3), keepdims=True) - K.min(x, axis=(1,2,3), keepdims=True))
        return x

    input_ = Input(shape=input_size, name='input')
    hist_norm = Lambda(normalization, name='hist_norm')(input_)
    conv_seg_1 = convReLU(16, name='conv_seg_1')(hist_norm)
    conv_stride_1 = convReLU(16, strides=(2,2), name='conv_stride_1')(conv_seg_1)
    conv_seg_2 = convReLU(32, name='conv_seg_2')(conv_stride_1)
    conv_stride_2 = convReLU(32, strides=(2,2), name='conv_stride_2')(conv_seg_2)
    conv_seg_3 = convReLU(64, name='conv_seg_3')(conv_stride_2)
    dropout_seg = Dropout(0.5, name='dropout_seg')(conv_seg_3)
    deconv_seg_1 = deconvReLU(8, name='deconv_seg_1')(dropout_seg)
    deconv_seg_2 = deconvReLU(8, name='decon_seg_2')(deconv_seg_1)
    seg = Conv2D(1, kernel_size=(1,1),
                 use_bias= False,
                 padding='same',
                 activation='sigmoid',
                 name='seg')(deconv_seg_2)

    conv_1 = convReLU(16, name='conv_1')(seg)
    conv_2 = convReLU(16, strides=(2,2), name='conv_2')(conv_1)
    conv_3 = convReLU(32, name='conv_3')(conv_2)
    conv_4 = convReLU(32, strides=(2,2), name='conv_4')(conv_3)
    conv_5 = convReLU(64, kernel=(5,5), name='conv_5')(conv_4)
    dropout = Dropout(0.5, name='dropout')(conv_5)
    deconv_1 = deconvReLU(8, name='deconv_1')(dropout)
    deconv_2 = deconvReLU(8, name='deconv_2')(deconv_1)
    density = Conv2D(1, kernel_size=(1,1),
                     use_bias= False,
                     padding='same',
                     activation='linear',
                     name='density')(deconv_2)

    if output == 'density':
        model = Model(inputs=input_, outputs=density)
        for layer in model.layers[2:11]:
            layer.trainable = False
    elif output == 'seg':
        model = Model(inputs=input_, outputs=seg)
    elif output == 'both':
        model = Model(inputs=input_, outputs=[density, seg])
    else:
        raise Exception('output must be "density", "seg" or "both".')
    return model

#%%
def DEFCoN_3D(input_size=(3,None,None,1), output='density'):

    input_ = Input(shape=input_size, name='input')
    conv_seg_1 = TimeDistributed(convReLU(16, name='conv_seg_1'))(input_)
    conv_stride_1 = TimeDistributed(convReLU(16, kernel=(3,3), strides=(2,2), name='conv_stride_1'))(conv_seg_1)
    conv_seg_2 = TimeDistributed(convReLU(32, name='conv_seg_2'))(conv_stride_1)
    conv_stride_2 = TimeDistributed(convReLU(32, kernel=(3,3), strides=(2,2), name='conv_stride_2'))(conv_seg_2)
    conv_seg_3 = TimeDistributed(convReLU(64, kernel=(3,3), name='conv_seg_3'))(conv_stride_2)
    dropout_seg = TimeDistributed(Dropout(0.5, name='dropout_seg'))(conv_seg_3)
    deconv_seg_1 = TimeDistributed(deconvReLU(8, name='deconv_seg_1'))(dropout_seg)
    deconv_seg_2 = TimeDistributed(deconvReLU(8, name='decon_seg_2'))(deconv_seg_1)
    seg = TimeDistributed(Conv2D(2, kernel_size=(1,1),
                 use_bias= False,
                 padding='same',
                 activation='softmax',
                 name='seg'))(deconv_seg_2)

    conv3D_1 = convReLU_3D(16, name='conv3D_1')(seg)
    conv3D_2 = convReLU_3D(16, strides=(1,2,2), name='conv3D_2')(conv3D_1)
    conv3D_3 = convReLU_3D(32, name='conv3D_3')(conv3D_2)
    conv3D_4 = convReLU_3D(32, strides=(1,2,2), name='conv3D_4')(conv3D_3)
    conv3D_5 = convReLU_3D(64, name='conv3D_5')(conv3D_4)
    dropout = Dropout(0.5, name='dropout')(conv3D_5)
    up3D_1 = UpSampling3D(name='up3D_1')(dropout)
    deconv3D_1 = convReLU_3D(8, name='deconv3D_1')(up3D_1)
    up3D_2 = UpSampling3D(name='up3D_2')(deconv3D_1)
    deconv3D_2 = convReLU_3D(8, name='deconv3D_2')(up3D_2)
    density3D = Conv3D(1, kernel_size=(1,1,1),
                     use_bias= False,
                     padding='same',
                     activation='linear',
                     name='density3D')(deconv3D_2)

    if output == 'density':
        model = Model(inputs=input_, outputs=density3D)
        for layer in model.layers[2:10]:
            layer.trainable = False
    elif output == 'seg':
        model = Model(inputs=input_, outputs=seg)
    elif output == 'both':
        model = Model(inputs=input_, outputs=[density3D, seg])
    else:
        raise Exception('output must be "density", "seg" or "both".')
    return model