# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE,
# Switzerland, Laboratory of Experimental Biophysics
# See the LICENSE.txt file for more details.

from keras import backend as K
import tensorflow as tf

def pixel_count_loss(lambda_factor=0.01):
    def loss(y_pred, y_true):
        pixel_loss = K.mean(K.sum(K.square(y_pred - y_true), axis=(1,2,3)), axis=0)
        count_loss = K.sum(y_pred, axis=(1,2,3)) - K.sum(y_true, axis=(1,2,3))
        count_loss = K.mean(K.square(count_loss), axis=0)
        return (pixel_loss + lambda_factor*count_loss)
    return loss

def pixel_loss(y_pred, y_true):
    return K.mean(K.sum(K.square(y_pred - y_true), axis=(1,2,3)), axis=0)

def pixel_area_loss(lambda_factor=0.01):
    def loss(y_pred, y_true):
        pixel_loss = K.mean(K.mean(K.square(y_pred - y_true), axis=(1,2,3)), axis=0)

        a_pred = tf.nn.avg_pool(y_pred, ksize=(1,12,12,1),
                                strides=(1,1,1,1), padding='VALID')
        a_true = tf.nn.avg_pool(y_true, ksize=(1,12,12,1),
                                strides=(1,1,1,1), padding='VALID')
        area_loss = K.mean(K.sum(K.square(a_pred - a_true), axis=(1,2,3)), axis=0)
        return pixel_loss + lambda_factor*area_loss
    return loss