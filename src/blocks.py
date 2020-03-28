"""
Blocks contain TensorFlow layers
"""

import tensorflow as tf

from tensorflow.keras import layers


def conv_bn_act(filters, kernel_size, padding, inputs):
    """
    This block contains Conv2D, BatchNormalizationand Activation(relu)
    """
    x = layers.Conv2D(filters, kernel_size, activation=None, use_bias=False, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    return x
