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


def mobilenet_v1_block(filters, inputs):
    """
    This block implement MobileNetV1 block.
    It's contains 3x3 Conv (DepthWise) -> BN -> ReLU -> Conv 1x1 -> BN -> ReLU -> 3x3 Conv (DepthWise + Stride 2)
    """
    # First Depthwise Block
    x = layers.DepthwiseConv2D(3, strides=1, activation=None, use_bias=False, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
 
    # Second 1x1 Block
    x = layers.Conv2D(filters, 1, activation=None, use_bias=False, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # First Depthwise Block
    x = layers.DepthwiseConv2D(3, strides=2, activation=None, use_bias=False, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
 
    return x
