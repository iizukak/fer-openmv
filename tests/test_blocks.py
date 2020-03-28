from tensorflow import keras
import tensorflow as tf
import blocks


def test_conv_bn_act_same():
    inputs = keras.Input(shape=(48, 48, 1), name='img', dtype="float32")
    block = blocks.conv_bn_act(64, 3, "same", inputs)
    assert type(block) == tf.Tensor
    assert block.shape[1] == 48
    assert block.shape[2] == 48
    assert block.shape[3] == 64


def test_conv_bn_act_valid():
    inputs = keras.Input(shape=(48, 48, 1), name='img', dtype="float32")
    block = blocks.conv_bn_act(64, 3, "valid", inputs)
    assert type(block) == tf.Tensor
    assert block.shape[1] == 46
    assert block.shape[2] == 46
    assert block.shape[3] == 64
