"""
DNN Model Implementations
"""

from tensorflow import keras
from tensorflow.keras import layers

from dataset import IMAGE_SIZE, CLASS_NUM
from blocks import conv_bn_act


def fer_small():
    """
    Fer Small: 7 Conv + 1 Fully Connected Layers
    """
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='img', dtype="float32")
    x = conv_bn_act(32, 3, "valid", inputs)
    x = conv_bn_act(64, 3, "valid", x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = conv_bn_act(64, 3, 'same', block_1_output)
    x = conv_bn_act(64, 3, 'same', x)
    block_2_output = layers.add([x, block_1_output])

    x = conv_bn_act(64, 3, 'same', block_2_output)
    x = conv_bn_act(64, 3, 'same', x)
    block_3_output = layers.add([x, block_2_output])

    x = conv_bn_act(64, 3, "valid", block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(CLASS_NUM, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='resnt_8')
    model.summary()
    return model


def resnet_50():
    """
    ResNet50
    """
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='img', dtype="float32")
    x = keras.applications.resnet50.ResNet50(include_top=False,
                                             weights=None,
                                             input_tensor=inputs,
                                             input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                                             pooling="avg",
                                             classes=7)
    x = layers.Dense(256, activation='relu')(x.output)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(CLASS_NUM, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name='resnet_50')
    model.summary()
    return model
