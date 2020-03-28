"""
ResNet Model Implementation

- ResNet-8
- ResNet-M
- ResNet-50

ResNet 50 are came from original paper.
https://arxiv.org/abs/1512.03385

ResNet-8 is from TensorFlow Tutorials
https://www.tensorflow.org/guide/keras/functional#a_toy_resnet_model
"""

from tensorflow import keras
from tensorflow.keras import layers

from config import IMAGE_SIZE, CLASS_NUM


def resnet_8():
    """
    ResNet 8: 7 Conv + 1 Fully Connected Layers
    """
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='img', dtype="float32")
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(CLASS_NUM, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='resnt_8')
    model.summary()
    return model


def resnet_m():
    """
    ResNet M: Middle Size Residual Network
    """
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='img', dtype="float32")
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    # block 2
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])
    block_2_skipconnection =\
        layers.Conv2D(128, 1, activation='relu', padding='same')(block_2_output)

    # block 3
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_skipconnection])
    block_3_skipconnection =\
        layers.Conv2D(256, 1, activation='relu', padding='same')(block_3_output)

    # block 4
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(block_3_output)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    block_4_output = layers.add([x, block_3_skipconnection])
    block_4_skipconnection =\
        layers.Conv2D(512, 1, activation='relu', padding='same')(block_4_output)
 
    # block 5
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(block_4_output)
    x = layers.Conv2D(523, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    block_5_output = layers.add([x, block_4_skipconnection])

    x = layers.Conv2D(512, 3, activation='relu')(block_5_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
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
