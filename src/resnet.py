"""
ResNet Model Implementation

- ResNet-8
- ResNet-18
- ResNet-50

ResNet-18 and 50 are came from original paper.
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

    model = keras.Model(inputs, outputs, name='toy_resnet')
    model.summary()
    return model
