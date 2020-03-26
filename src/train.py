"""
Main training loop script

Before execute this script, Please confirm your config file.
We need dataset downloaded before execute this file.

Usage:
    $ python3 src/train.py
"""

from tensorflow import keras
import tensorflow as tf

import config
import dataset
import resnet


def _train():
    (x_train, y_train), (x_test, y_test), (_, _) = dataset.load_dataset(config.DATASET_PATH)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = keras.utils.to_categorical(y_train, config.CLASS_NUM)
    y_test = keras.utils.to_categorical(y_test, config.CLASS_NUM)

    model = resnet.resnet_8()
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=config.BATCH_SIZE,
              epochs=config.EPOCH,
              validation_split=0.1)


if __name__ == '__main__':
    _train()
