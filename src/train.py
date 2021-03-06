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
import models
import data_augmentation


def _train():
    (x_train, y_train), (x_test, y_test), (_, _) = dataset.load_dataset(config.DATASET_PATH)

    # NHW to NHWC
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Float to on hot vector
    y_train = keras.utils.to_categorical(y_train, dataset.CLASS_NUM)
    y_test = keras.utils.to_categorical(y_test, dataset.CLASS_NUM)

    if config.NETWORK == "fer_small":
        model = models.fer_small()
    elif config.NETWORK == "fer_middle":
        model = models.fer_middle()
    elif config.NETWORK == "resnet_50":
        model = models.resnet_50()
    elif config.NETWORK == "mobilenet_small":
        model = models.mobilenet_small()

    # Instanciate Data Generator with Data Augmentation
    datagen = data_augmentation.generator()
    datagen.fit(x_train)


    model.compile(optimizer=keras.optimizers.Adam(lr=0.001,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=1e-8,
                                                  decay=1e-4,
                                                  amsgrad=False),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    checkpoint_path = "outputs/" + config.NETWORK + "-{epoch:04d}.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     verbose=1,
                                                     save_weights_only=True,
                                                     period=5)

    model.fit(datagen.flow(x_train, y_train, batch_size=config.BATCH_SIZE),
              epochs=config.EPOCH,
              callbacks=[cp_callback],
              validation_data=(x_test, y_test))

    # Save model as blobs
    model.save("trained_models/" + config.NETWORK + ".h5")

if __name__ == '__main__':
    _train()
