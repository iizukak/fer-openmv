"""
Convert Keras modes(.h5) to TensorFlow Lite model(.tflite)
"""

from tensorflow import keras
import tensorflow as tf
import config


def load_model():
    """
    Load trained model from Keras (.h5) file
    Model path is obtained from config file
    """
    model_path = "trained_models/" + config.NETWORK + ".h5"
    model = keras.models.load_model(model_path)
    model.summary()
    return model


def convert(model):
    """
    Convert Keras model to .tflite
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("trained_models/" + config.NETWORK + ".tflite", "wb").write(tflite_model)


if __name__ == '__main__':
    model = load_model()
    convert(model)
