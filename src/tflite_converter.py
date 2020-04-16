"""
Convert Keras modes(.h5) to TensorFlow Lite model(.tflite)
"""

from tensorflow import keras
import tensorflow as tf
import config
import dataset


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


def convert_quant(model):
    """
    Convert Keras model to quantized tflite

    reference:
    https://www.tensorflow.org/lite/performance/post_training_quantization
    """
    (x_train, _), (_, _), (_, _) = dataset.load_dataset(config.DATASET_PATH)
    x_train = x_train.astype('float32')

    # Calibration for 1 epoch
    def representative_dataset_gen():
      for i in range(len(x_train)):
        # Get sample input data as a numpy array in a method of your choosing.
        # Format NHW
        yield [x_train[i][tf.newaxis, ..., tf.newaxis]]

    
    model_path = "trained_models/" + config.NETWORK + ".h5"
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    open("trained_models/" + config.NETWORK + "_quant.tflite", "wb").write(tflite_quant_model)


if __name__ == '__main__':
    model = load_model()
    print("convert tflite start")
    convert(model)
    print("convert tflite finished")
    print("convert tflite quantize start")
    convert_quant(model)
    print("convert tflite quantize end")
