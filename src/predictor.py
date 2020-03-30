import tensorflow as tf

import config
import dataset


class Predictor():
    """
    Inference class

    Reference
    https://www.tensorflow.org/lite/convert/python_api
    """
    def __init__(self):
        self.interpreter = \
                tf.lite.Interpreter(model_path="trained_models/" + config.NETWORK + ".tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, input_data):
        """
        Data is HW format
        """

        # Set input data to interpreter
        input_data = input_data / 255
        input_data = input_data[tf.newaxis, ..., tf.newaxis]
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Execute inference
        self.interpreter.invoke()
        results = self.interpreter.get_tensor(self.output_details[0]['index'])
        return results[0]
