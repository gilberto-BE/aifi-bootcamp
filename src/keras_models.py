import tensorflow as tf


class KerasLSTM:
    def __init__(self, input_shape=1, output_shape=1, learning_rate=0.03):
        self.model = tf.keras.layers.Sequential()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate

    def create_model(self):
        pass