import tensorflow as tf
from tensorflow.keras.layers import Layer
class Sampling(Layer):
    def __init__(self):
        super(Sampling, self).__init__()
    
    def call(self, z_mean, z_log_var):
        batch, dim = z_mean.shape[:2]
        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + (tf.exp(0.5*z_log_var) * epsilon)


