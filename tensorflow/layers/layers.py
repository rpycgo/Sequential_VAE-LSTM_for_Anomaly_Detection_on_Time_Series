from ...config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten
class Sampling(Layer):
    def __init__(self):
        super(Sampling, self).__init__()
    
    def call(self, z_mean, z_log_var):
        batch, dim = z_mean.shape[:2]
        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + (tf.exp(0.5*z_log_var) * epsilon)


class Encoder(Layer):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        self.model_config = model_config
        self.latent_dim = 2
        
        self.sampling = Sampling()
    
    def call(self, x):
        x = Dense(units=x.shape[-1], activation="relu", name="encoder_fc1")(x)
        x = Dense(units=self.model_config.encoder.get('fc3_units'), activation="relu", name="encoder_fc2")(x)
        x = Dense(units=self.model_config.encoder.get('fc3_units'), activation="relu", name="encoder_fc3")(x)
        x = Flatten(name='flatten')(x)
        
        z_mean = Dense(units=self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(units=self.latent_dim, name='z_log_var')(x)
        
        return self.sampling(z_mean, z_log_var)

