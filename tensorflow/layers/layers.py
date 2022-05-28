from ...config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Reshape


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


class Decoder(Layer):
    def __init__(self, model_config):
        super(Decoder, self).__init__()
        self.model_config = model_config

    def call(self, x, decoder_units1):
        x = Dense(
            units=self.model_config.decoder.get('time_sequence') * decoder_units1, 
            activation='relu', 
            name='decoder_fc0'
            )(x)
        x = Reshape(
            target_shape=(self.model_config.decoder.get('time_sequence'), self.model_config.decoder.get('fc1_units')), 
            name='decoder_reshape'
            )(x)
        x = Dense(units=self.model_config.decoder.get('fc1_units'), activation="relu", name="decoder_fc1")(x)
        x = Dense(units=self.model_config.decoder.get('fc2_units'), activation="relu", name="decoder_fc2")(x)
        x = Dense(units=self.model_config.decoder.get('fc3_units'), activation="relu", name="decoder_fc3")(x)

        return x


class VAE(Layer):
    def __init__(self, model_config):
        super(VAE, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.decoder = Decoder(model_config)
    
    def call(self, x):
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output, x.shape[-1])

        return decoder_output
