from ...config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Reshape


class Sampling(Layer):
    def __init__(self):
        super(Sampling, self).__init__()
    
    def call(self, z_mean, z_log_var):
        batch_size, dim = z_mean.shape
        epsilon = tf.random.normal(shape=(batch_size, dim))

        return z_mean + (tf.exp(0.5*z_log_var) * epsilon)


class Encoder(Layer):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        self.config = model_config
        self.latent_dim = 2
        
        self.sampling = Sampling()
    
    def call(self, x):
        x = Dense(units=self.config.encoder.get('fc1_units'), activation="relu", name="encoder_fc1")(x)
        x = Dense(units=self.config.encoder.get('fc2_units'), activation="relu", name="encoder_fc2")(x)
        x = Dense(units=self.config.encoder.get('fc3_units'), activation="relu", name="encoder_fc3")(x)
        x = Flatten(name='flatten')(x)
        
        z_mean = Dense(units=self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(units=self.latent_dim, name='z_log_var')(x)
        
        return z_mean, z_log_var, self.sampling(z_mean, z_log_var)


class Decoder(Layer):
    def __init__(self, model_config):
        super(Decoder, self).__init__()
        self.model_config = model_config

    def call(self, x, output_dim):
        x = Dense(
            units=self.model_config.time_sequence * output_dim, 
            activation='relu', 
            name='decoder_fc0'
            )(x)
        x = Reshape(
            target_shape=(self.model_config.time_sequence, output_dim), 
            name='decoder_reshape'
            )(x)
        x = Dense(units=self.model_config.decoder.get('fc1_units'), activation="relu", name="decoder_fc1")(x)
        x = Dense(units=self.model_config.decoder.get('fc2_units'), activation="relu", name="decoder_fc2")(x)
        x = Dense(units=self.model_config.decoder.get('fc3_units'), activation="relu", name="decoder_fc3")(x)
        x = Dense(units=output_dim, activation="relu", name="decoder_output")(x)

        return x


class VAE(Layer):
    def __init__(self, config=model_config):
        super(VAE, self).__init__()
        self.config = model_config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
    
    def call(self, x):
        z_mean, z_log_var, encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output, x.shape[-1])

        return z_mean, z_log_var, decoder_output


class NormalityConfidenceWeight(Layer):
    def __init__(self, config=model_config):
        super(NormalityConfidenceWeight, self).__init__()
        self.config = config
        self.D0 = self.config.D0
    
    def call(self, x):
        x = tf.cast(x, dtype=tf.complex64)
        fft = tf.signal.fft(x[:, :, 0])
        A = tf.math.abs(fft)        
        L = tf.math.log(A)        
        _R = L - tf.reduce_mean(L, axis=-1, keepdims=True)
        _R = tf.cast(_R, dtype=tf.complex64)
        S = abs(tf.signal.ifft(_R))

        D = (S - tf.reduce_mean(S, axis=-1, keepdims=True)) / tf.reduce_mean(S, axis=-1, keepdims=True)
        w_n = 1 - 1 / (1 + tf.math.exp(-(D - self.D0)))

        return w_n[:, :, tf.newaxis]

