from ...config.config import model_config
from ..layers.layers import NormalityConfidenceWeight, Sampling

from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, LSTM, Dense, Reshape
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import Model


def build_encoder(x, config=model_config):
    input = Input(shape=x.shape[1:], batch_size=x.shape[0])
    x = Dense(units=config.encoder.get('fc1_units'), activation="relu", name="encoder_fc1")(input)
    x = Dense(units=config.encoder.get('fc2_units'), activation="relu", name="encoder_fc2")(x)
    x = Dense(units=config.encoder.get('fc3_units'), activation="relu", name="encoder_fc3")(x)
    x = Flatten(name='flatten')(x)
    
    z_mean = Dense(units=config.latent_dim, name='z_mean')(x)
    z_log_var = Dense(units=config.latent_dim, name='z_log_var')(x)
    
    return Model(input, [z_mean, z_log_var, Sampling()(z_mean, z_log_var)], name='encoder')


def build_decoder(x, config=model_config):
    input = Input(shape=config.latent_dim, batch_size=x.shape[0])
    _x = Dense(
        units=config.time_sequence * x.shape[-1], 
        activation='relu', 
        name='decoder_fc0'
        )(input)
    _x = Reshape(
        target_shape=(config.time_sequence, x.shape[-1]), 
        name='decoder_reshape'
        )(_x)
    _x = Dense(units=config.decoder.get('fc1_units'), activation="relu", name="decoder_fc1")(_x)
    _x = Dense(units=config.decoder.get('fc2_units'), activation="relu", name="decoder_fc2")(_x)
    _x = Dense(units=config.decoder.get('fc3_units'), activation="relu", name="decoder_fc3")(_x)
    output = Dense(units=x.shape[-1], activation="relu", name="decoder_output")(_x)
    
    return Model(input, output, name='decoder')


def build_vae_model(config=model_config):
    input = Input(shape=(config.time_sequence, 1), batch_size=config.batch_size)
    z_mean, z_log_var, encoder_output = build_encoder(input)(input)
    decoder_output = build_decoder(input)(encoder_output)
    
    return Model(input, [z_mean, z_log_var, decoder_output])


def build_lstm_layer(config=model_config):
    input = Input(shape=(config.time_sequence, 1), batch_size=config.batch_size)
    x = LSTM(units=config.h_t, return_sequences=True)(input)
    output = Dense(units=1)(x)

    return Model(input, output, name='lstm')


class VAELSTM(Model):
    def __init__(self, model_config):
        super(VAELSTM, self).__init__()
        self.config = model_config

        self.model = build_lstm_layer()
        self.vae = build_vae_model()
        self.normality_confidence_weight = NormalityConfidenceWeight()

        self.reconstruction_loss = MeanSquaredError(name='reconstruction_loss')
        self.kl_loss = MeanSquaredError(name='kl_loss')
        self.vae_loss = MeanSquaredError(name='vae_loss')
        self.lstm_loss = MeanSquaredError(name='lstm_loss')
        self.pad_loss = MeanSquaredError(name='pad_loss')
        self.f1 = MeanSquaredError(name='f1')
    
    @property
    def metrics(self):
        return [
            self.reconstruction_loss,
            self.kl_loss,
            self.vae_loss,
            self.lstm_loss,
            self.pad_loss,
            self.f1
        ]

    def train_step(self, data, real, anomal_true):
        with tf.GradientTape as tape:
            z_mean, z_log_var, reconstructed_data = self.vae(data)
            w_n = self.normality_confidence_weight(data)

            reconstruction_loss = tf.sqrt(tf.reduce_sum(tf.square((w_n * (data - reconstructed_data)))))
            _kl_loss = 0.5 * self.config.regularization.get('beta') * tf.reduce_mean(w_n) * (-tf.math.log(tf.square(z_log_var)) + tf.square(z_mean) + tf.square(z_log_var) -1)
            kl_loss = tf.reduce_mean(tf.reduce_sum(_kl_loss, axis=1))
            vae_loss = reconstruction_loss + kl_loss

            abs_diff = tf.abs(data - reconstructed_data)
            predicted_anormal_point = tf.where(abs_diff > (self.config.k*tf.math.reduce_std(abs_diff, axis=1, keepdims=True)), 1, 0)
            f1 = f1_score(anomal_true, predicted_anormal_point)
            self.f1.update_state(f1)

        grads = tape.gradient(vae_loss, self.vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.vae.trainable_weights))
        

        self.reconstruction_loss.update_state(reconstruction_loss)
        self.kl_loss.update_state(kl_loss)
        self.vae_loss.update_state(vae_loss)

        with tf.GradientTape as tape:
            y = self.model(reconstructed_data)
            
            lstm_loss = self.config.regularization.get('lambda') * tf.reduce_mean(w_n) * tf.sqrt(tf.reduce_sum(tf.square(real - y)))
            pad_loss = vae_loss + lstm_loss
        
        grads = tape.gradient(lstm_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.lstm_loss.update_state(lstm_loss)
        self.pad_loss.update_state(pad_loss)

        return {
            'reconstruction_loss': self.reconstruction_loss.result(),
            'kl_loss': self.kl_loss.result(),
            'vae_loss': self.vae_loss.result(),
            'lstm_loss': self.lstm_loss.result(),
            'pad_loss': self.pad_loss.result(),
            'f1': f1
        }
