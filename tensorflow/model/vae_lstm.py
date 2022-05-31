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

