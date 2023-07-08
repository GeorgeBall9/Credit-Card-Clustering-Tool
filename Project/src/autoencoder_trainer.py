import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model


class AutoencoderTrainer:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.autoencoder = None

    def build_autoencoder(self) -> keras.Model:
        input_dim = self.dataset.shape[1]
        encoding_dim = input_dim // 2

        input_layer = keras.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = keras.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

        return self.autoencoder

    def train_autoencoder(self, autoencoder: keras.Model) -> keras.Model:
        autoencoder.fit(self.dataset, self.dataset, epochs=100, batch_size=32, verbose=1)
        return autoencoder

    def encode_data(self, autoencoder: keras.Model) -> np.ndarray:
        encoder = keras.Model(autoencoder.input, autoencoder.layers[1].output)
        encoded_data = encoder.predict(self.dataset)
        return encoded_data
    
    def save_model(self, filepath):
        if self.autoencoder is None:
            raise Exception("Autoencoder not built. Nothing to save")
        self.autoencoder.save(filepath)
        
    def load_model(self, filepath):
        self.autoencoder = load_model(filepath)
