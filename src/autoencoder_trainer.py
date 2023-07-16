import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



class AutoencoderTrainer:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.autoencoder = None

    def build_autoencoder(self) -> keras.Model:
        input_dim = self.dataset.shape[1]
        encoding_dim = 2

        input_layer = keras.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = keras.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

        return self.autoencoder

    def train_autoencoder(self, autoencoder: keras.Model) -> keras.Model:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model_checkpoint = ModelCheckpoint('TrainedAutoencoder/best_model.h5', monitor='val_loss', save_best_only=True)

        autoencoder.fit(self.dataset, self.dataset,
                        epochs=110,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
        return autoencoder
    
    def reconstruction_error(self):
        if self.autoencoder is None:
            raise Exception("Autoencoder not built. Cannot calculate reconstruction error")
        reconstructed_data = self.autoencoder.predict(self.dataset)
        mse = np.mean(np.power(self.dataset - reconstructed_data, 2), axis=1)
        print(f"Mean Squared Error: {np.mean(mse)}")

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
