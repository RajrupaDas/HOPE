# src/train_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load truth data
states = np.load("data/truth_orbit.npy")
X, y = states[:-1], states[1:]

# reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))
y = y.reshape((y.shape[0], y.shape[1]))

# LSTM model
model = models.Sequential([
    layers.LSTM(128, activation="tanh", input_shape=(X.shape[1], X.shape[2])),
    layers.Dense(y.shape[1])
])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=100, batch_size=64, verbose=1)

# Predict
y_pred = model.predict(X)
np.save("data/lstm_pred.npy", y_pred)
print("Saved LSTM predictions → data/lstm_pred.npy")

