# lstm_model.py
"""
Build and train an LSTM model to predict satellite position (x, y, z) from past 5 time steps.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
X_train = np.load("../data/X_train.npy")
y_train = np.load("../data/y_train.npy")
X_test = np.load("../data/X_test.npy")
y_test = np.load("../data/y_test.npy")

# Model definition
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(5, 3)),
    LSTM(32),
    Dense(3)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

# Save the model
model.save("../models/lstm_orbit_predictor.keras")

