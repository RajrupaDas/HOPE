import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path

# -----------------------------
# Load Data
# -----------------------------
X_train = np.load("../data/X_train.npy")
y_train = np.load("../data/y_train.npy")
X_test = np.load("../data/X_test.npy")
y_test = np.load("../data/y_test.npy")

# -----------------------------
# Define LSTM Model
# -----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32),
    Dense(3)  # Predicting [x, y, z]
])

model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
model.summary()

# -----------------------------
# Training
# -----------------------------
checkpoint_dir = Path("../models")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

model_path = checkpoint_dir / "lstm_orbit_predictor.keras"

checkpoint = ModelCheckpoint(
    filepath=str(model_path),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint]
)

# -----------------------------
# Evaluate on Test Set
# -----------------------------
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {loss:.6f}, MAE: {mae:.6f}")

# -----------------------------
# Predict and Save
# -----------------------------
predictions = model.predict(X_test)
np.save("../data/lstm_predictions.npy", predictions)
print("Predictions saved to data/lstm_predictions.npy")

