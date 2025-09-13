# src/train_physlstm.py
import numpy as np
from scipy.integrate import solve_ivp
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Physical constants
# -------------------------------
mu = 398600.4418  # km^3/s^2
dt = 60.0  # seconds

# -------------------------------
# Load truth orbit (full dataset)
# -------------------------------
states = np.load("data/truth_orbit.npy")  # shape (timesteps, 6)
timesteps = len(states)

# -------------------------------
# RK4 baseline (2-body only)
# -------------------------------
def two_body(t, y):
    x, y_, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y_**2 + z**2)
    ax = -mu * x / r**3
    ay = -mu * y_ / r**3
    az = -mu * z / r**3
    return [vx, vy, vz, ax, ay, az]

rk4_states = [states[0]]  # start with initial condition

for i in range(1, timesteps):
    y_prev = rk4_states[-1]
    sol = solve_ivp(two_body, [0, dt], y_prev, method='RK45', t_eval=[dt], rtol=1e-9, atol=1e-12)
    rk4_states.append(np.array(sol.y).flatten())
rk4_states = np.array(rk4_states)

# -------------------------------
# Compute residuals and scale
# -------------------------------
residuals = states - rk4_states
scaler = MinMaxScaler()
residuals_scaled = scaler.fit_transform(residuals)

# -------------------------------
# Prepare sliding window sequences
# -------------------------------
seq_len = 10
X_seq, y_seq = [], []
for i in range(len(rk4_states) - seq_len):
    X_seq.append(rk4_states[i:i+seq_len])
    y_seq.append(residuals_scaled[i + seq_len])
X_seq = np.array(X_seq)  # shape: (samples, seq_len, 6)
y_seq = np.array(y_seq)  # shape: (samples, 6)

# -------------------------------
# Build Hybrid LSTM model
# -------------------------------
model = models.Sequential([
    layers.LSTM(256, return_sequences=True, input_shape=(seq_len, 6), dropout=0.2),
    layers.LSTM(128, dropout=0.2),
    layers.Dense(6)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# -------------------------------
# Train with EarlyStopping
# -------------------------------
early_stop = callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
model.fit(X_seq, y_seq, epochs=400, batch_size=64, verbose=1, callbacks=[early_stop])

# -------------------------------
# Predict hybrid residuals
# -------------------------------
resid_pred_scaled = model.predict(X_seq, batch_size=64)

# Inverse scale to km
resid_pred = scaler.inverse_transform(resid_pred_scaled)

# Align with RK4 baseline
hybrid_pred = rk4_states[seq_len:] + resid_pred

# -------------------------------
# Save hybrid predictions
# -------------------------------
np.save("data/hybrid_pred.npy", hybrid_pred)
print("✅ Hybrid LSTM prediction saved (inverse-scaled to km)")

