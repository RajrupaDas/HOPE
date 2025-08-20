import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# === Load Data ===
# Ground truth (CSV with columns: time (s), x (km), y (km), z (km), vx, vy, vz)
truth_df = pd.read_csv("../data/leo_orbit_with_j2_FIXED.csv")
truth_time = truth_df["time (s)"].values
truth_positions = truth_df[["x (km)", "y (km)", "z (km)"]].values

# RK4 numerical integration results
rk4_positions = np.load("../data/rk4_positions.npy")

# LSTM predictions (scaled)
lstm_predictions = np.load("../data/lstm_predictions.npy")

# Load scaler and invert scaling
scaler = joblib.load("../data/position_scaler.save")
lstm_positions = scaler.inverse_transform(lstm_predictions)

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(truth_time, truth_positions[:, 0], label="Truth (x)", color="black", linestyle="--")
plt.plot(truth_time[:rk4_positions.shape[0]], rk4_positions[:, 0], label="RK4 (x)", color="blue")
plt.plot(truth_time[:lstm_positions.shape[0]], lstm_positions[:, 0], label="LSTM (x)", color="red")

plt.xlabel("Time (s)")
plt.ylabel("x-position (km)")
plt.title("Truth vs RK4 vs LSTM - X coordinate")
plt.legend()
plt.grid(True)
plt.show()

