# src/preprocess_data.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
states = np.load("data/truth_orbit.npy")  # shape (timesteps, 6)

# Scale data to [0,1]
scaler = MinMaxScaler()
states_scaled = scaler.fit_transform(states)

np.save("data/truth_orbit_scaled.npy", states_scaled)
np.save("data/scaler.npy", scaler)
print("✅ Scaled orbit data saved")

