import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib  # to save scaler for later use (optional)

# -----------------------------
# Load CSV
# -----------------------------
csv_path = Path(__file__).resolve().parent.parent / "data" / "leo_orbit_with_j2_FIXED.csv"
df = pd.read_csv(csv_path)

# -----------------------------
# Select only position features: x, y, z
# -----------------------------
pos_df = df[["x (km)", "y (km)", "z (km)"]].copy()

# -----------------------------
# Normalize the data
# -----------------------------
scaler = MinMaxScaler()
normalized = scaler.fit_transform(pos_df)

# Save the scaler to reuse later in inference
joblib.dump(scaler, "../data/position_scaler.save")

# -----------------------------
# Create time series (t-5 → t)
# -----------------------------
WINDOW_SIZE = 5

X = []
y = []

for i in range(WINDOW_SIZE, len(normalized)):
    X.append(normalized[i-WINDOW_SIZE:i])  # shape: (5, 3)
    y.append(normalized[i])                # shape: (3,)

X = np.array(X)  # shape: (samples, 5, 3)
y = np.array(y)  # shape: (samples, 3)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# -----------------------------
# Split into train/test
# -----------------------------
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# -----------------------------
# Save for model training
# -----------------------------
np.save("../data/X_train.npy", X_train)
np.save("../data/y_train.npy", y_train)
np.save("../data/X_test.npy", X_test)
np.save("../data/y_test.npy", y_test)

print("✅ Time series dataset ready and saved in /data")



