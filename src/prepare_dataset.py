import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

csv_path = Path(__file__).resolve().parent.parent / "data" / "leo_orbit_with_j2_FIXED.csv"
df = pd.read_csv(csv_path)

pos_df = df[["x (km)", "y (km)", "z (km)"]].copy()

scaler = MinMaxScaler()
normalized = scaler.fit_transform(pos_df)

joblib.dump(scaler, "../data/position_scaler.save")

WINDOW_SIZE = 5

X = []
y = []

for i in range(WINDOW_SIZE, len(normalized)):
    X.append(normalized[i-WINDOW_SIZE:i])
    y.append(normalized[i])

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}, y shape: {y.shape}")

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

np.save("../data/X_train.npy", X_train)
np.save("../data/y_train.npy", y_train)
np.save("../data/X_test.npy", X_test)
np.save("../data/y_test.npy", y_test)

print(" Time series dataset ready and saved in /data")


