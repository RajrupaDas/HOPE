import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load

base_dir = Path(__file__).resolve().parent        
project_root = base_dir.parent                    
data_dir = project_root / "data"
models_dir = project_root / "models"
results_dir = project_root / "results"

results_dir.mkdir(parents=True, exist_ok=True)

y_test = np.load(data_dir / "y_test.npy")
y_pred = np.load(data_dir / "lstm_predictions.npy")
scaler = load(models_dir / "scaler.pkl")

# Load the scaler used in preprocessing
scaler = load("../models/scaler.pkl") # Inverse transform y_test & LSTM predictions to km
y_test_km = scaler.inverse_transform(y_test)
lstm_predictions_km = scaler.inverse_transform(lstm_predictions)

# If rk4_positions in km already skip inverse transform
# If in normalized units:
# rk4_positions_km = scaler.inverse_transform(rk4_positions)
rk4_positions_km = rk4_positions  # already in km

labels = ["X", "Y", "Z", "Vx", "Vy", "Vz"]

# Plot comparisons for each variable#
for i, label in enumerate(labels):
    plt.figure(figsize=(8, 6))
    plt.plot(y_test_km[:, i], label="True")
    plt.plot(lstm_predictions_km[:, i], '--', label="LSTM")
    plt.plot(rk4_positions_km[:, i], ':', label="RK4")
    plt.title(f"Prediction Comparison: {label}")
    plt.xlabel("Time step")
    plt.ylabel(f"{label} {'Position (km)' if i < 3 else 'Velocity (km/s)'}")
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / f"lstm_vs_truth_{labels[i].lower()}.png")
    plt.close()

print(" Updated plots saved in:", results_dir)


