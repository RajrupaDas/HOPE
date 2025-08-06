import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error

# Paths
this_dir = Path(__file__).resolve().parent
data_dir = this_dir.parent / "data"
results_dir = this_dir.parent / "results"
results_dir.mkdir(exist_ok=True)

# Load data
lstm_preds = np.load(data_dir / "lstm_predictions.npy")
y_test = np.load(data_dir / "y_test.npy")
rk4 = np.load(data_dir / "rk4_positions.npy")

# Align sizes if needed
min_len = min(len(y_test), len(lstm_preds), len(rk4))
lstm_preds = lstm_preds[:min_len]
y_test = y_test[:min_len]
rk4 = rk4[:min_len]

# RMSE per axis
rmse_lstm = np.sqrt(np.mean((y_test - lstm_preds)**2, axis=0))
rmse_rk4 = np.sqrt(np.mean((y_test - rk4)**2, axis=0))

print("\nRMSE (km) per coordinate:")
print(f"LSTM: x={rmse_lstm[0]:.4f}, y={rmse_lstm[1]:.4f}, z={rmse_lstm[2]:.4f}")
print(f"RK4 : x={rmse_rk4[0]:.4f}, y={rmse_rk4[1]:.4f}, z={rmse_rk4[2]:.4f}\n")

# Plot comparisons
labels = ['X', 'Y', 'Z']
for i in range(3):
    plt.figure()
    plt.plot(y_test[:, i], label="True", linewidth=2)
    plt.plot(lstm_preds[:, i], label="LSTM", linestyle="--")
    plt.plot(rk4[:, i], label="RK4", linestyle=":")
    plt.xlabel("Time step")
    plt.ylabel(f"{labels[i]} Position (km)")
    plt.title(f"Prediction Comparison: {labels[i]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f"compare_{labels[i].lower()}.png")

print("Comparison plots saved to results/")

