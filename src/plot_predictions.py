import numpy as np
import matplotlib.pyplot as plt

# Load data
y_test = np.load("../data/y_test.npy")
preds = np.load("../data/lstm_predictions.npy")

# Plot x-coordinate prediction
plt.plot(y_test[:, 0], label="True x")
plt.plot(preds[:, 0], label="Predicted x")
plt.legend()
plt.title("LSTM Prediction vs Ground Truth (X coordinate)")
plt.xlabel("Time steps")
plt.ylabel("Normalized X")
plt.grid()
plt.savefig("../plots/lstm_vs_truth_x.png")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Load Predictions and Truth
# -----------------------------
y_test = np.load("../data/y_test.npy")
preds = np.load("../data/lstm_predictions.npy")

# -----------------------------
# Plot Prediction vs Ground Truth for x/y/z
# -----------------------------
labels = ['X', 'Y', 'Z']
save_dir = Path("../results")
save_dir.mkdir(parents=True, exist_ok=True)

for i in range(3):
    plt.figure(figsize=(10, 4))
    plt.plot(y_test[:, i], label=f"True {labels[i]}")
    plt.plot(preds[:, i], label=f"Predicted {labels[i]}", linestyle='dashed')
    plt.title(f"LSTM Prediction vs Ground Truth ({labels[i]} coordinate)")
    plt.xlabel("Time steps")
    plt.ylabel("Normalized Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f"lstm_vs_truth_{labels[i].lower()}.png")
    plt.close()

print("Saved plots in /results")

