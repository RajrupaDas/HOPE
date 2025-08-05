from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Absolute path setup
# -----------------------------
this_dir = Path(__file__).resolve().parent
data_dir = this_dir.parent / "data"
results_dir = this_dir.parent / "results"

# Ensure results directory exists
results_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
y_test_path = data_dir / "y_test.npy"
preds_path = data_dir / "lstm_predictions.npy"

assert y_test_path.exists(), f"Missing file: {y_test_path}"
assert preds_path.exists(), f"Missing file: {preds_path}"

y_test = np.load(y_test_path)
preds = np.load(preds_path)

# -----------------------------
# Plotting
# -----------------------------
labels = ['X', 'Y', 'Z']

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

    save_path = results_dir / f"lstm_vs_truth_{labels[i].lower()}.png"
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path)
    plt.close()

print("✅ All plots saved to /results")

