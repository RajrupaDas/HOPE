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

