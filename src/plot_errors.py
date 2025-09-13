# src/plot_errors.py
import numpy as np
import matplotlib.pyplot as plt

y_true = np.load("data/truth_orbit.npy")
y_lstm = np.load("data/lstm_pred.npy")
y_hybrid = np.load("data/hybrid_pred.npy")
y_rk4 = np.load("data/rk4_baseline.npy")
y_sgp4 = np.load("data/sgp4_pred.npy")

def error(y_true, y_pred):
    return np.linalg.norm(y_true[:len(y_pred)] - y_pred, axis=1)

plt.figure(figsize=(10,6))
plt.plot(error(y_true[1:], y_lstm), label="LSTM")
plt.plot(error(y_true[1:], y_hybrid), label="Hybrid LSTM")
plt.plot(error(y_true, y_rk4), label="RK4")
plt.plot(error(y_true[:len(y_sgp4)], y_sgp4), label="SGP4")
plt.yscale('log')
plt.xlabel("Time step (min)")
plt.ylabel("Error (km)")
plt.title("Orbit Prediction Error Growth")
plt.legend()
plt.grid(True)
plt.savefig("data/error_growth.png", dpi=300)
plt.show()

