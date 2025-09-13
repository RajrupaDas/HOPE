# src/evaluate_models.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def smape(y_true, y_pred):
    return 100*np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-8))

def max_error(y_true, y_pred):
    return np.max(np.linalg.norm(y_true - y_pred, axis=1))

y_true = np.load("data/truth_orbit.npy")
y_lstm = np.load("data/lstm_pred.npy")
y_hybrid = np.load("data/hybrid_pred.npy")
y_rk4 = np.load("data/rk4_baseline.npy")
y_sgp4 = np.load("data/sgp4_pred.npy")

def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true-y_pred)/(y_true+1e-8)))*100,
        "Max Error (km)": max_error(y_true, y_pred)
    }

results = {
    "LSTM": evaluate(y_true[1:], y_lstm),
    "Hybrid LSTM": evaluate(y_true[1:], y_hybrid),
    "RK4": evaluate(y_true, y_rk4),
    "SGP4": evaluate(y_true[:len(y_sgp4)], y_sgp4)
}

df = pd.DataFrame(results).T
print(df.round(4))
df.to_csv("data/eval_results.csv")
print("✅ Saved evaluation table → data/eval_results.csv")

