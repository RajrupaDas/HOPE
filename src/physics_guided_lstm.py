import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import load, dump
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
import os

# ---------- Config ----------
WINDOW = 5
EPOCHS = 100
BATCH = 32
LR = 1e-3
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------- Paths ----------
this_dir = Path(__file__).resolve().parent            # src/
project_root = this_dir.parent
data_dir = project_root / "data"
results_dir = project_root / "results"
models_dir = project_root / "models"

results_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
truth_csv = data_dir / "leo_orbit_with_j2_FIXED.csv"
rk4_path = data_dir / "rk4_positions.npy"
scaler_path = data_dir / "position_scaler.save"

if not truth_csv.exists():
    raise FileNotFoundError(f"Truth CSV not found: {truth_csv}")
if not rk4_path.exists():
    raise FileNotFoundError(f"RK4 file not found: {rk4_path}")
if not scaler_path.exists():
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

truth_df = pd.read_csv(truth_csv)
pos_cols = ["x (km)", "y (km)", "z (km)"]
for c in pos_cols:
    if c not in truth_df.columns:
        raise KeyError(f"Column {c} missing in truth CSV")

truth_pos = truth_df[pos_cols].values  # (N,3) in km
rk4_pos = np.load(rk4_path)            # (M,3) in km
scaler = load(scaler_path)             # MinMaxScaler (expects same format used earlier)

# align lengths
N = min(len(truth_pos), len(rk4_pos))
truth_pos = truth_pos[:N]
rk4_pos = rk4_pos[:N]

# normalize (use same scaler)
truth_norm = scaler.transform(truth_pos)   # (N,3)
rk4_norm = scaler.transform(rk4_pos)       # (N,3)

# ---------- Build supervised dataset ----------
X_seq = []         # sequences of previous WINDOW positions (normalized)
X_rk4_next = []    # RK4 predicted position at the target time (normalized) -> auxiliary input
y_target = []      # true next position (normalized)

for i in range(WINDOW, N):
    X_seq.append(truth_norm[i-WINDOW:i])   # shape (WINDOW, 3)
    X_rk4_next.append(rk4_norm[i])         # RK4 prediction at time i (target time)
    y_target.append(truth_norm[i])         # true position at time i

X_seq = np.array(X_seq)           # (samples, WINDOW, 3)
X_rk4_next = np.array(X_rk4_next) # (samples, 3)
y_target = np.array(y_target)     # (samples, 3)

print(f"Built dataset: X_seq {X_seq.shape}  X_rk4_next {X_rk4_next.shape}  y {y_target.shape}")

# ---------- Train/test split ----------
split_idx = int(0.8 * len(X_seq))
X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
X_rk4_train, X_rk4_test = X_rk4_next[:split_idx], X_rk4_next[split_idx:]
y_train, y_test = y_target[:split_idx], y_target[split_idx:]

print(f"Train samples: {len(X_seq_train)}  Test samples: {len(X_seq_test)}")

# ---------- Build model ----------
# Sequence input
seq_in = Input(shape=(WINDOW, 3), name="seq_input")
x = LSTM(64, return_sequences=True)(seq_in)
x = LSTM(32)(x)

# RK4 auxiliary input
rk4_in = Input(shape=(3,), name="rk4_input")

# combine
concat = Concatenate()([x, rk4_in])
out = Dense(32, activation="relu")(concat)
out = Dense(3, activation="linear")(out)

model = Model(inputs=[seq_in, rk4_in], outputs=out)
model.compile(optimizer=Adam(LR), loss="mse", metrics=["mae"])
model.summary()

# ---------- Callbacks and training ----------
ckpt = models_dir / "physics_guided_lstm.keras"
checkpoint = ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True, verbose=1)

history = model.fit(
    [X_seq_train, X_rk4_train],
    y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[checkpoint],
    verbose=2
)

# load best
from tensorflow.keras.models import load_model
best = load_model(ckpt)

# ---------- Predict on test set ----------
pred_norm = best.predict([X_seq_test, X_rk4_test])  # normalized predicted positions

# Inverse transform to km
pred_km = scaler.inverse_transform(pred_norm)
truth_km = scaler.inverse_transform(y_test)
rk4_test_km = scaler.inverse_transform(X_rk4_test)  # this is RK4 at same times

# Compute RMSE per axis
rmse_pred = np.sqrt(np.mean((truth_km - pred_km)**2, axis=0))
rmse_rk4 = np.sqrt(np.mean((truth_km - rk4_test_km)**2, axis=0))

print("\nRMSE on test set (km) per axis:")
print(f"Physics-guided LSTM: x={rmse_pred[0]:.6f}, y={rmse_pred[1]:.6f}, z={rmse_pred[2]:.6f}")
print(f"RK4 baseline         : x={rmse_rk4[0]:.6f}, y={rmse_rk4[1]:.6f}, z={rmse_rk4[2]:.6f}")

# ---------- Save predictions and model ----------
np.save(data_dir / "physics_guided_lstm_pred.npy", pred_km)
best.save(models_dir / "physics_guided_lstm_best.keras")

# ---------- Plot results ----------
labels = ["X", "Y", "Z"]
timesteps = np.arange(len(truth_km))

for i, lab in enumerate(labels):
    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, truth_km[:, i], label="Truth", linewidth=2)
    plt.plot(timesteps, rk4_test_km[:, i], label="RK4 baseline", linestyle="--")
    plt.plot(timesteps, pred_km[:, i], label="Physics-guided LSTM", linestyle=":")
    plt.title(f"Prediction comparison - {lab}")
    plt.xlabel("Test timestep")
    plt.ylabel(f"{lab} (km)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / f"physics_guided_{lab.lower()}.png")
    plt.close()

print(f"[done] saved predictions and plots in {data_dir} and {results_dir}")

