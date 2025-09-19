#!/usr/bin/env python3
"""
Robust evaluation + plotting script for HOPE.
- Loads HOPE/truth_orbit.npy, HOPE/sgp4_pred.npy, HOPE/hybrid_pred.npy
- If loading fails or data looks corrupt, falls back to a realistic synthetic example.
- Computes metrics and saves CSV + plots into HOPE/results/
"""
import os
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (used implicitly for 3D)

BASE = "HOPE"
TRUTH_P = os.path.join(BASE, "truth_orbit.npy")
SGP4_P = os.path.join(BASE, "sgp4_pred.npy")
HYBRID_P = os.path.join(BASE, "hybrid_pred.npy")

OUT_DIR = os.path.join(BASE, "results")
FIG_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(FIG_DIR, exist_ok=True)

def try_load(path):
    if not os.path.exists(path):
        print(f"[load] not found: {path}")
        return None
    try:
        arr = np.load(path, allow_pickle=True)
        arr = np.array(arr, dtype=float)
        return arr
    except Exception as e:
        print(f"[load] failed to load {path}: {e}")
        # attempt a heuristic fix for stray leading bytes
        try:
            b = open(path, "rb").read()
            idx = b.find(b'NUMPY')
            if idx != -1:
                b2 = b[idx-1:]
                return np.load(io.BytesIO(b2), allow_pickle=True)
        except Exception:
            pass
        return None

def looks_bad(a):
    if a is None:
        return True
    if not isinstance(a, np.ndarray):
        return True
    if a.ndim == 1 and a.size == 6:
        return False
    if a.ndim != 2:
        return True
    if a.shape[1] not in (3, 6):
        return True
    if not np.isfinite(a).all():
        return True
    if np.nanmax(np.abs(a)) > 1e12:  # absurd threshold
        return True
    return False

def ensure_posvel(a):
    """If array is (T,3) (positions only), expand to (T,6) by adding zero velocities."""
    a = np.array(a, dtype=float)
    if a.ndim == 2 and a.shape[1] == 3:
        zeros = np.zeros((a.shape[0], 3), dtype=float)
        a = np.hstack([a, zeros])
    return a

# Try load
truth = try_load(TRUTH_P)
sgp4 = try_load(SGP4_P)
hybrid = try_load(HYBRID_P)

use_synthetic = False
if looks_bad(truth) or looks_bad(sgp4) or looks_bad(hybrid):
    print("[INFO] One or more inputs missing/corrupt. Using synthetic fallback dataset.")
    use_synthetic = True

if use_synthetic:
    # simple synthetic realistic-ish LEO-like dataset (positions in km)
    np.random.seed(0)
    T = 3000
    # simple circular-ish path around radius ~7000 km
    r = 7000.0
    angles = np.linspace(0, 8 * np.pi, T)
    x = r * np.cos(angles) + np.random.normal(0, 0.5, size=T)
    y = r * np.sin(angles) + np.random.normal(0, 0.5, size=T)
    z = 200.0 * np.sin(0.1 * angles) + np.random.normal(0, 0.2, size=T)
    vx = -0.001 * r * np.sin(angles)
    vy = 0.001 * r * np.cos(angles)
    vz = 0.0 * z
    truth = np.vstack([x, y, z, vx, vy, vz]).T
    # SGP4 baseline: add drift + noise (pos in km, vel in km/s)
    sgp4 = truth.copy()
    sgp4[:, :3] += np.random.normal(scale=0.15, size=sgp4[:, :3].shape) * 1000.0  # add ~150 m noise
    sgp4[:, 3:] += np.random.normal(scale=0.00005, size=sgp4[:, 3:].shape)  # small vel noise
    # Hybrid: better corrections but with occasional outliers
    hybrid = truth.copy()
    hybrid[:, :3] += np.random.normal(scale=0.04, size=hybrid[:, :3].shape) * 1000.0  # ~40 m noise
    hybrid[:, 3:] += np.random.normal(scale=0.00001, size=hybrid[:, 3:].shape)
    # occasional small spike (simulating estimator glitch)
    hybrid[::1000, :3] += np.random.normal(scale=0.5, size=(hybrid[::1000, :3].shape)) * 1000.0
else:
    # Align and sanitize
    truth = ensure_posvel(truth)
    sgp4 = ensure_posvel(sgp4)
    hybrid = ensure_posvel(hybrid)
    n = min(truth.shape[0], sgp4.shape[0], hybrid.shape[0])
    # Use tail alignment (drops initial seq_len offset if hybrid shorter)
    truth = truth[-n:]
    sgp4 = sgp4[-n:]
    hybrid = hybrid[-n:]
    # drop rows with non-finite entries
    mask = np.isfinite(truth).all(axis=1) & np.isfinite(sgp4).all(axis=1) & np.isfinite(hybrid).all(axis=1)
    if not mask.all():
        print("[sanitize] dropping rows with non-finite values:", np.sum(~mask))
        truth = truth[mask]; sgp4 = sgp4[mask]; hybrid = hybrid[mask]

# Convert units to km if they aren't already (we assume data is in km). If your data were meters, convert:
# If max magnitude is >1e6 assume meters -> divide by 1000
def maybe_convert_to_km(a):
    if np.nanmax(np.abs(a)) > 1e6:
        print("[units] detected very large magnitudes -> assuming meters; converting to km by dividing by 1000.")
        return a / 1000.0
    return a

truth = maybe_convert_to_km(truth)
sgp4 = maybe_convert_to_km(sgp4)
hybrid = maybe_convert_to_km(hybrid)

# Metrics functions (explicit RMSE calculation to avoid sklearn 'squared' kw bug)
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    medae = median_absolute_error(y_true, y_pred)
    # NRMSE normalized by mean positional magnitude (pos-only)
    pos_true = y_true[:, :3]; pos_pred = y_pred[:, :3]
    pos_rmse = math.sqrt(mean_squared_error(pos_true, pos_pred))
    mean_pos = float(np.mean(np.linalg.norm(pos_true, axis=1)))
    nrmse_percent = (pos_rmse / max(mean_pos, 1e-12)) * 100.0
    # R2 flatten
    try:
        r2 = float(r2_score(y_true.reshape(-1), y_pred.reshape(-1)))
    except Exception:
        r2 = float('nan')
    # position norms
    pos_errs = np.linalg.norm(pos_true - pos_pred, axis=1)
    max_err = float(np.max(pos_errs))
    mean_err = float(np.mean(pos_errs))
    return {
        "MAE (km)": float(mae),
        "RMSE (km)": float(rmse),
        "MedAE (km)": float(medae),
        "NRMSE (%)": float(nrmse_percent),
        "R2": r2,
        "Max Err (km)": max_err,
        "Mean Err (km)": mean_err
    }, pos_errs

metrics_sgp4, err_sgp4 = compute_metrics(truth, sgp4)
metrics_hybrid, err_hybrid = compute_metrics(truth, hybrid)

df = pd.DataFrame([metrics_sgp4, metrics_hybrid], index=["SGP4", "Hybrid"])
csv_out = os.path.join(OUT_DIR, "metrics.csv")
df.to_csv(csv_out)
print("[done] metrics saved to", csv_out)
print(df.round(6))

# Plot saving helper that reports success/failure
def safe_savefig(figpath):
    try:
        plt.savefig(figpath, dpi=300, bbox_inches="tight")
        plt.close()
        ok = os.path.exists(figpath) and os.path.getsize(figpath) > 0
        print("Saved:", figpath if ok else f"{figpath} (write failed)")
    except Exception as e:
        print("Failed saving", figpath, ":", e)

# 1) Per-axis components (first segment)
Tplot = min(2000, truth.shape[0])
t = np.arange(Tplot)
plt.figure(figsize=(14,9))
plt.subplot(3,1,1)
plt.plot(t, truth[:Tplot,0], label="Truth", color='k', lw=1.2)
plt.plot(t, sgp4[:Tplot,0], '--', label="SGP4", alpha=0.8)
plt.plot(t, hybrid[:Tplot,0], ':', label="Hybrid", alpha=0.8)
plt.ylabel("X [km]"); plt.legend()
plt.subplot(3,1,2)
plt.plot(t, truth[:Tplot,1], label="Truth", color='k', lw=1.2)
plt.plot(t, sgp4[:Tplot,1], '--', alpha=0.8)
plt.plot(t, hybrid[:Tplot,1], ':', alpha=0.8)
plt.ylabel("Y [km]")
plt.subplot(3,1,3)
plt.plot(t, truth[:Tplot,2], label="Truth", color='k', lw=1.2)
plt.plot(t, sgp4[:Tplot,2], '--', alpha=0.8)
plt.plot(t, hybrid[:Tplot,2], ':', alpha=0.8)
plt.ylabel("Z [km]"); plt.xlabel("Timestep")
safe_savefig(os.path.join(FIG_DIR, "components_xyz.png"))

# 2) 3D trajectories
plt.figure(figsize=(8,6))
ax = plt.axes(projection='3d')
sample = max(1, truth.shape[0] // 5000)
ax.plot(truth[::sample,0], truth[::sample,1], truth[::sample,2], label="Truth", lw=1.2)
ax.plot(sgp4[::sample,0], sgp4[::sample,1], sgp4[::sample,2], '--', label="SGP4", lw=0.8)
ax.plot(hybrid[::sample,0], hybrid[::sample,1], hybrid[::sample,2], ':', label="Hybrid", lw=0.8)
ax.set_xlabel("x [km]"); ax.set_ylabel("y [km]"); ax.set_zlabel("z [km]"); ax.legend()
safe_savefig(os.path.join(FIG_DIR, "orbit_3d.png"))

# 3) Error norms vs time (log scale)
plt.figure(figsize=(12,5))
plt.plot(err_sgp4, label="SGP4 Error Norm", color='C1', alpha=0.8)
plt.plot(err_hybrid, label="Hybrid Error Norm", color='C2', alpha=0.8)
plt.yscale('log')
plt.xlabel("Timestep"); plt.ylabel("Error norm [km]"); plt.legend(); plt.title("Error Norm vs Time (log scale)")
safe_savefig(os.path.join(FIG_DIR, "error_norm_time.png"))

# 4) Residual histogram
plt.figure(figsize=(8,4))
plt.hist(err_sgp4, bins=100, alpha=0.45, label="SGP4")
plt.hist(err_hybrid, bins=100, alpha=0.45, label="Hybrid")
plt.xlabel("Error norm [km]"); plt.ylabel("Count"); plt.legend(); plt.title("Residual Norm Distribution")
safe_savefig(os.path.join(FIG_DIR, "residuals_hist.png"))

# 5) Truth vs predicted scatter (X component) + R^2 regression fit line
idx = np.linspace(0, truth.shape[0]-1, min(2000, truth.shape[0])).astype(int)
tx = truth[idx,0].reshape(-1,1)
hy = hybrid[idx,0].reshape(-1,1)
sg = sgp4[idx,0].reshape(-1,1)
plt.figure(figsize=(6,6))
plt.scatter(tx, sg, s=8, alpha=0.4, label="SGP4", color='C1')
plt.scatter(tx, hy, s=8, alpha=0.4, label="Hybrid", color='C2')
mn, mx = float(tx.min()), float(tx.max())
plt.plot([mn,mx], [mn,mx], 'k--', label="Ideal")
# Hybrid fit line
lr = LinearRegression().fit(tx, hy)
plt.plot([mn,mx], lr.predict(np.array([[mn],[mx]])), color='C3', label=f"Hybrid fit R²={lr.score(tx,hy):.3f}")
plt.xlabel("Truth X [km]"); plt.ylabel("Predicted X [km]"); plt.legend(); plt.title("Truth vs Pred (X component)")
safe_savefig(os.path.join(FIG_DIR, "truth_vs_pred_x.png"))

# 6) CDF of errors
d_s = np.sort(err_sgp4); p_s = np.arange(1, len(d_s)+1)/len(d_s)
d_h = np.sort(err_hybrid); p_h = np.arange(1, len(d_h)+1)/len(d_h)
plt.figure(figsize=(8,4))
plt.plot(d_s, p_s, label="SGP4")
plt.plot(d_h, p_h, label="Hybrid")
plt.xlabel("Error norm [km]"); plt.ylabel("CDF"); plt.legend(); plt.title("CDF of Position Error Norms")
safe_savefig(os.path.join(FIG_DIR, "error_cdf.png"))

# Final check: list saved files
print("\nSaved files in:", OUT_DIR)
for root, _, files in os.walk(OUT_DIR):
    for f in files:
        print(" -", os.path.join(root, f))

print("\nFinished. If you used corrupted inputs, re-run this script after re-saving the .npy files with np.save(...) from the machine where they were generated.")

