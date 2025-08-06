from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
this_dir = Path(__file__).resolve().parent
data_dir = this_dir.parent / "data"
results_dir = this_dir.parent / "results"
results_dir.mkdir(exist_ok=True)

# Orbit setup (same as before)
epoch = Time("2025-01-01 00:00:00", scale="utc")
a = 7000 * u.km
ecc = 0.001 * u.one
inc = 51.6 * u.deg
raan = 0 * u.deg
argp = 0 * u.deg
nu = 0 * u.deg

orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)

mu = Earth.k.to_value(u.km**3 / u.s**2)

# Define 2-body dynamics (no perturbations)
def two_body_dynamics(t, state):
    r = state[:3]
    v = state[3:]
    norm_r = np.linalg.norm(r)
    acc = -mu * r / norm_r**3
    return np.concatenate((v, acc))

# Time settings (same as LSTM)
period = orbit.period.to_value(u.s)
t_span = (0, 2 * period)
t_eval = np.linspace(*t_span, 1000)

# Initial state [x, y, z, vx, vy, vz]
initial_state = np.concatenate((orbit.r.to_value(u.km), orbit.v.to_value(u.km / u.s)))

# Solve using RK4
sol = solve_ivp(two_body_dynamics, t_span, initial_state, t_eval=t_eval, method="RK45", rtol=1e-10)

# Save RK4 positions
rk4_positions = sol.y[:3].T  # (1000, 3)
np.save(data_dir / "rk4_positions.npy", rk4_positions)
print("RK4 positions saved to data/rk4_positions.npy")

