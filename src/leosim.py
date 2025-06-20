from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.core.perturbations import J2_perturbation
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd

# -----------------------------
# Initial orbit setup
# -----------------------------
epoch = Time("2025-01-01 00:00:00", scale="utc")
a = 7000 * u.km
ecc = 0.001 * u.one
inc = 51.6 * u.deg
raan = 0 * u.deg
argp = 0 * u.deg
nu = 0 * u.deg

# Create orbit
orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)

# -----------------------------
# Constants
# -----------------------------
mu_earth = Earth.k.to(u.km**3 / u.s**2).value
J2 = Earth.J2.value
R = Earth.R.to(u.km).value

# -----------------------------
# Derivative function (includes J2)
# -----------------------------
def two_body_j2(t, state_vec):
    r = state_vec[:3]
    v = state_vec[3:]

    # Acceleration = gravity + J2
    acc = -mu_earth * r / np.linalg.norm(r)**3
    acc += J2_perturbation(t, state_vec, k=mu_earth, J2=J2, R=R)

    return np.concatenate((v, acc))

# -----------------------------
# Initial state vector (r, v)
# -----------------------------
initial_state = np.concatenate((
    orbit.r.to_value(u.km),
    orbit.v.to_value(u.km / u.s)
))

# -----------------------------
# Time span
# -----------------------------
period_sec = orbit.period.to_value(u.s)
t_eval = np.linspace(0, 2 * period_sec, 1000)  # simulate 2 orbits

# -----------------------------
# Solve the equations of motion
# -----------------------------
sol = solve_ivp(two_body_j2, (0, t_eval[-1]), initial_state, t_eval=t_eval, rtol=1e-10)

# -----------------------------
# Extract data to DataFrame
# -----------------------------
df = pd.DataFrame({
    "time (s)": sol.t,
    "x (km)": sol.y[0],
    "y (km)": sol.y[1],
    "z (km)": sol.y[2],
    "vx (km/s)": sol.y[3],
    "vy (km/s)": sol.y[4],
    "vz (km/s)": sol.y[5],
})

df.to_csv("data/leo_orbit_with_j2_FIXED.csv", index=False)
print("CSV saved: leo_orbit_with_j2_FIXED.csv")

