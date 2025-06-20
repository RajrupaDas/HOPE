from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import propagate
import numpy as np
import pandas as pd

# -----------------------------
# Initial Orbit - LEO Example
# -----------------------------
epoch = Time("2025-01-01 00:00:00", scale="utc")
a = 7000 * u.km  # ~400km above Earth's surface
ecc = 0.001 * u.one
inc = 51.6 * u.deg
raan = 0 * u.deg
argp = 0 * u.deg
nu = 0 * u.deg

orbit = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch)

# -----------------------------
# Time samples (simulate 2 orbits)
# -----------------------------
period = orbit.period.to(u.s).value
num_points = 1000
times = np.linspace(0, 2 * period, num_points) * u.s

# -----------------------------
# Propagate and record position/velocity
# -----------------------------
data = []
for t in times:
    propagated = orbit.propagate(t)
    r = propagated.r.to(u.km).value
    v = propagated.v.to(u.km / u.s).value
    data.append([t.to(u.s).value, *r, *v])

# -----------------------------
# Export to CSV
# -----------------------------
df = pd.DataFrame(data, columns=[
    "time (s)", "x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"
])
df.to_csv("leo_orbit_basic.csv", index=False)
print("Basic LEO orbit CSV saved: leo_orbit_basic.csv")

