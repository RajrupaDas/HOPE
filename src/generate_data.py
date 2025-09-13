

# src/generate_data.py
import numpy as np
from scipy.integrate import solve_ivp

# Simulation params
timesteps = 24*60*7  # 1 week, 1-min steps
dt = 60.0  # seconds

# Physical constants
mu = 398600.4418  # km^3/s^2, Earth
R_e = 6378.137
J2 = 1.08263e-3
Cd, A, m = 2.2, 3.6, 970
rho0 = 3e-12
H = 88.667
Cr = 1.3
mu_sun = 1.327124e11  # km^3/s^2 (simplified)

# Initial circular orbit at 500 km
r0 = R_e + 500  # km
v0 = np.sqrt(mu/r0)  # circular velocity km/s
y0 = [r0, 0, 0, 0, v0, 0]  # x, y, z, vx, vy, vz

# Perturbation function
def derivatives(t, y):
    x, y_, z, vx, vy, vz = y
    r_vec = np.array([x, y_, z])
    v_vec = np.array([vx, vy, vz])
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # 1️⃣ Central gravity
    a_grav = -mu/r**3 * r_vec

    # 2️⃣ J2
    z2 = z**2
    factor = -1.5*J2*(R_e/r)**2 * mu/r**3
    ax_j2 = factor * x * (5*z2/r**2 - 1)
    ay_j2 = factor * y_ * (5*z2/r**2 - 1)
    az_j2 = factor * z * (5*z2/r**2 - 3)
    a_j2 = np.array([ax_j2, ay_j2, az_j2])

    # 3️⃣ Drag
    rho = rho0 * np.exp(-(r - R_e)/H)
    a_drag = -0.5 * Cd * A/m * rho * v * v_vec

    # 4️⃣ SRP (simplified radial from Earth)
    a_srp = Cr * A/m * mu_sun / r**2 * r_vec/r

    # Total acceleration
    a_total = a_grav + a_j2 + a_drag + a_srp
    return [vx, vy, vz, a_total[0], a_total[1], a_total[2]]

# Integrate orbit
t_eval = np.arange(0, timesteps*dt, dt)
sol = solve_ivp(derivatives, [0, t_eval[-1]], y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

states = sol.y.T  # shape (timesteps, 6)
np.save("data/truth_orbit.npy", states)
print("✅ Saved truth orbit → data/truth_orbit.npy")

