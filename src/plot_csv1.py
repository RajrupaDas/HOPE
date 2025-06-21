import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your CSV
df = pd.read_csv("../data/leo_orbit_basic.csv")

# 3D Trajectory Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(df["x (km)"], df["y (km)"], df["z (km)"], lw=0.5)
ax.set_title("3D Trajectory of LEO Satellite")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
plt.show()

# 2D Position vs Time
plt.figure(figsize=(10, 6))
plt.plot(df["time (s)"], df["x (km)"], label="x")
plt.plot(df["time (s)"], df["y (km)"], label="y")
plt.plot(df["time (s)"], df["z (km)"], label="z")
plt.legend()
plt.title("Position Components vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Position (km)")
plt.show()

