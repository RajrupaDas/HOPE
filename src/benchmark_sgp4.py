# src/benchmark_sgp4.py
import numpy as np
from sgp4.api import Satrec, jday

line1 = "1 25544U 98067A   21020.59097222  .00000200  00000-0  10270-4 0  9008"
line2 = "2 25544  51.6447  44.4206 0001111  70.9635  25.2181 15.48915306  8943"
sat = Satrec.twoline2rv(line1, line2)

states = []
for k in range(24*60*7):  # 1 week
    jd, fr = jday(2025,1,1,0,0,k*60)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        r, v = [np.nan]*3, [np.nan]*3
    states.append(np.concatenate([r, v]))

states = np.array(states)
np.save("data/sgp4_pred.npy", states)
print("✅ Saved SGP4 predictions → data/sgp4_pred.npy")

