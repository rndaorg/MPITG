import numpy as np
from src.math.kepler import keplerian_to_cartesian

# Example: ISS-like orbit (approximate)
a       = 6780e3          # ~400 km altitude + Earth radius
e       = 0.001
i       = np.radians(51.6)
Omega   = np.radians(100.0)
omega   = np.radians(45.0)
nu      = np.radians(30.0)

r, v = keplerian_to_cartesian(a, e, i, Omega, omega, nu)

print("Position [m]:", r)
print("Velocity [m/s]:", v)