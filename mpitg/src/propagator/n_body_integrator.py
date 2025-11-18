# File: solar_nbody_pure_python.py

import numpy as np
import spiceypy as spice
import os
from mpitg.src.almanac.constants import EPOCH_STR, FRAME, G_SI, SECONDS_PER_YEAR, YEARS, body_names, masses_kg
from mpitg.src.almanac.ephemeris import load_spice_kernels

# ----------------------------
# SPICE
# ----------------------------

load_spice_kernels()

et0 = spice.str2et(EPOCH_STR)

positions, velocities, masses = [], [], []
for name in body_names:
    print(name)
    state, _ = spice.spkezr(name, et0, FRAME, "NONE", "SSB")
    positions.append(np.array(state[:3]) * 1e3)
    velocities.append(np.array(state[3:6]) * 1e3)
    masses.append(masses_kg[name])

masses = np.array(masses)
N = len(masses)
spice.kclear()

# ----------------------------
# Physics
# ----------------------------

def accelerations(r, masses, G=G_SI):
    """Compute gravitational accelerations for N bodies."""
    N = len(masses)
    a = np.zeros_like(r)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dr = r[j] - r[i]
            dist2 = np.dot(dr, dr)
            if dist2 == 0:
                continue
            dist = np.sqrt(dist2)
            a[i] += G * masses[j] * dr / (dist2 * dist)
    return a

def nbody_derivatives(t, y, masses):
    """Return dy/dt = [v, a]"""
    r = y[:3*N].reshape((N, 3))
    v = y[3*N:].reshape((N, 3))
    a = accelerations(r, masses)
    return np.hstack([v.flatten(), a.flatten()])

def total_energy(y, masses, G=G_SI):
    r = y[:3*N].reshape((N, 3))
    v = y[3*N:].reshape((N, 3))
    ke = 0.5 * np.sum(masses * np.sum(v**2, axis=1))
    pe = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dr = r[i] - r[j]
            dist = np.linalg.norm(dr)
            if dist > 0:
                pe -= G * masses[i] * masses[j] / dist
    return ke + pe

# ----------------------------
# Pure Python Adaptive RK45 (Dormand–Prince)
# ----------------------------

# Dormand–Prince 5(4) coefficients (7 stages)
# From: https://en.wikipedia.org/wiki/Dormand–Prince_method
A = [
    [],
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
]

B = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]  # 5th order
B_hat = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]  # 4th order

C = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]

def rk45_step(func, t, y, h, masses):
    """Perform one Dormand–Prince RK45 step."""
    k = []
    for i in range(7):
        if i == 0:
            dy = func(t, y, masses)
        else:
            yi = y + h * sum(A[i][j] * k[j] for j in range(i))
            dy = func(t + C[i] * h, yi, masses)
        k.append(dy)
    
    y5 = y + h * sum(B[i] * k[i] for i in range(7))
    y4 = y + h * sum(B_hat[i] * k[i] for i in range(7))
    error = np.linalg.norm(y5 - y4, ord=2)
    return y5, error

def integrate_rk45(func, y0, t0, tf, masses, rtol=1e-9, atol=1e-12, h0=86400.0):
    """
    Adaptive RK45 integrator in pure Python/NumPy.
    Returns t_list, y_list (as lists of arrays)
    """
    t = t0
    y = y0.copy()
    h = h0
    t_vals = [t]
    y_vals = [y.copy()]

    while t < tf:
        if t + h > tf:
            h = tf - t

        y_new, error = rk45_step(func, t, y, h, masses)

        # Scale error
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        eps = np.sqrt(np.mean((error / scale) ** 2))

        if eps <= 1.0:
            # Accept step
            t += h
            y = y_new
            t_vals.append(t)
            y_vals.append(y.copy())

        # Update step size
        if eps == 0:
            factor = 2.0
        else:
            factor = min(2.0, max(0.2, 0.9 * eps ** (-1/5)))
        h *= factor

        # Safety: avoid too small/large steps
        h = max(h, 1e-6)
        h = min(h, 365.25*24*3600)  # max 1 year

    return np.array(t_vals), np.array(y_vals)

# ----------------------------
# Run Integration
# ----------------------------

y0 = np.hstack([np.array(positions).flatten(), np.array(velocities).flatten()])
t0 = 0.0
tf = YEARS * SECONDS_PER_YEAR

print("Integrating with pure Python adaptive RK45...")
t_vals, y_vals = integrate_rk45(
    nbody_derivatives,
    y0,
    t0,
    tf,
    masses,
    rtol=1e-10,
    atol=1e-12,
    h0=86400.0  # 1 day initial step
)

print(f"Completed {len(t_vals)} adaptive steps.")

# Reshape
r_sol = y_vals[:, :3*N].reshape((-1, N, 3))
v_sol = y_vals[:, 3*N:].reshape((-1, N, 3))

# ----------------------------
# Energy & SPICE Comparison
# ----------------------------

energies = np.array([total_energy(y, masses) for y in y_vals])
E0 = energies[0]
rel_energy_error = np.abs((energies - E0) / E0)
print(f"Max relative energy error: {rel_energy_error.max():.2e}")

# SPICE comparison for Earth
load_spice_kernels()

earth_spice_pos = []
for t in t_vals:
    et = et0 + t
    pos_km, _ = spice.spkpos("EARTH", et, FRAME, "NONE", "SSB")
    earth_spice_pos.append(np.array(pos_km) * 1e3)
earth_spice_pos = np.array(earth_spice_pos)
spice.kclear()

earth_integrated = r_sol[:, 3, :]
pos_error_km = np.linalg.norm(earth_spice_pos - earth_integrated, axis=1) / 1e3
print(f"Max Earth position error vs SPICE: {pos_error_km.max():.2f} km")

# ----------------------------
# Save Results
# ----------------------------

np.savez_compressed(
    "nbody_pure_python.npz",
    time_seconds=t_vals,
    positions_m=r_sol,
    velocities_mps=v_sol,
    body_names=body_names,
    masses_kg=masses,
    energy_joules=energies,
    earth_spice_error_km=pos_error_km
)
print("Saved to 'nbody_pure_python.npz'")

# ----------------------------
# Plot (if main)
# ----------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(r_sol[:,3,0]/1e9, r_sol[:,3,1]/1e9, 'b', label='RK45')
    plt.plot(earth_spice_pos[:,0]/1e9, earth_spice_pos[:,1]/1e9, 'r--', alpha=0.7, label='SPICE')
    plt.axis('equal'); plt.grid(); plt.legend()
    plt.xlabel("X (Gm)"); plt.ylabel("Y (Gm)")
    plt.title("Earth Orbit")

    plt.subplot(1, 2, 2)
    plt.semilogy(t_vals/SECONDS_PER_YEAR, rel_energy_error, label="Energy Error")
    plt.twinx().plot(t_vals/SECONDS_PER_YEAR, pos_error_km, 'g', label="Pos Error (km)")
    plt.xlabel("Time (years)")
    plt.grid()
    plt.title("Diagnostics")

    plt.tight_layout()
    plt.show()