import numpy as np
import spiceypy as sp

from mpitg.src.almanac.ephemeris import load_spice_kernels


# ===========================================================
# Lambert solver (universal variables, no SciPy)
# ===========================================================
def lambert_universal(r1, r2, tof, mu):
    """
    Solves Lambert's problem using universal variables.
    Returns departure and arrival velocity vectors.
    No SciPy; pure NumPy implementation.
    """

    r1 = np.array(r1)
    r2 = np.array(r2)
    R1 = np.linalg.norm(r1)
    R2 = np.linalg.norm(r2)

    cos_dtheta = np.dot(r1, r2) / (R1 * R2)
    dtheta = np.arccos(np.clip(cos_dtheta, -1.0, 1.0))

    # Assume prograde short-way transfer
    A = np.sin(dtheta) * np.sqrt(R1 * R2 / (1 - cos_dtheta))

    if A == 0:
        raise ValueError("No feasible Lambert path.")

    # --- Iterate for psi until converged ---
    psi = 0.0
    dpsi = 1.0
    MAX_ITER = 50
    EPS = 1e-8

    def stumpC(z):
        if z > 0:
            return (1 - np.cos(np.sqrt(z))) / z
        elif z < 0:
            return (np.cosh(np.sqrt(-z)) - 1) / -z
        return 1/2

    def stumpS(z):
        if z > 0:
            return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)**3)
        elif z < 0:
            return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / ((-z)**1.5)
        return 1/6

    for _ in range(MAX_ITER):
        C = stumpC(psi)
        S = stumpS(psi)

        y = R1 + R2 + A*(psi*S - 1)/np.sqrt(C)
        if y < 0:
            psi += dpsi
            continue

        chi = np.sqrt(y / C)
        tof_new = (chi**3 * S + A*np.sqrt(y)) / np.sqrt(mu)

        if abs(tof_new - tof) < EPS:
            break

        # Newton step
        dtdpsi = (chi**3 * (0.5/C*(C - 3*S/C)) + A*(0.5/np.sqrt(y)*S))/np.sqrt(mu)
        psi += (tof - tof_new) / dtdpsi

    # Compute velocities
    f = 1 - y/R1
    g = A * np.sqrt(y/mu)
    gdot = 1 - y/R2

    v1 = (r2 - f*r1) / g
    v2 = (gdot*r2 - r1) / g
    return v1, v2


# ===========================================================
# Compute porkchop grid (C3 only)
# ===========================================================
def porkchop_c3(
    date_start="2024-04-01",
    date_end="2026-12-01",
    ndays_depart=50,
    ndays_tof=60,
):

    load_spice_kernels()
    mu_sun = 1.32712440018e11  # km^3/s^2 (GM of the Sun)

    # --- Departure dates ---
    start_et = sp.utc2et(date_start)
    end_et = sp.utc2et(date_end)
    dep_times = np.linspace(start_et, end_et, ndays_depart)

    # --- Time-of-flight array (120–300 days typical Mars window) ---
    tof_days = np.linspace(120, 300, ndays_tof)
    tof_seconds = tof_days * 86400.0

    # Allocate output C3 grid
    C3 = np.full((ndays_depart, ndays_tof), np.nan)

    # Loop grid
    for i, t_dep in enumerate(dep_times):
        # Earth state at departure
        #r1, v1_earth = sp.spkezr("EARTH", t_dep, "ECLIPJ2000", "NONE", "SUN")[0][:3], None
        state_earth, _ = sp.spkezr("EARTH", t_dep, 'ECLIPJ2000', 'NONE', 'SUN')
        r1 = state_earth[0:3]
        v1_earth = np.array(state_earth[3:6])

        for j, tof in enumerate(tof_seconds):
            t_arr = t_dep + tof
            r2, v2_mars = sp.spkezr("MARS BARYCENTER", t_arr, "ECLIPJ2000", "NONE", "SUN")[0][:3], None

            try:
                v1_trans, _ = lambert_universal(r1, r2, tof, mu_sun)
            except Exception:
                continue

            v_inf = np.linalg.norm(v1_trans - v1_earth)
            C3[i, j] = v_inf**2

    return dep_times, tof_days, C3


# Example usage (no plotting)
if __name__ == "__main__":
    dep, tof, C3 = porkchop_c3()
    print("Departure ET samples:", dep.shape)
    print("TOF samples:", tof.shape)
    print("C3 grid:", C3.shape)
    print("Minimum C3:", np.nanmin(C3))
