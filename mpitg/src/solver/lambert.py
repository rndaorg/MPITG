import numpy as np

from mpitg.src.almanac.constants import MU_SUN_KM


# ===========================================================
# Lambert solver (Universal Variables)
# ===========================================================
def lambert_universal(r1, r2, tof, mu=MU_SUN_KM):
    r1 = np.array(r1)
    r2 = np.array(r2)
    R1, R2 = np.linalg.norm(r1), np.linalg.norm(r2)

    cos_dtheta = np.dot(r1, r2) / (R1 * R2)
    dtheta = np.arccos(np.clip(cos_dtheta, -1.0, 1.0))

    A = np.sin(dtheta) * np.sqrt(R1 * R2 / (1 - cos_dtheta))
    if A == 0:
        raise ValueError("No feasible Lambert path")

    psi = 0.0
    EPS = 1e-8
    MAX_ITER = 50

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

        y = R1 + R2 + A * (psi*S - 1) / np.sqrt(C)
        if y < 0:
            psi += 0.1
            continue

        chi = np.sqrt(y / C)
        tof_new = (chi**3 * S + A*np.sqrt(y)) / np.sqrt(mu)

        if abs(tof_new - tof) < EPS:
            break

        dtdpsi = (
            chi**3 * (0.5/C*(C - 3*S/C)) +
            A*(0.5*np.sqrt(y)*S/y)
        ) / np.sqrt(mu)

        psi += (tof - tof_new) / dtdpsi

    f = 1 - y / R1
    g = A * np.sqrt(y / mu)
    gdot = 1 - y / R2

    v1 = (r2 - f*r1) / g
    v2 = (gdot*r2 - r1) / g
    return v1, v2
