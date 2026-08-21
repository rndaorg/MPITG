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


def lambert_solver(r1, r2, tof, mu=MU_SUN_KM, short_way=True, Nrev=0, tol=1e-8, max_iter=200):
    """
    Robust Lambert solver matching GMAT/Orekit/EMTG behavior.
    
    Parameters
    ----------
    r1 : array-like
        Departure position vector (km)
    r2 : array-like
        Arrival position vector (km)
    tof : float
        Time of flight (s)
    mu : float
        Gravitational parameter (km^3/s^2)
    short_way : bool
        True for short-way (<180 deg) transfer, False for long-way
    Nrev : int
        Number of full revolutions (N >= 0)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    v1 : np.ndarray
        Departure velocity (km/s)
    v2 : np.ndarray
        Arrival velocity (km/s)
    """
    r1 = np.array(r1, dtype=float)
    r2 = np.array(r2, dtype=float)
    R1 = np.linalg.norm(r1)
    R2 = np.linalg.norm(r2)

    cos_dtheta = np.dot(r1, r2)/(R1*R2)
    cos_dtheta = np.clip(cos_dtheta, -1.0, 1.0)
    dtheta = np.arccos(cos_dtheta)
    if not short_way:
        dtheta = 2*np.pi - dtheta

    A = np.sin(dtheta) * np.sqrt(R1*R2 / (1 - cos_dtheta))
    if A == 0:
        raise ValueError("Lambert: no feasible transfer (r1 parallel to r2)")

    # Stumpff functions
    def stumpC(z):
        if z > 1e-8:
            return (1 - np.cos(np.sqrt(z))) / z
        elif z < -1e-8:
            return (np.cosh(np.sqrt(-z)) - 1) / -z
        return 0.5

    def stumpS(z):
        if z > 1e-8:
            return (np.sqrt(z) - np.sin(np.sqrt(z))) / (z**1.5)
        elif z < -1e-8:
            return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / ((-z)**1.5)
        return 1.0/6.0

    # Function to compute y, TOF for given z
    def y_tof(z):
        C = stumpC(z)
        S = stumpS(z)
        if C == 0:
            return None, None, None
        y = R1 + R2 + A*(z*S - 1)/np.sqrt(C)
        if y < 0:
            return None, None, None
        chi = np.sqrt(y / C)
        tof_z = (chi**3*S + A*np.sqrt(y)) / np.sqrt(mu)
        return y, chi, tof_z

    # Determine bounds for z iteration
    z = 0.0
    z_up = 4*np.pi**2
    z_low = -4*np.pi**2

    # Iteration loop
    for _ in range(max_iter):
        y, chi, tof_z = y_tof(z)
        if y is None:
            # rebound: adjust z to keep y positive
            z = z + 0.1 if z >= 0 else z - 0.1
            continue
        tof_err = tof_z - tof
        if abs(tof_err) < tol:
            break
        # derivative dTOF/dz
        C = stumpC(z)
        S = stumpS(z)
        dtdz = (chi**3*(C - 3*S/(2*C)) + A/8 * (3*S*np.sqrt(y)/C - A*np.sqrt(y))) / np.sqrt(mu)
        if dtdz == 0:
            z += 0.1
        else:
            z -= tof_err/dtdz
        # constrain z to reasonable range
        z = np.clip(z, z_low, z_up)
    else:
        raise RuntimeError("Lambert solver did not converge: check TOF or geometry")

    # Final y, C, S
    C = stumpC(z)
    S = stumpS(z)
    y = R1 + R2 + A*(z*S - 1)/np.sqrt(C)

    f = 1 - y / R1
    g = A * np.sqrt(y / mu)
    gdot = 1 - y / R2

    v1 = (r2 - f*r1) / g
    v2 = (gdot*r2 - r1) / g

    return v1, v2


if __name__ == "__main__":
    print("IT PRINTS!") 