import numpy as np

from mpitg.src.almanac.constants import MU_EARTH, MU_SUN_KM

def kepler_propagate(a, e, i, Omega, omega, m0, t0, t, mu=MU_SUN_KM):
    """
    Propagate to time t (seconds) and return position (km) in ECLIPJ2000.
    Only supports elliptic orbits (e < 1).
    """
    if e >= 1.0:
        return np.array([np.nan, np.nan, np.nan])
    
    n = np.sqrt(mu / np.abs(a)**3)  # mean motion (rad/s)
    M = m0 + n * (t - t0)           # mean anomaly
    M = np.mod(M + np.pi, 2*np.pi) - np.pi  # wrap to [-pi, pi]

    # Solve Kepler's equation: M = E - e*sin(E)
    E = M  # initial guess
    for _ in range(50):
        dE = (M - (E - e * np.sin(E))) / (1 - e * np.cos(E))
        E += dE
        if np.abs(dE) < 1e-10:
            break
    else:
        return np.array([np.nan, np.nan, np.nan])  # failed to converge

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    # Radius
    r = a * (1 - e * np.cos(E))

    # Position in perifocal frame
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)
    z_p = 0.0

    # Rotation matrix: perifocal -> ECLIPJ2000
    cos_O = np.cos(Omega); sin_O = np.sin(Omega)
    cos_w = np.cos(omega);   sin_w = np.sin(omega)
    cos_i = np.cos(i);   sin_i = np.sin(i)

    # Combine rotations: 3-1-3 Euler angles (Omega, i, w)
    P = np.array([
        [cos_O*cos_w - sin_O*sin_w*cos_i,  -cos_O*sin_w - sin_O*cos_w*cos_i,  sin_O*sin_i],
        [sin_O*cos_w + cos_O*sin_w*cos_i,  -sin_O*sin_w + cos_O*cos_w*cos_i, -cos_O*sin_i],
        [sin_w*sin_i,                       cos_w*sin_i,                       np.cos(i)]
    ])

    pos = P @ np.array([x_p, y_p, z_p])
    return pos


def keplerian_to_cartesian(a, e, i, Omega, omega, nu, mu=MU_EARTH):
    """
    Convert Keplerian orbital elements to inertial position and velocity vectors.

    Parameters:
    -----------
    a : float
        Semi-major axis [m]
    e : float 
        Eccentricity (dimensionless)
    i : float
        Inclination [rad]
    Omega : float
        Right ascension of ascending node [rad]
    omega : float
        Argument of periapsis [rad]
    nu : float
        True anomaly [rad]
    mu : float, optional
        Gravitational parameter of central body [m^3/s^2] (default: Earth)

    Returns:
    --------
    r : ndarray, shape (3,)
        Position vector in inertial frame [m]
    v : ndarray, shape (3,)
        Velocity vector in inertial frame [m/s]
    """
    # Ensure valid eccentricity for elliptical orbit
    if not (0 <= e < 1):
        raise ValueError("Eccentricity must satisfy 0 <= e < 1 for elliptical orbits.")

    # Distance from central body
    r_mag = a * (1 - e**2) / (1 + e * np.cos(nu))

    # Position in perifocal frame (PQW)
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])

    # Velocity magnitude factor
    h = np.sqrt(mu * a * (1 - e**2))  # specific angular momentum
    v_pqw = (mu / h) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotation matrix from perifocal to inertial (ECI)
    cos_Omega, sin_Omega = np.cos(Omega), np.sin(Omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)

    # 3-1-3 Euler rotation: R = R3(Omega) @ R1(i) @ R3(omega)
    R = np.array([
        [cos_Omega*cos_omega - sin_Omega*sin_omega*cos_i,
         -cos_Omega*sin_omega - sin_Omega*cos_omega*cos_i,
         sin_Omega*sin_i],
        [sin_Omega*cos_omega + cos_Omega*sin_omega*cos_i,
         -sin_Omega*sin_omega + cos_Omega*cos_omega*cos_i,
         -cos_Omega*sin_i],
        [sin_omega*sin_i,
         cos_omega*sin_i,
         cos_i]
    ])

    # Transform to inertial frame
    r = R @ r_pqw
    v = R @ v_pqw

    return r, v