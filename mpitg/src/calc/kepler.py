import numpy as np

from mpitg.src.almanac.constants import MU_EARTH

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