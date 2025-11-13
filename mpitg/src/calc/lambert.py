import numpy as np
from mpitg.src.almanac.constants import MU_SUN

def stumpff_C(z):
    if z > 0:
        sz = np.sqrt(z)
        return (1 - np.cos(sz)) / z
    elif z < 0:
        sz = np.sqrt(-z)
        return (np.cosh(sz) - 1) / (-z)
    else:
        return 0.5

def stumpff_S(z):
    if z > 0:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (z * sz)
    elif z < 0:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / (-z * sz)
    else:
        return 1.0 / 6.0

def lambert_universal(r1, r2, tof, mu=MU_SUN, prograde=True, max_iter=100, tol=1e-8):
    """
    Solve Lambert's problem using universal variables (Battin's method).
    
    Parameters:
    -----------
    r1, r2 : array-like, shape (3,)
        Initial and final position vectors [m]
    tof : float
        Time of flight [s] (> 0)
    mu : float
        Gravitational parameter [m^3/s^2]
    prograde : bool
        If True, choose short-way transfer; else long-way
    max_iter : int
        Max Newton iterations
    tol : float
        Convergence tolerance on time residual
        
    Returns:
    --------
    v1, v2 : ndarray, shape (3,)
        Initial and final velocity vectors [m/s]
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    
    if tof <= 0:
        raise ValueError("Time of flight must be positive")
    
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    
    if r1_norm == 0 or r2_norm == 0:
        raise ValueError("Position vectors must be non-zero")
    
    # Chord and transfer angle
    c_vec = r2 - r1
    c = np.linalg.norm(c_vec)
    
    # Compute transfer angle (0 to 2π)
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    dnu = np.arccos(cos_dnu)
    
    # Determine direction using cross product
    h_trial = np.cross(r1, r2)
    if np.linalg.norm(h_trial) == 0:
        # Collinear case: assume prograde in xy-plane
        h_trial = np.array([0, 0, 1])
    if np.dot(h_trial, np.array([0, 0, 1])) < 0:
        dnu = 2 * np.pi - dnu
    
    # Short-way vs long-way
    if not prograde:
        dnu = 2 * np.pi - dnu
    
    # Minimum semi-perimeter
    s = (r1_norm + r2_norm + c) / 2.0
    if s == 0:
        raise ValueError("Invalid geometry: degenerate triangle")
    
    # Ensure sqrt argument non-negative
    tmp = s * (s - c)
    if tmp < 0:
        tmp = 0
    am = s / 2.0  # semi-major axis of minimum-energy ellipse
    
    # Initial guess for x (universal anomaly related)
    # Use Battin's initial guess
    A = np.sqrt(r1_norm * r2_norm) * np.sin(dnu)
    if A == 0:
        A = 1e-12  # avoid division by zero in degenerate case
    
    # Initial guess for z = alpha * x^2 (alpha = 1/a)
    z = 0.0
    if tof < np.sqrt(2 * s**3 / mu):  # short time → elliptic
        z = 0.1
    else:
        z = -0.1
    
    # Newton iteration on z
    for _ in range(max_iter):
        z = x * x / am if am != 0 else 0.0  # am = s/2, but handle am=0

        C = stumpff_C(z)
        S = stumpff_S(z)
        
        if A == 0:
            raise RuntimeError("A became zero; geometry invalid")
        
        y = r1_norm + r2_norm - A * (1 - z * S) / np.sqrt(C) if C > 0 else r1_norm + r2_norm - A * (1 - z * S) / np.sqrt(-C + 1e-16)
        if y < 0:
            y = 0.0
        
        x = np.sqrt(y / C) if C > 0 else np.sqrt(-y / C + 1e-16)
        
        # Time of flight
        tof_calc = (x**3) * S + A * np.sqrt(y) / np.sqrt(mu)
        
        # Derivative dt/dz
        if z == 0:
            dtdz = np.sqrt(2) / 40.0 * (s**3.5) / np.sqrt(mu)
        else:
            dtdz = (x**3 / (4 * np.sqrt(mu))) * (
                (r1_norm + r2_norm) * (C - 3*S) / np.sqrt(C) + 3 * S * c
            )
        
        # Newton correction
        dt = tof - tof_calc
        if abs(dt) < tol:
            break
        
        dz = dt / (dtdz + 1e-16)
        z += dz
        
        # Safety: prevent divergence
        if abs(z) > 1e6:
            raise RuntimeError("Lambert solver diverged")
    else:
        raise RuntimeError(f"Lambert solver failed to converge after {max_iter} iterations (residual: {abs(tof - tof_calc):.3e})")
    
    # Final velocities
    C = stumpff_C(z)
    S = stumpff_S(z)
    y = r1_norm + r2_norm - A * (1 - z * S) / np.sqrt(C) if C > 0 else r1_norm + r2_norm - A * (1 - z * S) / np.sqrt(-C + 1e-16)
    if y < 0:
        y = 0.0
    
    f = 1 - y / r1_norm
    g = tof - np.sqrt(y**3 / mu) * S
    gdot = 1 - y / r2_norm
    
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    
    return v1, v2


def lambert_gooding(r1, r2, tof, mu=MU_SUN, prograde=True, max_iter=100, tol=1e-8):
    """
    Solve Lambert's problem using Gooding's universal variable method.
    Robust for all elliptic, parabolic, and hyperbolic cases.
    
    Parameters:
    -----------
    r1, r2 : array-like, shape (3,)
        Initial and final position vectors (must be in consistent units, e.g., meters)
    tof : float
        Time of flight (> 0, in seconds if mu in m^3/s^2)
    mu : float
        Gravitational parameter
    prograde : bool
        Short-way (True) or long-way (False) transfer
    max_iter : int
        Maximum iterations
    tol : float
        Tolerance on time residual

    Returns:
    --------
    v1, v2 : ndarray, shape (3,)
        Initial and final velocity vectors
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    
    if tof <= 0:
        raise ValueError("Time of flight must be positive")

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    
    if r1_norm == 0 or r2_norm == 0:
        raise ValueError("Position vectors must be non-zero")

    # Chord and transfer angle
    c_vec = r2 - r1
    c = np.linalg.norm(c_vec)
    
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    dnu = np.arccos(cos_dnu)
    
    # Resolve quadrant
    h_trial = np.cross(r1, r2)
    if np.linalg.norm(h_trial) == 0:
        # Collinear: assume motion in +z direction
        h_z = 1.0
    else:
        h_z = h_trial[2]
    
    if h_z < 0:
        dnu = 2 * np.pi - dnu
    if not prograde:
        dnu = 2 * np.pi - dnu

    # Geometry parameters
    s = (r1_norm + r2_norm + c) / 2.0
    if s == 0:
        raise ValueError("Degenerate orbit")

    # Minimum energy ellipse semi-major axis
    am = s / 2.0
    # A parameter (Gooding)
    A = np.sqrt(r1_norm * r2_norm) * np.sin(dnu)
    if A == 0:
        raise ValueError("Geometry singular (A=0)")

    # Independent variable: x (related to universal anomaly)
    # Initial guess
    if tof < np.sqrt(2 * s**3 / mu):
        x = np.sqrt(mu) * (tof / s**1.5)  # short time → near parabolic
    else:
        x = np.sqrt(mu) / np.sqrt(am) * tof / s  # long time → near circular

    # Newton iteration on x
    for _ in range(max_iter):
        # Compute Stumpff functions via series for stability
        z = x**2 / am if am != 0 else 0.0
        if abs(z) < 1e-6:
            C = 0.5 - z/24.0 + z**2/720.0
            S = 1.0/6.0 - z/120.0 + z**2/5040.0
        else:
            if z > 0:
                sqrt_z = np.sqrt(z)
                C = (1 - np.cos(sqrt_z)) / z
                S = (sqrt_z - np.sin(sqrt_z)) / (z * sqrt_z)
            else:
                sqrt_nz = np.sqrt(-z)
                C = (np.cosh(sqrt_nz) - 1) / (-z)
                S = (np.sinh(sqrt_nz) - sqrt_nz) / (-z * sqrt_nz)

        y = r1_norm + r2_norm - A * (1 - z * S) / np.sqrt(C) if C > 1e-12 else r1_norm + r2_norm - A * (1 - z * S) / np.sqrt(1e-12)
        if y < 0:
            y = 0.0

        # Time of flight
        tof_new = (x**3) * S + A * np.sqrt(y) / np.sqrt(mu)
        dt = tof_new - tof

        if abs(dt) < tol:
            break

        # Derivative dt/dx (Gooding's expression)
        dtdx = (x**2) * np.sqrt(y / mu) * (1 - A / (4 * y) * (z * S - 1 + C)) + A * x / (2 * np.sqrt(mu * y)) * (1 - z * S)

        if dtdx == 0:
            raise RuntimeError("Zero derivative")

        dx = -dt / dtdx
        x += dx

        if abs(dx) > 1e6:
            raise RuntimeError("Divergence detected")

    else:
        raise RuntimeError(f"Lambert failed to converge (residual: {abs(dt):.3e})")

    # Final velocities
    f = 1 - y / r1_norm
    g = tof - np.sqrt(y**3 / mu) * S
    g_dot = 1 - y / r2_norm

    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g

    return v1, v2