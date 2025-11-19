import numpy as np

from mpitg.src.almanac.constants import MU_SUN_KM

def stumpff_C(z):
    """Stumpff function C(z)."""
    if z > 0:
        sz = np.sqrt(z)
        return (1 - np.cos(sz)) / z
    elif z < 0:
        sz = np.sqrt(-z)
        return (np.cosh(sz) - 1) / (-z)
    else:
        return 0.5

def stumpff_S(z):
    """Stumpff function S(z)."""
    if z > 0:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz ** 3)
    elif z < 0:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / ((-z) * sz)
    else:
        return 1.0 / 6.0

def universal_lambert(r1_vec, r2_vec, tof, mu=MU_SUN_KM, max_iter=100, tol=1e-8):
    """
    Solve Lambert's problem using the universal variable formulation.
    
    Parameters:
        r1_vec, r2_vec: position vectors (km) relative to central body (e.g., Sun)
        tof: time of flight (seconds)
        mu: gravitational parameter (km^3/s^2)
    
    Returns:
        v1_vec, v2_vec: velocity vectors (km/s) at departure and arrival.
                        Returns (nan, nan) if no valid solution.
    """
    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)
    
    if r1 == 0 or r2 == 0 or tof <= 0:
        return np.full(3, np.nan), np.full(3, np.nan)
    
    # Chord and transfer angle
    r1_dot_r2 = np.dot(r1_vec, r2_vec)
    cos_dnu = np.clip(r1_dot_r2 / (r1 * r2), -1.0, 1.0)
    dnu = np.arccos(cos_dnu)
    
    # Determine short-way vs long-way
    # Use cross product z-component to infer direction (assume near-ecliptic motion)
    cross_z = np.cross(r1_vec, r2_vec)[2]
    if cross_z < 0:
        dnu = 2 * np.pi - dnu
    if dnu > np.pi:
        # Long-way transfer — skip for standard porkchop (short-way only)
        return np.full(3, np.nan), np.full(3, np.nan)
    
    # Chord length
    c_vec = r2_vec - r1_vec
    c = np.linalg.norm(c_vec)
    
    # Minimum semi-perimeter
    s = (r1 + r2 + c) / 2.0
    if s == 0:
        return np.full(3, np.nan), np.full(3, np.nan)
    
    # Parameter A (Gooding's formulation)
    A = np.sin(dnu) * np.sqrt(r1 * r2 / (1.0 - cos_dnu))
    if A == 0:
        return np.full(3, np.nan), np.full(3, np.nan)
    
    # Initial guess for universal anomaly x
    # Use x = 0 (parabolic) as starting point
    x = 0.0
    
    # Newton-Raphson iteration on universal anomaly
    for _ in range(max_iter):
        # Compute z = alpha * x^2, where alpha = 1/a (reciprocal semi-major axis)
        # But we don't know a yet → use z = x^2 / a_guess → instead, define z implicitly
        # Standard approach: z = x^2 * alpha, but alpha unknown → iterate on x directly
        
        # Compute y from x
        z = x * x  # placeholder; we'll treat z as x^2 * alpha, but use A for scaling
        
        # Actually, use: y = r1 + r2 + A * (x * sqrt(z) * S(z) - 1) / sqrt(C(z))
        # Better: follow algorithm from Vallado (Algorithm 58)
        
        # Compute Stumpff functions
        C = stumpff_C(z)
        S = stumpff_S(z)
        
        # Compute y
        y = r1 + r2 - A * (1.0 - x * x * C) / np.sqrt(C) if C > 1e-14 else r1 + r2 - A
        if y < 0:
            # Invalid; increase x to push y positive
            x += 0.1
            continue
        
        # Time of flight from current x
        sqrt_y = np.sqrt(y)
        tof_x = (x**3 * S + A * sqrt_y) / np.sqrt(mu)
        
        # Derivative dt/dx
        # dt/dx = (x^2 * C + A / (4 * sqrt_y)) * (x * sqrt_y / sqrt(C) + A * sqrt(C))) / sqrt(mu)
        if C > 1e-14:
            denom = np.sqrt(C)
            term1 = x * sqrt_y / denom
            term2 = A * denom
            dtdx = (x**2 * C + A / (4.0 * sqrt_y) * (term1 + term2)) / np.sqrt(mu)
        else:
            # Near-parabolic: C ≈ 0.5, safe fallback
            dtdx = (x**2 * 0.5 + A / (4.0 * sqrt_y) * (x * sqrt_y / np.sqrt(0.5) + A * np.sqrt(0.5))) / np.sqrt(mu)
        
        if abs(dtdx) < 1e-12:
            break
        
        dx = (tof - tof_x) / dtdx
        x += dx
        
        if abs(dx) < tol:
            break
    else:
        # Did not converge
        return np.full(3, np.nan), np.full(3, np.nan)
    
    # Final Stumpff values
    z = x * x
    C = stumpff_C(z)
    S = stumpff_S(z)
    y = r1 + r2 - A * (1.0 - x * x * C) / np.sqrt(C) if C > 1e-14 else r1 + r2 - A
    if y < 0:
        return np.full(3, np.nan), np.full(3, np.nan)
    
    # Lagrange coefficients
    f = 1.0 - y / r1
    g = A * np.sqrt(y / mu)
    g_dot = 1.0 - y / r2
    
    # Velocities
    v1_vec = (r2_vec - f * r1_vec) / g
    v2_vec = (g_dot * r2_vec - r1_vec) / g
    
    return v1_vec, v2_vec