import numpy as np

def gravity_assist(Vin, Vp, rp, mu, prograde=True):
    """
    Compute outgoing heliocentric velocity after a planetary flyby.
    
    Parameters:
    - Vin: Incoming heliocentric velocity vector (km/s), shape (3,)
    - Vp: Planet's heliocentric velocity vector (km/s), shape (3,)
    - rp: Periapsis distance (km)
    - mu: Planet's gravitational parameter (km^3/s^2)
    - prograde: If True, turn in direction of planet's motion (default)
    
    Returns:
    - Vout: Outgoing heliocentric velocity vector (km/s), shape (3,)
    """
    # Hyperbolic excess velocity (incoming)
    v_inf_in = Vin - Vp
    v_inf = np.linalg.norm(v_inf_in)
    
    if v_inf == 0:
        raise ValueError("Hyperbolic excess speed v∞ cannot be zero.")
    
    # Turn angle δ (radians)
    delta = 2 * np.arcsin(1 / (1 + (rp * v_inf**2) / mu))
    
    # For 3D, rotation axis is normal to v_inf_in and planet's orbital angular momentum.
    # Simplify to 2D in the plane of v_inf_in (assume maneuver in that plane).
    # Construct rotation matrix to turn v_inf_in by +δ (prograde) or -δ (retrograde).
    # Find perpendicular vector in the plane (rotate 90° in 2D sense).
    # Project to 2D using arbitrary orthonormal basis in the plane of motion.
    
    # We'll assume the turn happens in the plane defined by v_inf_in and an arbitrary
    # reference direction. For general 3D, you'd need the B-plane or approach asymptote.
    # Here, we rotate around an axis perpendicular to v_inf_in in the orbital plane.
    # Simplification: pick a reference axis not parallel to v_inf_in.
    ref = np.array([1, 0, 0])
    if np.allclose(np.abs(np.dot(ref, v_inf_in)) / (np.linalg.norm(ref)*v_inf), 1.0):
        ref = np.array([0, 1, 0])
    
    # Create orthonormal basis in the plane of rotation
    u = v_inf_in / v_inf
    w = np.cross(u, ref)
    if np.linalg.norm(w) < 1e-8:
        # Fallback
        ref = np.array([0, 0, 1])
        w = np.cross(u, ref)
    w = w / np.linalg.norm(w)
    v = np.cross(w, u)  # completes right-handed system, lies in plane perp to w
    
    # Rotation in the (u, v) plane by angle delta
    sign = 1 if prograde else -1
    v_inf_out = v_inf * (np.cos(sign*delta) * u + np.sin(sign*delta) * v)
    
    # Outgoing heliocentric velocity
    Vout = Vp + v_inf_out
    return Vout

'''
if __name__ == "__main__":
    # Jupiter flyby example (approximate)
    Vin = np.array([10.0, -5.0, 0.0])   # km/s
    Vp = np.array([13.0, 0.0, 0.0])     # Jupiter orbital velocity ~13 km/s
    rp = 700000.0                       # 700,000 km (above cloud tops)
    mu_jupiter = 1.26686534e8           # km^3/s^2
    
    Vout = gravity_assist(Vin, Vp, rp, mu_jupiter)
    print("Incoming heliocentric velocity (km/s):", Vin)
    print("Outgoing heliocentric velocity (km/s):", Vout)
    print("Delta-V from assist (km/s):", np.linalg.norm(Vout - Vin))
'''