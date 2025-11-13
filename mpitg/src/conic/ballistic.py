import numpy as np
import spiceypy as spice

from mpitg.src.almanac.constants import BODY_IDS, MU_SUN
from mpitg.src.almanac.ephemeris import load_spice_kernels
from mpitg.src.calc.izzo import izzo_initial_guess
from mpitg.src.calc.lambert import solve_lambert
# ----------------------------
# Ballistic Leg Evaluator
# ----------------------------
def evaluate_ballistic_leg(departure_epoch, target_body, tof_days, prograde=True):
    """
    Evaluate heliocentric departure ΔV for a ballistic interplanetary leg.

    Parameters:
    -----------
    departure_epoch : str or datetime
        Departure epoch, e.g., '2026-05-15'
    target_body : str
        Target body name (e.g., 'Mars', 'Venus')
    tof_days : float
        Time of flight in days
    prograde : bool
        True for short-way transfer

    Returns:
    --------
    delta_v : float
        Departure ΔV relative to Earth (km/s)
    v_depart_heliocentric : ndarray
        Heliocentric departure velocity (km/s)
    v_earth_heliocentric : ndarray
        Earth's heliocentric velocity at departure (km/s)
    """
    if target_body not in BODY_IDS:
        raise ValueError(f"Unsupported target: {target_body}. Choose from: {list(BODY_IDS.keys())}")

    # Convert epoch to ephemeris time (ET)
    if isinstance(departure_epoch, str):
        et_depart = spice.str2et(departure_epoch)
    else:
        et_depart = spice.str2et(departure_epoch.strftime('%Y-%m-%d'))

    tof_sec = tof_days * 86400.0
    et_arrival = et_depart + tof_sec

    # Get Earth position/velocity at departure (heliocentric)
    earth_state, _ = spice.spkgeo(targ=399, et=et_depart, ref='ECLIPJ2000', obs=10)  # 10 = Sun
    r_earth = earth_state[:3]  # m
    v_earth = earth_state[3:]  # m/s

    # Get target position at arrival
    target_id = BODY_IDS[target_body]
    target_pos, _ = spice.spkezp(targ=target_id, et=et_arrival, ref='ECLIPJ2000', abcorr='NONE', obs=10)
    r_target = np.array(target_pos, dtype=np.float64)  # m

    initial_guess = izzo_initial_guess(r_earth, r_target, tof_days)
    print(initial_guess /  86400.0)

    # Solve Lambert
    try:
        v_depart_helio = solve_lambert(r_earth, r_target, tof_sec, mu=MU_SUN, prograde=prograde)
    except Exception as e:
        raise RuntimeError(f"Lambert solve failed: {e}")

    # Compute ΔV = |v_depart_helio - v_earth|
    delta_v_vec = v_depart_helio - v_earth
    delta_v = np.linalg.norm(delta_v_vec) / 1000.0  # convert to km/s

    return delta_v, v_depart_helio / 1000.0, v_earth / 1000.0

# ----------------------------
# 4. Example usage (if run directly)
# ----------------------------
if __name__ == "__main__":
    # Load SPICE kernels (adjust paths as needed)
    #spice.furnsh("kernels/naif0012.tls")
    #spice.furnsh("kernels/de440s.bsp")

    load_spice_kernels()

    try:
        dv, v_dep, v_earth = evaluate_ballistic_leg(
            departure_epoch="2026-05-15",
            target_body="Mars-Barycocenter",
            tof_days=210,
            prograde=True
        )
        print(f"Earth→Mars transfer on 2026-05-15 (210 days):")
        print(f"  Departure ΔV = {dv:.3f} km/s")
        print(f"  Earth velocity = {np.linalg.norm(v_earth):.3f} km/s")
        print(f"  Departure velocity = {np.linalg.norm(v_dep):.3f} km/s")
    finally:
        spice.kclear()