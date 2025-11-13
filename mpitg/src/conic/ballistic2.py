import numpy as np
import spiceypy as spice

from mpitg.src.almanac.constants import BODY_IDS, MU_SUN
from mpitg.src.almanac.ephemeris import load_spice_kernels
from mpitg.src.calc.lambert import lambert_gooding, lambert_universal


# ----------------------------
# Core evaluator (single point)
# ----------------------------
def _evaluate_single_leg(et_depart, target_id, tof_sec, prograde=True):
    # Earth state at departure (heliocentric)
    earth_state, _ = spice.spkgeo(targ=399, et=et_depart, ref='ECLIPJ2000', obs=10)
    r_earth = earth_state[:3]
    v_earth = earth_state[3:]

    # Target position at arrival
    r_target, _ = spice.spkezp(targ=target_id, et=et_depart + tof_sec, ref='ECLIPJ2000', abcorr='NONE', obs=10)
    r_target = np.array(r_target, dtype=np.float64)

    # Solve Lambert
    v_depart_helio = lambert_gooding(r_earth, r_target, tof_sec, mu=MU_SUN, prograde=prograde)

    # Compute ΔV
    dv_vec = v_depart_helio - v_earth
    dv = np.linalg.norm(dv_vec) / 1000.0  # km/s

    print(dv)

    return dv, v_depart_helio / 1000.0, v_earth / 1000.0

# ----------------------------
# Scanning evaluator
# ----------------------------
def evaluate_ballistic_leg_scan(
    departure_epoch,
    target_body,
    tof_days=None,
    tof_range_days=None,
    launch_window_days=30,
    prograde=True,
    n_launch_samples=61,
    n_tof_samples=31,
    return_all=False
):
    """
    Evaluate ballistic transfers by scanning launch window and/or TOF.

    Parameters:
    -----------
    departure_epoch : str
        Central launch date (e.g., "2026-05-15")
    target_body : str
        Target (e.g., "Mars")
    tof_days : float, optional
        Fixed TOF (days). If given, ignores tof_range.
    tof_range_days : tuple, optional
        (min_tof, max_tof) in days. Default: (150, 300) for Mars.
    launch_window_days : float
        ± half-window in days (total window = 2 * launch_window_days)
    n_launch_samples : int
        Number of launch dates to sample
    n_tof_samples : int
        Number of TOFs to sample (if scanning TOF)
    return_all : bool
        If True, return all valid transfers; else only best.

    Returns:
    --------
    If return_all=False: dict with best transfer
    If return_all=True: list of dicts
    """
    if target_body not in BODY_IDS:
        raise ValueError(f"Unknown target: {target_body}")

    target_id = BODY_IDS[target_body]
    print(target_id)

    # Set TOF range
    if tof_days is not None:
        tof_list = [tof_days]
    else:
        if tof_range_days is None:
            # Default ranges
            if target_body in ['Venus', 'Mercury']:
                tof_range_days = (100, 200)
            elif target_body == 'Mars-Barycenter':
                print("mars center")
                tof_range_days = (150, 300)
            else:
                tof_range_days = (300, 1000)
        tof_list = np.linspace(tof_range_days[0], tof_range_days[1], n_tof_samples)

    # Generate launch dates
    et_center = spice.str2et(departure_epoch)
    launch_ets = np.linspace(
        et_center - launch_window_days * 86400,
        et_center + launch_window_days * 86400,
        n_launch_samples
    )

    results = []

    for et_dep in launch_ets:
        for tof in tof_list:
            try:
                dv, v_dep, v_earth = _evaluate_single_leg(et_dep, target_id, tof * 86400.0, prograde=prograde)
                
                launch_date = spice.et2utc(et_dep, 'C', 3)
                results.append({
                    'launch_date': launch_date,
                    'tof_days': tof,
                    'delta_v_km_s': dv,
                    'v_depart_helio_km_s': v_dep,
                    'v_earth_helio_km_s': v_earth
                })
            except Exception as e:
                # Silently skip failed cases
                print(e)
                continue

    print(results)
    if not results:
        raise RuntimeError("No valid transfers found in scan window")

    # Sort by ΔV
    results.sort(key=lambda x: x['delta_v_km_s'])

    if return_all:
        return results
    else:
        return results[0]

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    '''
    spice.furnsh("kernels/naif0012.tls")
    spice.furnsh("kernels/de440s.bsp")
    '''

    load_spice_kernels()

    try:
        # Scan for best Earth→Mars transfer around May 2026
        best = evaluate_ballistic_leg_scan(
            departure_epoch="2026-05-15",
            target_body="Mars-Barycenter",
            launch_window_days=30,
            n_launch_samples=61
        )
        print(f"Best transfer found:")
        print(f"  Launch: {best['launch_date']}")
        print(f"  TOF: {best['tof_days']:.1f} days")
        print(f"  ΔV: {best['delta_v_km_s']:.3f} km/s")

        # Get top 5 alternatives
        all_transfers = evaluate_ballistic_leg_scan(
            departure_epoch="2026-05-15",
            target_body="Mars-Barycenter",
            launch_window_days=30,
            n_launch_samples=61,
            return_all=True
        )
        print(f"\nTop 5 transfers:")
        for i, t in enumerate(all_transfers[:5], 1):
            print(f"  {i}. {t['launch_date']} | TOF: {t['tof_days']:.0f}d | ΔV: {t['delta_v_km_s']:.3f} km/s")

    finally:
        spice.kclear()