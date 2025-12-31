import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import spiceypy as sp

from mpitg.src.almanac.constants import MU_SUN_KM
from mpitg.src.almanac.ephemeris import load_spice_kernels
from mpitg.src.solver.izzo import lambert_izzo2015


# ===========================================================
# Compute TOTAL C3 grid: departure × arrival
# ===========================================================
def compute_ballistic_arc(START_BODY,
                          DESTINATION,
                          date_start,
                          date_end,
                        ):

    load_spice_kernels()
    mu_sun = MU_SUN_KM

    t_dep = sp.utc2et(date_start)
    t_arr = sp.utc2et(date_end)

    state_E = sp.spkezr(START_BODY, t_dep, "ECLIPJ2000", "NONE", "SSB")[0]
    r1 = state_E[:3]
    v1_earth = state_E[3:6]

    print(state_E)

    tof = t_arr - t_dep

    state_M = sp.spkezr(DESTINATION, t_arr,
                                "ECLIPJ2000", "NONE", "SSB")[0]
    r2 = state_M[:3]
    v2_mars = state_M[3:6]

    print(state_M)

    try:
        v1t, v2t = lambert_izzo2015(mu_sun, r1, r2, tof)
    except Exception as e:
        print(e)

    return v1t, v2t

    '''
    v_inf_dep = np.linalg.norm(v1t - v1_earth)
    v_inf_arr = np.linalg.norm(v2t - v2_mars)

    TOTAL_C3[i, j] = v_inf_dep**2 + v_inf_arr**2

    return dep_times, arr_times, TOTAL_C3
    '''


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    date_start = "2024-01-01"
    date_end = "2024-10-10"
    START_BODY = "EARTH BARYCENTER"
    DESTINATION = "MARS BARYCENTER"

    v1t, v2t = compute_ballistic_arc(START_BODY, DESTINATION, date_start, date_end)

    print("Departure and Arrival Velocities: ")
    print(v1t, v2t)
    
