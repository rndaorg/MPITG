from matplotlib import pyplot as plt
import numpy as np
import spiceypy as spice

from mpitg.src.almanac.constants import DAY, MU_SUN, MU_SUN_KM
from mpitg.src.almanac.ephemeris import get_keplerian_elements, load_spice_kernels
from mpitg.src.solver.kepler import kepler_propagate
from mpitg.src.solver.lambert import universal_lambert

load_spice_kernels()

# Reference epoch (midpoint of your window)
ref_date = '2024-06-01'
t0_et = spice.str2et(ref_date)

# Get elements once
earth_elts = get_keplerian_elements(3, t0_et)
mars_elts  = get_keplerian_elements(4, t0_et)

# Time grid
t_launch0 = spice.str2et('2024-01-01')
t_launch1 = spice.str2et('2024-12-31')
t_arrival1 = spice.str2et('2026-01-01')

launch_ets = np.arange(t_launch0, t_launch1, 4 * DAY)
arrival_ets = np.arange(t_launch0 + 200*DAY, t_arrival1, 4 * DAY)

C3_grid = np.full((len(arrival_ets), len(launch_ets)), np.nan)


'''
for i, t_arr in enumerate(arrival_ets):
    r2 = kepler_propagate(*mars_elts, t=t_arr)
    if np.any(np.isnan(r2)): continue
    for j, t_dep in enumerate(launch_ets):
        if t_arr <= t_dep: 
            continue
        tof = t_arr - t_dep
        r1 = kepler_propagate(*earth_elts, t=t_dep)
        if np.any(np.isnan(r1)): 
            continue
        
        v1, v2 = universal_lambert(r1, r2, tof)
        if np.any(np.isnan(v1)): 
            continue

        # Planet velocities via analytical derivative (or approximate)
        # Simple: use circular orbit approximation for v_planet
        v_earth = np.sqrt(MU_SUN_KM / np.linalg.norm(r1)) * np.array([-r1[1], r1[0], 0]) / np.linalg.norm(r1[:2])
        v_mars  = np.sqrt(MU_SUN_KM / np.linalg.norm(r2)) * np.array([-r2[1], r2[0], 0]) / np.linalg.norm(r2[:2])

        v_inf = v1 - v_earth
        C3_grid[i, j] = np.dot(v_inf, v_inf)

        print(C3_grid[i,j])

'''

for i, t_arr in enumerate(arrival_ets):
    # Get Mars position from your Kepler propagator (or SPICE)
    r2 = kepler_propagate(*mars_elts, t=t_arr)
    if np.any(np.isnan(r2)): continue

    # Get TRUE Mars velocity from SPICE
    state_mars, _ = spice.spkezr('4', t_arr, 'ECLIPJ2000', 'NONE', '10')
    
    v_mars_true = np.array(state_mars[3:6])

    for j, t_dep in enumerate(launch_ets):
        if t_arr <= t_dep: 
            continue
        tof = t_arr - t_dep
        r1 = kepler_propagate(*earth_elts, t=t_dep)
        if np.any(np.isnan(r1)): 
            continue

        v1, v2 = universal_lambert(r1, r2, tof)
        if np.any(np.isnan(v1)): 
            continue

        # Get TRUE Earth velocity from SPICE
        state_earth,_ = spice.spkezr('3', t_dep, 'ECLIPJ2000', 'NONE', '10')
        v_earth_true = np.array(state_earth[3:6])

        # Compute C3 correctly
        v_inf = v1 - v_earth_true
        C3_grid[i,j] = np.dot(v_inf, v_inf)

        print(C3_grid[i,j])

        '''
        # Sanity check
        if 5 < C3 < 50:  # reasonable range
            C3_grid[i, j] = C3
        else:
            C3_grid[i, j] = np.nan
        '''

# Plot
plt.figure(figsize=(10, 8))
extent = [
    (launch_ets[0] - t_launch0)/DAY,
    (launch_ets[-1] - t_launch0)/DAY,
    (arrival_ets[0] - t_launch0)/DAY,
    (arrival_ets[-1] - t_launch0)/DAY
]
plt.imshow(C3_grid, origin='lower', aspect='auto', extent=extent, cmap='viridis')
plt.colorbar(label='C3 (km²/s²)')
plt.xlabel('Days from 2024-01-01 (launch)')
plt.ylabel('Days from 2024-01-01 (arrival)')
plt.title('Earth-Mars Porkchop (Keplerian Ephemeris)')
plt.show()
