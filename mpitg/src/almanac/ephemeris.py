import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from datetime import datetime, timedelta

# -----------------------------
# 1. Load SPICE kernels
# -----------------------------
 
spice.furnsh("mpitg\\src\\almanac\\kernels\\naif0012.tls")
spice.furnsh("mpitg\\src\\almanac\\kernels\\de440s.bsp")

# -----------------------------
# 2. Define time span (5 years)
# -----------------------------
start_date = datetime(2025, 1, 1)
end_date = start_date + timedelta(days=5*365.25)
n_steps = 2000

et_start = spice.str2et(start_date.strftime('%Y-%m-%d'))
et_end = spice.str2et(end_date.strftime('%Y-%m-%d'))
ets = np.linspace(et_start, et_end, n_steps)

# SPICE body IDs:
# Sun = 10, Mercury = 199, Venus = 299, Earth = 399, Mars = 499
bodies = {
    'Mercury': 199,
    'Venus': 299,
    'Earth': 399,
    'Mars': 4
}
colors = {
    'Mercury': 'gray',
    'Venus': 'gold',
    'Earth': 'mediumblue',
    'Mars': 'red'
}

# -----------------------------
# 3. Fetch positions (heliocentric)
# -----------------------------
positions = {name: [] for name in bodies}

for et in ets:
    for name, body_id in bodies.items():
        print(body_id)
        # Get position of planet relative to Sun (reference frame: ECLIPJ2000)
        pos, _ = spice.spkezp(
            targ=body_id,
            et=et,
            ref='ECLIPJ2000',
            abcorr='NONE',
            obs=10  # Sun
        )
        positions[name].append(pos)

# Convert to AU for plotting (1 AU = 149,597,870,700 m)
AU = 149_597_870_700.0
for name in positions:
    positions[name] = np.array(positions[name]) / AU

# -----------------------------
# 4. Plot orbits
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('X (AU)', fontsize=12)
ax.set_ylabel('Y (AU)', fontsize=12)
ax.set_title('Heliocentric Orbits of Inner Planets (2025–2030)', fontsize=14)

# Plot each planet's orbit
for name in bodies:
    x = positions[name][:, 0]
    y = positions[name][:, 1]
    ax.plot(x, y, color=colors[name], label=name, linewidth=1.2)
    # Mark final position
    ax.plot(x[-1], y[-1], 'o', color=colors[name], markersize=6)

# Sun at origin
ax.plot(0, 0, 'yo', markersize=10, label='Sun')

ax.legend()
plt.tight_layout()
plt.savefig('inner_planets_orbits.png', dpi=150)
plt.show()

# -----------------------------
# 5. Clean up
# -----------------------------
spice.kclear()