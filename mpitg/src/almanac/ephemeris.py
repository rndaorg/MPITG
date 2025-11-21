import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from datetime import datetime, timedelta

from mpitg.src.almanac.constants import MU_SUN

# -----------------------------
# Load SPICE kernels
# -----------------------------

def load_spice_kernels(): 
    try:
        spice.furnsh("mpitg\\src\\almanac\\kernels\\naif0012.tls")
        spice.furnsh("mpitg\\src\\almanac\\kernels\\de440s.bsp")
        spice.furnsh("mpitg\\src\\almanac\\kernels\\pck00010.tpc")
        spice.furnsh("mpitg\\src\\almanac\\kernels\\Gravity.tpc")
    except: 
        print(f"Missing SPICE kernel: {''}")

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

solar_system_constants = {}

def fetch_all_solar_system_constants():
    # Load SPICE kernels (make sure to download these kernels)
    load_spice_kernels()
    
    # List of known solar system bodies (planets and some moons)
    solar_system_bodies = [
        "MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE", 
        "PLUTO", "MOON", "MARS MOON", "IO", "EUROPA", "GANYMEDE", "CALLISTO",  # Jupiter's moons
        "TITAN", "ENCELADUS", "RHEA", "DIONE", "TETHYS",  # Saturn's moons
        "TRITON",  # Neptune's moon
        "CERES",  # Dwarf planet in the asteroid belt
        "VESTA", "PALLAS", "JUNO", "CIGAR", "EUNOMIA"  # Some large asteroids
    ]

    # Store constants for all bodies

    for body in solar_system_bodies:
        try:
            # Fetch Gravitational Parameter (GM) for the body (in km^3/s^2)
            mu = spice.bodvrd(body, "GM", 1)[1][0]
            
            # Fetch Radius for the body (in km)
            radius = spice.bodvrd(body, "RADII", 3)[1][0]  # Returns an array, we take the first value
            
            # Store in dictionary
            solar_system_constants[body] = {
                'mu': mu,             # Gravitational parameter (km^3/s^2)
                'radius_km': radius,  # Radius (km)
            }
        except Exception as e:
            # Handle bodies that might not have available data in SPICE
            print(f"Data for {body} not found. Error: {e}")
            solar_system_constants[body] = {
                'mu': None,
                'radius_km': None,
            }

    # Fetch the Astronomical Unit (AU) in km and the speed of light (CLIGHT)
    AU_km = spice.bodvrd("SUN", "AU", 1)[1][0]  # AU in kilometers
    CLIGHT = spice.bodvrd("SUN", "LIGHTSPEED", 1)[1][0]  # Speed of light in km/s

    solar_system_constants['AU_km'] = AU_km
    solar_system_constants['CLIGHT'] = CLIGHT

    # Unload SPICE kernels when done
    spice.kclear()

    #return solar_system_constants



def get_keplerian_elements(body_id, epoch_et, mu=MU_SUN):
    """
    Return (a, e, i, LAN, w, M0, epoch) using SPICE osculating elements.
    Angles in radians, distances in km, time in seconds.
    """
    state, _ = spice.spkezr(str(body_id), epoch_et, 'ECLIPJ2000', 'NONE', '10')  # Sun-centered
    pos = np.array(state[:3])   # km
    vel = np.array(state[3:])   # km/s
    
    elements = spice.oscelt(state, epoch_et, mu)
    # oscelt returns: (perifocus_dist, eccentricity, inclination, LAN, argp, mean_anomaly, epoch)
    p = elements[0]
    e = elements[1]
    i = elements[2]
    LAN = elements[3]
    w = elements[4]
    M0 = elements[5]
    t0 = elements[6]
    
    # Convert p to semi-major axis
    if e < 1.0:
        a = p / (1.0 - e**2)
    else:
        a = -p / (e**2 - 1.0)  # hyperbolic
    return a, e, i, LAN, w, M0, t0



def validate_spice():
    load_spice_kernels()

    # -----------------------------
    # Define time span (5 years)
    # -----------------------------
    start_date = datetime(2025, 1, 1)
    end_date = start_date + timedelta(days=5*365.25)
    n_steps = 2000

    et_start = spice.str2et(start_date.strftime('%Y-%m-%d'))
    et_end = spice.str2et(end_date.strftime('%Y-%m-%d'))
    ets = np.linspace(et_start, et_end, n_steps)

    # -----------------------------
    # Fetch positions (heliocentric)
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
    # Plot orbits
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
    # Clean up
    # -----------------------------
    spice.kclear()


if __name__ == "__main__":
    print("")
    #validate_spice()

    fetch_all_solar_system_constants()

    # Fetch and print constants for all bodies in the solar system
    for body, data in solar_system_constants.items():
        if data['mu'] is not None and data['radius_km'] is not None:
            print(f"{body}:")
            print(f"  Gravitational Parameter (mu): {data['mu']} km^3/s^2")
            print(f"  Radius: {data['radius_km']} km")
        else:
            print(f"{body} data not available.")