import numpy as np
import matplotlib.pyplot as plt
import spiceypy as sp
from datetime import datetime, timedelta

from mpitg.src.almanac.ephemeris import load_spice_kernels

# Constants
AU = 1.496e11  # 1 AU in meters
MU_SUN = 1.327e20  # Gravitational parameter for the Sun in m^3/s^2

# Load SPICE kernels (assuming you have them downloaded)
# For example: sp.furnsh("naif0012.tls")
#sp.furnsh("naif0012.tls")  # You can change this path to your kernel location


# Function to convert SPICE state vector to AU (for visualization)
def convert_to_au(state_vector):
    return state_vector[:3] / AU  # Convert from meters to AU

# Function to get the position of a body at a specific date
def get_position_at_date(body, date):
    et = sp.str2et(date.strftime("%Y-%m-%dT%H:%M:%S"))  # Convert date to ephemeris time (ET)
    state, _ = sp.spkezr(body, et, 'ECLIPJ2000', 'NONE', 'SUN')  # Get position relative to the Sun
    return convert_to_au(state)

# Function to generate orbit for plotting
def generate_orbit(body, start_date, end_date, step_days=1):
    dates = []
    positions = []
    current_date = start_date
    
    while current_date <= end_date:
        pos = get_position_at_date(body, current_date)
        positions.append(pos)
        dates.append(current_date)
        current_date += timedelta(days=step_days)
    
    return np.array(positions), dates

# Function to plot orbits and positions for a list of bodies
def plot_orbits_and_positions(start_date, end_date, bodies, step_days=1):
    # Generate orbits and positions for each body
    plt.figure(figsize=(10, 10))

    # Set distinct colors and markers for each body
    colors = plt.cm.get_cmap("tab10", len(bodies))  # Generate distinct colors
    markers = ['o', 'x', '^', 's', 'D', '*', 'p', 'H', '+', '|']  # Different markers

    # Plot each body
    for idx, body in enumerate(bodies):
        orbit, dates = generate_orbit(body, start_date, end_date, step_days)
        
        # Plot the orbit path
        plt.plot(orbit[:, 0], orbit[:, 1], label=f'{body} Orbit', color=colors(idx), lw=1)
        
        # Plot the positions at the given times
        plt.scatter(orbit[:, 0], orbit[:, 1], color=colors(idx), marker=markers[idx % len(markers)], 
                    s=20, label=f'{body} Positions')

        # Highlight the starting positions
        plt.scatter(orbit[0, 0], orbit[0, 1], color=colors(idx), s=50, edgecolor='black', zorder=5)

    # Labels and grid
    plt.title(f'Orbits and Positions of Bodies from {start_date.date()} to {end_date.date()}')
    plt.xlabel('x [AU]')
    plt.ylabel('y [AU]')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()


load_spice_kernels()

# Example usage
start_date = datetime(2026, 1, 1)  # Start date
end_date = datetime(2026, 12, 31)  # End date

# List of bodies to plot (example: Earth, Mars, Venus, Jupiter)
bodies = ['EARTH', 'MARS BARYCENTER', 'VENUS']

# Plot orbits and positions for the list of bodies
plot_orbits_and_positions(start_date, end_date, bodies, step_days=15)
