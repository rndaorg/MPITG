import matplotlib.pyplot as plt
import numpy as np
import spiceypy as sp

def plot_porkchop(dep_times, tof_days, C3):
    """
    Plot a C3 porkchop diagram using matplotlib only.
    """

    # Convert ET ➝ calendar dates for the x-axis
    dep_dates = np.array([sp.et2utc(t, "C", 0) for t in dep_times])

    # Make mesh for plotting
    D, T = np.meshgrid(dep_dates, tof_days, indexing="ij")

    # Plot
    plt.figure(figsize=(12, 8))

    # Contour levels (km^2/s^2) — typical Mars window range
    levels = [5, 7.5, 10, 12.5, 15, 20, 25, 30]

    cp = plt.contour(
        D,
        T,
        C3,
        levels=levels,
        cmap="viridis",
        linewidths=1.0
    )

    plt.clabel(cp, inline=True, fontsize=8, fmt="%.1f")

    plt.title("Earth → Mars Porkchop Plot (C3)")
    plt.xlabel("Departure Date (UTC)")
    plt.ylabel("Time of Flight (days)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
