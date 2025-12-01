import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import spiceypy as sp

from mpitg.src.almanac.ephemeris import load_spice_kernels
from mpitg.src.solver.lambert import lambert_universal


# ===========================================================
# Compute TOTAL C3 grid: departure × arrival
# ===========================================================
def compute_total_c3_grid(date_start="2024-04-01",
                          date_end="2024-12-01",
                          n_dates=80):

    load_spice_kernels()
    mu_sun = 1.32712440018e11

    t0 = sp.utc2et(date_start)
    t1 = sp.utc2et(date_end)

    dep_times = np.linspace(t0, t1, n_dates)
    arr_times = np.linspace(t0, t1, n_dates)

    TOTAL_C3 = np.full((n_dates, n_dates), np.nan)

    for i, t_dep in enumerate(dep_times):

        state_E = sp.spkezr("EARTH", t_dep, "ECLIPJ2000", "NONE", "SUN")[0]
        r1 = state_E[:3]
        v1_earth = state_E[3:6]

        for j, t_arr in enumerate(arr_times):
            if t_arr <= t_dep:
                continue

            tof = t_arr - t_dep

            state_M = sp.spkezr("MARS BARYCENTER", t_arr,
                                "ECLIPJ2000", "NONE", "SUN")[0]
            r2 = state_M[:3]
            v2_mars = state_M[3:6]

            try:
                v1t, v2t = lambert_universal(r1, r2, tof, mu_sun)
            except Exception:
                continue

            v_inf_dep = np.linalg.norm(v1t - v1_earth)
            v_inf_arr = np.linalg.norm(v2t - v2_mars)

            TOTAL_C3[i, j] = v_inf_dep**2 + v_inf_arr**2
            print(TOTAL_C3[i, j])

    return dep_times, arr_times, TOTAL_C3


# ===========================================================
# NASA-style porkchop plot (arrival × departure)
# ===========================================================
NASA_COLORS = [
    "#52006A",  # deep purple
    "#A000A0",  # magenta
    "#FF007F",  # hot pink/red
    "#FF5500",  # orange
    "#FFAA00",  # golden yellow
    "#CCFF00",  # yellow-green
    "#44FFAA",  # mint
    "#00DDFF",  # cyan
]

def plot_total_c3_porkchop(dep_et, arr_et, TOTAL_C3,
                           title="EARTH–MARS TOTAL C3 PORKCHOP"):

    def et2dt(et):
        utc = sp.et2utc(et, "C", 0)
        return dt.datetime.strptime(utc, "%Y %b %d %H:%M:%S")

    dep_dt = np.array([et2dt(t) for t in dep_et])
    arr_dt = np.array([et2dt(t) for t in arr_et])

    dep_num = mdates.date2num(dep_dt)
    arr_num = mdates.date2num(arr_dt)

    # Adjust these based on your mission window
    C3_LEVELS = [0, 40, 80, 120, 160, 200, 240, 280, 320]

    fig, ax = plt.subplots(figsize=(14, 8))

    cp = ax.contourf(
        arr_dt, dep_dt, TOTAL_C3,
        levels=C3_LEVELS,
        colors=NASA_COLORS,
        extend="max"
    )

    cl = ax.contour(
        arr_dt, dep_dt, TOTAL_C3,
        levels=C3_LEVELS,
        colors="black",
        linewidths=0.8
    )
    ax.clabel(cl, inline=True, fontsize=8, fmt="%.0f")

    # Grid overlay
    ax.set_xticks(arr_dt[::6])
    ax.set_yticks(dep_dt[::6])
    ax.grid(True, linestyle="-", linewidth=0.3, color="black")

    # Time-of-flight diagonal lines (6, 7, 8, 9 months)
    for months in [6, 7, 8, 9]:
        days = months * 30.437
        for d in dep_dt[::10]:
            a = d + dt.timedelta(days=days)
            ax.plot([a], [d], "k-", alpha=0.5)

    ax.set_xlabel("Arrival at Mars")
    ax.set_ylabel("Depart from Earth")
    ax.set_title(title, fontsize=16, fontweight="bold")

    cbar = plt.colorbar(cp)
    cbar.set_label("Total C3 (km²/s²)")

    plt.tight_layout()
    plt.show()


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    dep, arr, totC3 = compute_total_c3_grid()
    plot_total_c3_porkchop(dep, arr, totC3)
    print("Minimum total C3:", np.nanmin(totC3))
