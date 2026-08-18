import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import spiceypy as sp

from mpitg.src.almanac.ephemeris import load_spice_kernels
from mpitg.src.solver.izzo import lambert_izzo2015
from mpitg.src.solver.lambert import lambert_universal


# ===========================================================
# Compute TOTAL C3 grid: departure × arrival
# ===========================================================
def compute_total_c3_grid(date_start="2029-01-01",
                          date_end="2031-01-01",
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
                v1t, v2t = lambert_izzo2015(r1, r2, tof, mu_sun)
            except Exception:
                continue

            v_inf_dep = np.linalg.norm(v1t - v1_earth)
            v_inf_arr = np.linalg.norm(v2t - v2_mars)

            TOTAL_C3[i, j] = v_inf_dep**2 + v_inf_arr**2
            #print(TOTAL_C3[i, j])

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

    # Convert ET → datetime
    def et2dt(et):
        utc = sp.et2utc(et, "C", 0)
        return dt.datetime.strptime(utc, "%Y %b %d %H:%M:%S")

    dep_dt = np.array([et2dt(t) for t in dep_et])
    arr_dt = np.array([et2dt(t) for t in arr_et])

    # Convert to matplotlib floats
    dep_num = mdates.date2num(dep_dt)
    arr_num = mdates.date2num(arr_dt)

    # Mask invalids to remove jagged white noise
    C3 = np.ma.masked_invalid(TOTAL_C3)

    C3_LEVELS = np.array([0, 40, 80, 120, 160, 200, 240, 280, 320])

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- Filled contour ---
    cp = ax.contourf(
        dep_num, arr_num, C3,
        levels=C3_LEVELS,
        cmap="turbo",
        extend="max"
    )

    # --- Line contours ---
    cl = ax.contour(
        dep_num, arr_num, C3,
        levels=C3_LEVELS,
        colors="black",
        linewidths=0.8
    )
    ax.clabel(cl, inline=True, fontsize=8, fmt="%.0f")

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)

    # Grid
    ax.grid(True, linestyle="-", linewidth=0.3, color="black", alpha=0.4)

    # --- Time-of-flight diagonal lines ---
    for months in [6, 7, 8, 9]:
        days = months * 30.437
        tof_arr = dep_num + days
        ax.plot(tof_arr, dep_num, "k--", linewidth=1, alpha=0.7,
                label=f"{months} mo TOF" if months == 6 else None)

    # Labels
    ax.set_ylabel("Arrival at Mars")
    ax.set_xlabel("Departure from Earth")
    ax.set_title(title, fontsize=16, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(cp)
    cbar.set_label("Total C3 (km²/s²)")

    plt.tight_layout()
    plt.show()


# ===========================================================
# MAIN
# ===========================================================
if __name__ == "__main__":
    dep, arr, totC3 = compute_total_c3_grid()
    print(totC3)
    plot_total_c3_porkchop(dep, arr, totC3)
    print("Minimum total C3:", np.nanmin(totC3))
