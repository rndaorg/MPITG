# earth_escape_simulation.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ----------------------------
# Constants
# ----------------------------
mu_earth = 398600.4418      # km^3/s^2
mu_moon = 4902.8            # km^3/s^2
mu_sun = 132712440018.0     # km^3/s^2

R_earth = 6378.0            # km
r0_alt = 200.0              # km
r0 = R_earth + r0_alt       # km
v0 = np.sqrt(mu_earth / r0) # km/s

state0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])

thrust_acc = 0.0001  # km/s^2

# Ephemeris approximations (circular, coplanar)
T_moon = 27.3217 * 24 * 3600
T_sun = 365.25 * 24 * 3600
R_moon = 384400.0
R_sun = 149.6e6

def moon_position(t):
    theta = 2 * np.pi * t / T_moon
    return np.array([R_moon * np.cos(theta), R_moon * np.sin(theta), 0.0])

def sun_position(t):
    theta = 2 * np.pi * t / T_sun
    return np.array([R_sun * np.cos(theta), R_sun * np.sin(theta), 0.0])

def derivative(t, state):
    x, y, z, vx, vy, vz = state
    r_sc = np.array([x, y, z])
    v_sc = np.array([vx, vy, vz])
    
    # Earth
    r_e = np.linalg.norm(r_sc)
    a_earth = -mu_earth * r_sc / r_e**3 if r_e > 0 else np.zeros(3)
    
    # Moon
    r_m = moon_position(t)
    r_sc2m = r_sc - r_m
    d = np.linalg.norm(r_sc2m)
    a_moon = -mu_moon * r_sc2m / d**3 if d > 0 else np.zeros(3)
    
    # Sun
    r_s = sun_position(t)
    r_sc2s = r_sc - r_s
    d = np.linalg.norm(r_sc2s)
    a_sun = -mu_sun * r_sc2s / d**3 if d > 0 else np.zeros(3)
    
    # Thrust (prograde)
    v_norm = np.linalg.norm(v_sc)
    thrust_dir = v_sc / v_norm if v_norm > 1e-10 else np.zeros(3)
    a_thrust = thrust_acc * thrust_dir
    
    a_total = a_earth + a_moon + a_sun + a_thrust
    return np.array([vx, vy, vz, a_total[0], a_total[1], a_total[2]])

def rk4_step(t, state, dt):
    k1 = derivative(t, state)
    k2 = derivative(t + dt/2, state + dt/2 * k1)
    k3 = derivative(t + dt/2, state + dt/2 * k2)
    k4 = derivative(t + dt, state + dt * k3)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# ----------------------------
# Integration
# ----------------------------
dt = 60.0
t_max = 500000.0
t = 0.0
state = state0.copy()

t_list = [t]
traj = [state.copy()]

escape_radius = 1e6
while t < t_max:
    state = rk4_step(t, state, dt)
    t += dt
    x, y, z = state[0:3]
    vx, vy, vz = state[3:6]
    r = np.linalg.norm([x, y, z])
    v = np.linalg.norm([vx, vy, vz])
    energy = v**2 / 2 - mu_earth / r
    t_list.append(t)
    traj.append(state.copy())
    if r > escape_radius or energy >= 0:
        print(f"Escaped at t = {t/3600:.2f} h, r = {r:.0f} km")
        break

t_array = np.array(t_list)
traj_array = np.array(traj)

# ----------------------------
# Export
# ----------------------------
np.savez('earth_escape_spiral.npz', 
         time_s=t_array,
         x=traj_array[:,0], y=traj_array[:,1], z=traj_array[:,2],
         vx=traj_array[:,3], vy=traj_array[:,4], vz=traj_array[:,5])
print("Trajectory saved.")

# ----------------------------
# Static Plots
# ----------------------------
x, y, z = traj_array[:,0], traj_array[:,1], traj_array[:,2]

# 2D
plt.figure(figsize=(10,8))
plt.plot(x, y, 'b-', linewidth=1)
plt.plot(0,0,'go', markersize=10, label='Earth')
plt.axis('equal')
plt.title('Low-Thrust Earth Escape Spiral (2D)')
plt.xlabel('X (km)'); plt.ylabel('Y (km)')
plt.grid(True); plt.legend()
plt.savefig('escape_2d.png', dpi=150)
plt.show()

# 3D
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'b-')
ax.scatter([0],[0],[0], color='g', s=100, label='Earth')
max_range = np.ptp() #np.array([x.ptp(), y.ptp(), z.ptp()]).max() / 2.0
mid_x, mid_y, mid_z = np.mean(x), np.mean(y), np.mean(z)
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_title('Low-Thrust Earth Escape Spiral (3D)')
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
ax.legend()
plt.savefig('escape_3d.png', dpi=150)
plt.show()

# ----------------------------
# 2D Animation
# ----------------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal')
ax.set_xlim(x.min()-5000, x.max()+5000)
ax.set_ylim(y.min()-5000, y.max()+5000)
ax.plot(0,0,'go', markersize=10)
line, = ax.plot([], [], 'b-', lw=1.5)
point, = ax.plot([], [], 'ro', ms=6)
ax.set_title('Escape Spiral (2D Animation)')
ax.grid(True)

def init(): 
    line.set_data([], []); point.set_data([], []); return line, point
def animate(i):
    line.set_data(x[:i+1], y[:i+1])
    point.set_data(x[i], y[i])
    return line, point

anim = animation.FuncAnimation(fig, animate, frames=len(x), init_func=init,
                               blit=True, interval=30)
anim.save('escape_2d.mp4', writer='ffmpeg', fps=30, dpi=150)
plt.close()
print("2D animation saved as escape_2d.mp4")

# ----------------------------
# 3D Animation (uncomment to run locally)
# ----------------------------
"""
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
max_range = np.array([x.ptp(), y.ptp(), z.ptp()]).max() / 2.0
mid = np.array([x.mean(), y.mean(), z.mean()])
ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
ax.scatter([0],[0],[0], color='g', s=100)
line, = ax.plot([], [], [], 'b-', lw=1.5)
point, = ax.plot([], [], [], 'ro', ms=6)

def init3d():
    line.set_data([], []); line.set_3d_properties([])
    point.set_data([], []); point.set_3d_properties([])
    return line, point

def animate3d(i):
    line.set_data(x[:i+1], y[:i+1])
    line.set_3d_properties(z[:i+1])
    point.set_data([x[i]], [y[i]])
    point.set_3d_properties([z[i]])
    return line, point

anim3d = animation.FuncAnimation(fig, animate3d, frames=len(x), init_func=init3d,
                                 blit=True, interval=30)
anim3d.save('escape_3d.mp4', writer='ffmpeg', fps=30, dpi=150)
plt.close()
print("3D animation saved as escape_3d.mp4")
"""