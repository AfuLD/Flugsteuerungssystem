import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ------------------------------
# FUNCTIONS
# ------------------------------ 
def moment_of_inertia (a, W, R, g):
    """
    Calculates moments of intertia of an aircraft (Ixx or Iyy or Izz)

    Parameters:
    - a: Aircraft geometry (span or length or e = (b+L)/2)
    - W: Aircraft mass
    - R: Non-dimensional radius of gyration
    - g: Gravitational constant

    Output:
    - I: Moment of inertia 
    """
    I = a**2 * W * R**2 / (4 * g)
    
    return I
    

# ------------------------------
# AIRCRAFT DATA
# ------------------------------  

# Aircraft length
L = 53.670 # [m]

# Aircraft CoG-position
x_CG = 23.778 # [m]

# Wing span
b = 47.574 # [m]

# Surface area - wing
S_wing = 256.32 # [m2]

# Wing MAC
MAC_wing = 6.0325 # [m]

# Surface area - vertical stabilizer (incl. elevator)
S_vert = 46.15 # [m2]

# Surface area - horizontal stabilizer (incl. rudder)
S_hor = 77.67 # [m2]

# Maximum take-off weight
MTOW = 158800.0 # [kg]

# Cruise speed
V_cruise = 243.06 # [m/s]

# Cruise altitude
h_cruise = 12000.0 # [m]

# ------------------------------
# ENVIRONMENT
# ------------------------------ 
# Gravitational constant
g = 9.81 # [m/s2]

# Air density (at cruise altitude)
rho = 0.3108 # [kg/m3]


# ------------------------------
# CONTROL DERIVATIVES
# ------------------------------ 
# Change in pitching moment due to elevator deflection
C_m_del_elev = 1.20

# Change in yawing moment due to rudder deflection
C_n_del_rud = 0.90

# Change in rolling moment due to aileron deflection
C_r_del_ail = 0.014

# damping
# C_m_q = 25.5 # pitch
# C_n_r = 0.33 # yaw
# C_r_p = 0.32 # roll

C_r_st = 0.095 # roll 
C_n_st = 0.33 # yaw

# ------------------------------
# CONTROL SURFACE DEFLECTIONS
# ------------------------------ 
# angles_ail = np.array([7, -7], dtype=np.float64)
# angles_elev = np.array([10, -10], dtype=np.float64)
# angles_rud = np.array([30, -30], dtype=np.float64)

angles_ail = np.array([7, -7], dtype=np.float64)
angles_elev = np.array([5, -5], dtype=np.float64)
angles_rud = np.array([5, -5], dtype=np.float64)

# ------------------------------
# NON-DIMENSIONAL RADII OF GYRATION
# ------------------------------ 
Rx = 0.301
Ry = 0.349
Rz = 0.434

# ------------------------------
# SIMULATION PARAMETERS
# ------------------------------ 
duration = 200
dt = 0.05
T = np.arange(0, duration, dt)

# ------------------------------
# MAIN
# ------------------------------
# Calculate moments of inertia 
Ixx = moment_of_inertia(b, MTOW, Rx, g)
Iyy = moment_of_inertia(L, MTOW, Ry, g)
Izz = moment_of_inertia((b+L)/2, MTOW, Rz, g)

# Calculate dynamic pressure in cruise flight
q_cruise = 0.5 * rho * V_cruise**2

# deflection amplitudes [deg]
defl_amplitude_ail = np.sum(np.abs(angles_ail))/2
defl_amplitude_elev = np.sum(np.abs(angles_elev))/2
defl_amplitude_rud = np.sum(np.abs(angles_rud))/2

# frequency of control surfaces deflection_ail
freq = 2 / duration

deflection_ail = np.radians(defl_amplitude_ail * np.sin(2 * np.pi * T * freq))
deflection_elev = np.radians(defl_amplitude_elev * np.sin(2 * np.pi * T * freq))
deflection_rud = np.radians(defl_amplitude_rud * np.sin(2 * np.pi * T * freq))

# Initialize arrays
M_del_elev = np.zeros_like(T)
M_del_ail = np.zeros_like(T)
M_del_rud = np.zeros_like(T)

ang_acc_elev = np.zeros_like(T)
ang_acc_ail = np.zeros_like(T)
ang_acc_rud = np.zeros_like(T)

ang_vel_elev = np.zeros_like(T)
ang_vel_ail = np.zeros_like(T)
ang_vel_rud = np.zeros_like(T)

ang_angle_elev = np.zeros_like(T)
ang_angle_ail = np.zeros_like(T)
ang_angle_rud = np.zeros_like(T)


# Main loop
for i in range(1, len(T)):
    
    # Roll moment coefficient
    C_r = C_r_del_ail * deflection_ail[i] - C_r_st * ang_vel_ail[i-1]
    # Roll moment
    M_del_ail[i] = C_r * q_cruise * S_wing * MAC_wing
    # Angular acceleration
    ang_acc_ail[i]  = M_del_ail[i] / Ixx
    # Angular velocity (Euler integration)
    ang_vel_ail[i]  = ang_vel_ail[i-1]  + ang_acc_ail[i] * dt
    # Angular position (Euler integration)
    ang_angle_ail[i]  = ang_angle_ail[i-1]  + ang_vel_ail[i] * dt
    
    # Yaw moment coefficient
    C_n = C_n_del_rud * deflection_rud[i] - C_n_st * ang_angle_rud[i-1]
    # Yaw moment
    M_del_rud[i] = C_n * q_cruise * S_wing * MAC_wing
    # Angular acceleration
    ang_acc_rud[i]  = M_del_rud[i] / Izz
    # Angular velocity (Euler integration)
    ang_vel_rud[i]  = ang_vel_rud[i-1]  + ang_acc_rud[i] * dt
    # Angular velocity (Euler integration)
    ang_angle_rud[i]  = ang_angle_rud[i-1]  + ang_vel_rud[i] * dt
    
    

# Convert to degrees
phi_deg = np.degrees(ang_angle_ail)
psi_deg = np.degrees(ang_angle_rud)


# ------------------------------
# PLOTTING
# ------------------------------

# Create subplots
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 6), constrained_layout=True)

# Plot Aileron Deflection
plot_steps = 2000

# axs[0].plot(T[:plot_steps], np.degrees(deflection_ail)[:plot_steps], label="Aileron Deflection [deg]", color='blue')
axs[0].plot(T, np.degrees(deflection_ail), label="Aileron Deflection [deg]", color='blue')
axs[0].set_title("Aileron Deflection vs Time")
axs[0].set_ylabel("Deflection [deg]")
axs[0].legend(loc="upper right")
axs[0].grid(True)

# Plot Roll Angle (phi)
# axs[1].plot(T[:plot_steps], phi_deg[:plot_steps], label="Roll Angle φ [deg]", color='orange')
axs[1].plot(T, phi_deg, label="Roll Angle [deg]", color='orange')
axs[1].set_title("Roll Angle vs Time")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Angle [deg]")
axs[1].legend(loc="upper right")
axs[1].grid(True)

# axs[0].plot(T[:plot_steps], np.degrees(deflection_ail)[:plot_steps], label="Aileron Deflection [deg]", color='blue')
axs[2].plot(T, np.degrees(deflection_rud), label="Rudder Deflection [deg]", color='blue')
axs[2].set_title("Rudder Deflection vs Time")
axs[2].set_ylabel("Deflection [deg]")
axs[2].legend(loc="upper right")
axs[2].grid(True)

# axs[1].plot(T[:plot_steps], phi_deg[:plot_steps], label="Roll Angle φ [deg]", color='orange')
axs[3].plot(T, psi_deg, label="Yaw Angle [deg]", color='orange')
axs[3].set_title("Yaw Angle vs Time")
axs[3].set_xlabel("Time [s]")
axs[3].set_ylabel("Angle [deg]")
axs[3].legend(loc="upper right")
axs[3].grid(True)


# Show plots
plt.show()