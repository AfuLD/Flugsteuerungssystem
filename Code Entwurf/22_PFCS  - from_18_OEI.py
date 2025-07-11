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

# Angle of wing incidence # [deg]
alfa_inc = 4

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
C_m_del_elev = -1.20

# Change in yawing moment due to rudder deflection
C_n_del_rud = -0.095 #0.90

# Change in rolling moment due to aileron deflection
C_r_del_ail = 0.05 #0.014 #0.05

# Stability derivative for roll
C_l_p = -0.320

# Stability derivative for yaw
C_n_r = -0.33

# Stability derivatives for pitch
C_m_0 = -0.041
C_m_q = -25.5
C_m_alfa = -1.60

# Cross coupling
# change in rolling moment coefficient due to rudder deflection
C_l_del_R = 0.005
# change in yawing moment coefficient due to aileron deflection
C_n_del_A = -0.0028

# change in yawing moment coefficient due to roll rate
C_n_p = 0.020

# Influence of the sideslip angle
# Change in rolling moment coefficient due to a unit of sideslip angle
C_l_beta = -0.095
# Change in yawing moment coefficient due to a unit of sideslip angle
C_n_beta = 0.210

# ------------------------------
# CONTROL SURFACE DEFLECTIONS
# ------------------------------ 
angles_ail = np.array([20, -20], dtype=np.float64)
angles_elev = np.array([25, -25], dtype=np.float64)
angles_rud = np.array([30, -30], dtype=np.float64)

# angles_ail = np.array([7, -7], dtype=np.float64)
# angles_elev = np.array([5, -5], dtype=np.float64)
# angles_rud = np.array([5, -5], dtype=np.float64)

# ------------------------------
# NON-DIMENSIONAL RADII OF GYRATION
# ------------------------------ 
Rx = 0.301
Ry = 0.349
Rz = 0.434

# ------------------------------
# SIMULATION PARAMETERS
# ------------------------------ 
duration = 20
dt = 0.01
T = np.arange(0, duration, dt)

# ------------------------------
# OEI CONDITIONS
# ------------------------------ 

# One engine inoperative (assumed at t=0)
engine_thrust = 266_000  # [N]
engine_arm = 6.0         # [m] from aircraft centerline
moment_engine = engine_thrust * engine_arm  # [Nm] positive moment - dead engine on the right side

# Rudder starts counteracting at t = 2s (pilot reaction time assumption)
rudder_start_time = 2.0
rudder_deflection_deg = np.max(angles_rud)#25  # constant after 2s
rudder_deflection_rad = np.radians(rudder_deflection_deg)

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

# for OEI
rudder_defl = np.zeros_like(T)
M_total_yaw = np.zeros_like(T)


# ------------------------------
# SIMULATION LOOP
# ------------------------------
for i in range(1, len(T)):

    # Rudder deflection starts at 2 seconds
    if T[i] >= rudder_start_time:
        rudder_defl[i] = rudder_deflection_rad

    # Aerodynamic yaw moment
    C_n = C_n_del_rud * rudder_defl[i] + C_n_r * ang_vel_rud[i-1] + C_n_p * 0 - C_n_beta * ang_angle_rud[i-1]
    M_aero_rud = C_n * q_cruise * S_wing * MAC_wing

    # Total yaw moment
    M_total_yaw[i] = M_aero_rud + moment_engine  # Engine moment causes yaw right, rudder counters left

    # Angular acceleration
    ang_acc_rud[i] = M_total_yaw[i] / Izz
    ang_vel_rud[i] = ang_vel_rud[i-1] + ang_acc_rud[i] * dt
    ang_angle_rud[i] = ang_angle_rud[i-1] + ang_vel_rud[i] * dt

# ------------------------------
# PLOTTING
# ------------------------------

psi_deg = np.degrees(ang_angle_rud)
print(psi_deg[-1])

fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

# Rudder deflection
axs[0].plot(T, np.degrees(rudder_defl), label='Rudder Deflection [deg]', color='blue')
axs[0].set_ylabel('Deflection [deg]')
axs[0].set_title('Rudder Deflection vs Time')
axs[0].legend()
axs[0].grid()

# Yaw angle
axs[1].plot(T, psi_deg, label='Yaw Angle [deg]', color='orange')
axs[1].set_ylabel('Yaw [deg]')
axs[1].set_title('Yaw Angle Response to OEI and Rudder Input')
axs[1].legend()
axs[1].grid()


plt.show()
