import numpy as np
import matplotlib.pyplot as plt

# Case: cargo attachment fails causes it to slide to the aft of the aircraft cargo compartment
# resulting CG-shift leads to a pitch moment along the lateral axis

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
# CG-SHIFT CONDITIONS
# ------------------------------ 
delta_x_CG = 0.02 * MAC_wing  # [m] aft shift
# Pitch moment from CG shift
M_CGshift = MTOW * g * delta_x_CG  # [Nm], negative, nose-down moment

# Elevator starts counteracting at t = 2s (pilot reaction time assumption)
elev_start_time = 1.0
elev_deflection_deg = -1 # constant after 2s
elev_deflection_rad = np.radians(elev_deflection_deg)

# deflection_elev_deg = np.zeros_like(T)
# deflection_step_time = 2  # seconds
# deflection_elev_deg[T >= deflection_step_time] = -1.0  # elevator deflection in degrees (negative for nose down)

# deflection_elev_rad = np.radians(deflection_elev_deg)

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

# for CG-shift
elev_defl = np.zeros_like(T)
M_total_pitch = np.zeros_like(T)

# ------------------------------
# Simulation loop
# ------------------------------
for i in range(1, len(T)):
    # Elevator deflection starts at 2 seconds
    if T[i] >= elev_start_time:
        elev_defl[i] = elev_deflection_rad
        
    # Aerodynamic pitch moment
    C_m = (C_m_0 +
           C_m_alfa * ang_angle_elev[i-1] +
           C_m_del_elev * elev_defl[i] +
           C_m_q * MAC_wing / (2 * V_cruise) * ang_vel_elev[i-1])
    
    M_aero_elev = C_m * q_cruise * S_wing * MAC_wing
    
    # Total pitch moment
    M_total_pitch[i] = M_aero_elev - M_CGshift
    
    # # Store total pitch moment for plotting
    # pitch_moment[i] = M_total
    
    # Angular acceleration, velocity, angle (pitch)
    ang_acc_elev[i] = M_total_pitch[i] / Iyy
    ang_vel_elev[i] = ang_vel_elev[i-1] + ang_acc_elev[i] * dt
    ang_angle_elev[i] = ang_angle_elev[i-1] + ang_vel_elev[i] * dt

# Convert pitch angle to degrees
theta_deg = np.degrees(ang_angle_elev)

# ------------------------------
# Plotting results
# ------------------------------


fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)

# Elevator deflection
axs[0].plot(T, np.degrees(elev_defl), label='Elevator Deflection [deg]', color='blue')
axs[0].set_ylabel('Deflection [deg]')
axs[0].set_title('Elevator Deflection vs Time')
axs[0].legend()
axs[0].grid()

# Pitch angle
axs[1].plot(T, theta_deg, label='Pitch Angle [deg]', color='orange')
axs[1].set_ylabel('Pitch [deg]')
axs[1].set_title('Pitch Angle Response to CG-shift and Elevator Input')
axs[1].legend()
axs[1].grid()


plt.show()


# plt.figure(figsize=(10,8))

# plt.subplot(2,1,1)
# plt.plot(T, deflection_elev_deg, label='Elevator Deflection [deg]', color='blue')
# plt.ylabel('Deflection [deg]')
# plt.title('Elevator Deflection')
# plt.grid(True)
# plt.legend()

# plt.subplot(4,1,2)
# plt.plot(T, theta_deg, label='Pitch Angle [deg]', color='green')
# plt.ylabel('Pitch Angle [deg]')
# plt.title('Pitch Angle Response')
# plt.grid(True)
# plt.legend()

# # plt.subplot(4,1,3)
# # plt.plot(T, np.degrees(ang_vel_elev), label='Pitch Rate [deg/s]', color='orange')
# # plt.ylabel('Pitch Rate [deg/s]')
# # plt.title('Pitch Rate Response')
# # plt.grid(True)
# # plt.legend()

# # plt.subplot(4,1,4)
# # plt.plot(T, pitch_moment, label='Total Pitch Moment [Nm]', color='red')
# # plt.ylabel('Moment [Nm]')
# # plt.xlabel('Time [s]')
# # plt.title('Total Pitch Moment (Aerodynamic + CG shift)')
# # plt.grid(True)
# # plt.legend()

# plt.tight_layout()
# plt.show()
