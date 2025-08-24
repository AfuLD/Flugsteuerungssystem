import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import find_peaks

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

# Wing span
b = 47.574 # [m]

# Surface area - wing
S_wing = 256.32 # [m2]

# Wing MAC
MAC_wing = 6.0325 # [m]

# Maximum take-off weight
MTOW = 158800.0 # [kg]

# Cruise speed
V_cruise = 243.06 # [m/s]

# ------------------------------
# ENVIRONMENT
# ------------------------------ 
# Gravitational constant
g = 9.81 # [m/s2]

# Air density (at cruise altitude)
rho = 0.3108 # [kg/m3]

# ------------------------------
# NON-DIMENSIONAL RADII OF GYRATION
# ------------------------------ 
Rx = 0.301
Ry = 0.349
Rz = 0.434

# ------------------------------
# CONTROL SURFACE DEFLECTIONS
# ------------------------------ 
angles_ail = np.array([20, -20], dtype=np.float64)
angles_elev = np.array([25, -25], dtype=np.float64)
angles_rud = np.array([30, -30], dtype=np.float64)

# ------------------------------
# CONTROL DERIVATIVES
# ------------------------------ 
# Change in pitching moment due to elevator deflection
C_m_del_elev = -1.20

# Change in yawing moment due to rudder deflection
C_n_del_rud = -0.095 #0.90

# Change in rolling moment due to aileron deflection
C_l_del_ail = 0.05 #0.014 #0.05

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

# # Stability and control derivatives
# C_m_del_elev = -1.20
# C_n_del_rud = -0.095
# C_r_del_ail = 0.05
# C_l_p = -0.320
# C_n_r = -0.33

# C_m_0 = -0.041
# C_m_q = -25.5
# C_m_alfa = -1.60

# C_l_del_R = 0.005
# C_n_del_A = -0.0028

# C_n_p = 0.020

# C_l_beta = -0.095
# C_n_beta = 0.210

# # Control limits
# angles_ail = np.array([20, -20], dtype=np.float64)
# angles_elev = np.array([25, -25], dtype=np.float64)
# angles_rud = np.array([30, -30], dtype=np.float64)

# # Radii of gyration
# Rx, Ry, Rz = 0.301, 0.349, 0.434

# Inertia
Ixx = moment_of_inertia(b, MTOW, Rx, g)
Iyy = moment_of_inertia(L, MTOW, Ry, g)
Izz = moment_of_inertia((b+L)/2, MTOW, Rz, g)

# Dynamic pressure
q_cruise = 0.5 * rho * V_cruise**2

# ------------------------------
# BASELINE SIMULATION (3-DOF)
# ------------------------------
def simulate_normal(duration=200, dt=0.01):
    
    # time discretization
    T = np.arange(0, duration, dt)
    
    # Control surfaces deflection amplitudes
    defl_amplitude_ail = np.sum(np.abs(angles_ail))/2
    defl_amplitude_elev = np.sum(np.abs(angles_elev))/2
    defl_amplitude_rud = np.sum(np.abs(angles_rud))/2
    
    # Frequency of deflections
    freq = 2 / duration

    # Sinus-wave input for control surfaces
    deflection_ail = np.radians(defl_amplitude_ail * np.sin(2 * np.pi * T * freq))
    deflection_elev = np.radians(defl_amplitude_elev * np.sin(2 * np.pi * T * freq))
    deflection_rud = np.radians(defl_amplitude_rud * np.sin(2 * np.pi * T * freq))

    # Arrays initialization
    ang_vel_ail = np.zeros_like(T)
    ang_angle_ail = np.zeros_like(T)
    ang_vel_rud = np.zeros_like(T)
    ang_angle_rud = np.zeros_like(T)
    ang_vel_elev = np.zeros_like(T)
    ang_angle_elev = np.zeros_like(T)

    # Simulation loop
    for i in range(1, len(T)):
        C_l = 2*C_l_del_ail*deflection_ail[i] + C_l_beta*ang_angle_ail[i-1] + C_l_p*ang_vel_ail[i-1]
        M_roll = C_l * q_cruise * S_wing * MAC_wing
        ang_vel_ail[i] = ang_vel_ail[i-1] + (M_roll/Ixx) * dt
        ang_angle_ail[i] = ang_angle_ail[i-1] + ang_vel_ail[i] * dt

        C_n = C_n_del_rud*deflection_rud[i] + C_n_r*ang_vel_rud[i-1] + C_n_p*ang_vel_ail[i-1] - C_n_beta*ang_angle_rud[i-1]
        M_yaw = C_n * q_cruise * S_wing * MAC_wing
        ang_vel_rud[i] = ang_vel_rud[i-1] + (M_yaw/Izz) * dt
        ang_angle_rud[i] = ang_angle_rud[i-1] + ang_vel_rud[i] * dt

        C_m = C_m_0 + C_m_alfa*ang_angle_elev[i-1] + C_m_del_elev*deflection_elev[i] + C_m_q*MAC_wing/(2*V_cruise)*ang_vel_elev[i-1]
        M_pitch = C_m * q_cruise * S_wing * MAC_wing
        ang_vel_elev[i] = ang_vel_elev[i-1] + (M_pitch/Iyy) * dt
        ang_angle_elev[i] = ang_angle_elev[i-1] + ang_vel_elev[i] * dt

    return T, deflection_ail, deflection_rud, deflection_elev, ang_angle_ail, ang_angle_rud, ang_angle_elev

# ------------------------------
# ONE ENGINE INOPERATIVE (OEI)
# ------------------------------
def simulate_OEI(duration=20, dt=0.01):
    T = np.arange(0, duration, dt)
    engine_thrust = 266_000
    engine_arm = 6.0
    moment_engine = engine_thrust * engine_arm

    rudder_start_time = 2.0
    rudder_deflection_rad = np.radians(np.max(angles_rud))

    ang_vel_rud = np.zeros_like(T)
    ang_angle_rud = np.zeros_like(T)
    rudder_defl = np.zeros_like(T)

    for i in range(1, len(T)):
        if T[i] >= rudder_start_time:
            rudder_defl[i] = rudder_deflection_rad

        C_n = C_n_del_rud*rudder_defl[i] + C_n_r*ang_vel_rud[i-1] - C_n_beta*ang_angle_rud[i-1]
        M_aero = C_n * q_cruise * S_wing * MAC_wing
        M_total = M_aero + moment_engine

        ang_vel_rud[i] = ang_vel_rud[i-1] + (M_total/Izz) * dt
        ang_angle_rud[i] = ang_angle_rud[i-1] + ang_vel_rud[i] * dt

    return T, rudder_defl, ang_angle_rud

# ------------------------------
# FUEL IMBALANCE CASE
# ------------------------------

def simulate_fuel_imbalance(duration=20, dt=0.01):
    T = np.arange(0, duration, dt)
    # fuel_imbalance_mass = 500.0  # kg extra on one wing
    # wing_arm = b / 2
    # moment_imbalance = fuel_imbalance_mass * g * wing_arm

    # aileron_start_time = 5.0
    # aileron_deflection_rad = np.radians(np.max(angles_ail))

    # ------------------------------
    # FUEL IMBALANCE CONDITIONS
    # ------------------------------ 
    fuel_capacity_max = 90770 #[l]
    fuel_density = 0.8 # [kg/l]

    # assumption left wing tanks on half capacity, right wing tanks at 10% capacity
    m_fuel_left = fuel_capacity_max * fuel_density * 0.5 * 0.5
    m_fuel_right = fuel_capacity_max * fuel_density * 0.5 * 0.7

    # wing fuel y-axis CG-position assumpted to be in the middle of the wing
    fuel_CG = b / 4

    # Roll moment from fuel imbalance
    M_roll_fuel = (m_fuel_right - m_fuel_left) * g * fuel_CG  # [Nm]

    # Elevator starts counteracting at t = 2s (pilot reaction time assumption)
    ail_start_time = 5.0
    ail_deflection_deg = -15 # constant after 2s
    ail_deflection_rad = np.radians(ail_deflection_deg)
    
    M_del_ail = np.zeros_like(T)
    ang_acc_ail = np.zeros_like(T)
    ang_vel_ail = np.zeros_like(T)
    ang_angle_ail = np.zeros_like(T)
    
    ail_defl = np.zeros_like(T)
    M_total_roll = np.zeros_like(T)


    for i in range(1, len(T)):
        # Aileron deflection starts at 2 seconds
        if T[i] >= ail_start_time:
            ail_defl[i] = ail_deflection_rad
        
        # Aerodynamic pitch moment
        C_l = 2 * C_l_del_ail * ail_defl[i] + C_l_beta * ang_angle_ail[i-1] + C_l_p * ang_vel_ail[i-1] 
        # Roll moment
        M_del_ail[i] = C_l * q_cruise * S_wing * MAC_wing
        # Angular acceleration
        ang_acc_ail[i]  = M_del_ail[i] / Ixx
        # Angular velocity (Euler integration)
        ang_vel_ail[i]  = ang_vel_ail[i-1]  + ang_acc_ail[i] * dt
        # Angular position (Euler integration)
        ang_angle_ail[i]  = ang_angle_ail[i-1]  + ang_vel_ail[i] * dt
    
        M_aero_ail = C_l * q_cruise * S_wing * MAC_wing
    
        # Total pitch moment
        M_total_roll[i] = M_aero_ail + M_roll_fuel
    
        # Angular acceleration, velocity, angle (pitch)
        ang_acc_ail[i] = M_total_roll[i] / Ixx
        ang_vel_ail[i] = ang_vel_ail[i-1] + ang_acc_ail[i] * dt
        ang_angle_ail[i] = ang_angle_ail[i-1] + ang_vel_ail[i] * dt

    # phi_deg = np.degrees(ang_angle_ail)
    return T, ail_defl, ang_angle_ail

# ------------------------------
# CG SHIFT CASE
# ------------------------------
def simulate_CoG_shift(duration=20, dt=0.01):
    T = np.arange(0, duration, dt)
    
    delta_x_CG = 0.02 * MAC_wing  # [m] aft shift
    # Pitch moment from CG shift
    M_CGshift = MTOW * g * delta_x_CG  # [Nm], negative, nose-down moment

    # Elevator starts counteracting at t = 2s (pilot reaction time assumption)
    elev_start_time = 1.0
    elev_deflection_deg = -1 # constant after 2s
    elev_deflection_rad = np.radians(elev_deflection_deg)
    
    # for CG-shift
    elev_defl = np.zeros_like(T)
    M_total_pitch = np.zeros_like(T)
    
    cog_shift = 2.0  # meters forward
    extra_weight = MTOW * 0.05  # 5% of MTOW shifted
    moment_cog = extra_weight * g * cog_shift

    elevator_start_time = 5.0
    elevator_deflection_rad = np.radians(np.max(angles_elev))

    ang_vel_pitch = np.zeros_like(T)
    ang_angle_pitch = np.zeros_like(T)
    elevator_defl = np.zeros_like(T)
    
    ang_angle_elev = np.zeros_like(T)
    ang_vel_elev = np.zeros_like(T)
    ang_acc_elev = np.zeros_like(T)
    
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
    
        # Angular acceleration, velocity, angle (pitch)
        ang_acc_elev[i] = M_total_pitch[i] / Iyy
        ang_vel_elev[i] = ang_vel_elev[i-1] + ang_acc_elev[i] * dt
        ang_angle_elev[i] = ang_angle_elev[i-1] + ang_vel_elev[i] * dt

    # Convert pitch angle to degrees
    # theta_deg = np.degrees(ang_angle_elev)

    return T, elev_defl, ang_angle_elev

# ------------------------------
# PLOTTING FUNCTIONS
# ------------------------------
def plot_peaks_lines(ax, T, y):
    y_deg = np.degrees(y)
    # Find maxima
    peaks, _ = find_peaks(y_deg)
    # Find minima
    troughs, _ = find_peaks(-y_deg)
    # Plot vertical dashed lines
    for idx in peaks:
        ax.axvline(T[idx], color='black', linestyle='--', linewidth=1)
    for idx in troughs:
        ax.axvline(T[idx], color='black', linestyle='--', linewidth=1)

def plot_normal():
    T, da, dr, de, phi, psi, theta = simulate_normal()
    # fig, axs = plt.subplots(6, 1, figsize=(12, 8), constrained_layout=True)
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(8, 1, height_ratios=[1,1,0.6,1,1,0.6,1,1], hspace=0.8)  # gaps at rows 3 and 6

    axs = [
        fig.add_subplot(gs[0]),
        fig.add_subplot(gs[1]),
        fig.add_subplot(gs[3]),
        fig.add_subplot(gs[4]),
        fig.add_subplot(gs[6]),
        fig.add_subplot(gs[7]),
    ]
    
    axs[0].plot(T, np.degrees(da), label="Aileron Deflection [deg]", color='blue')
    plot_peaks_lines(axs[0], T, da)  # add vertical lines at max/min
    axs[0].set_title("Aileron Deflection vs Time")
    axs[0].set_ylabel("Deflection [deg]")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)
    
    # axs[1].plot(T, np.degrees(phi)); axs[1].set_title("Roll Angle")
    axs[1].plot(T, np.degrees(phi), label="Roll Angle [deg]", color='orange')
    plot_peaks_lines(axs[1], T, phi)  # add vertical lines at max/min
    axs[1].set_title("Roll Angle vs Time")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Angle [deg]")
    axs[1].legend(loc="upper right")
    axs[1].grid(True)
    
    # axs[2].plot(T, np.degrees(dr)); axs[2].set_title("Rudder Deflection")
    axs[2].plot(T, np.degrees(dr), label="Rudder Deflection [deg]", color='blue')
    plot_peaks_lines(axs[2], T, dr)  # add vertical lines at max/min
    axs[2].set_title("Rudder Deflection vs Time")
    axs[2].set_ylabel("Deflection [deg]")
    axs[2].legend(loc="upper right")
    axs[2].grid(True)

    # axs[3].plot(T, np.degrees(psi)); axs[3].set_title("Yaw Angle")
    axs[3].plot(T, np.degrees(psi), label="Yaw Angle [deg]", color='orange')
    plot_peaks_lines(axs[3], T, psi)  # add vertical lines at max/min
    axs[3].set_title("Yaw Angle vs Time")
    axs[3].set_xlabel("Time [s]")
    axs[3].set_ylabel("Angle [deg]")
    axs[3].legend(loc="upper right")
    axs[3].grid(True)

    
    # axs[4].plot(T, np.degrees(de)); axs[4].set_title("Elevator Deflection")
    axs[4].plot(T, np.degrees(de), label="Elevator Deflection [deg]", color='blue')
    plot_peaks_lines(axs[4], T, de)  # add vertical lines at max/min
    axs[4].set_title("Elevator Deflection vs Time")
    axs[4].set_ylabel("Deflection [deg]")
    axs[4].legend(loc="upper right")
    axs[4].grid(True)
    
    # axs[5].plot(T, np.degrees(theta)); axs[5].set_title("Pitch Angle")
    axs[5].plot(T, np.degrees(theta), label="Pitch Angle [deg]", color='orange')
    plot_peaks_lines(axs[5], T, theta)  # add vertical lines at max/min
    axs[5].set_title("Pitch Angle vs Time")
    axs[5].set_xlabel("Time [s]")
    axs[5].set_ylabel("Angle [deg]")
    axs[5].legend(loc="upper right")
    axs[5].grid(True)
    
    plt.subplots_adjust(top=0.97, bottom=0.11)
    plt.show()

def plot_OEI():
    T, rudder_defl, yaw = simulate_OEI()
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    # Rudder deflection
    axs[0].plot(T, np.degrees(rudder_defl), label='Rudder Deflection [deg]', color='blue')
    axs[0].set_ylabel('Deflection [deg]')
    axs[0].set_title('Rudder Deflection vs Time')
    axs[0].legend()
    axs[0].grid()

    # Yaw angle
    axs[1].plot(T, np.degrees(yaw), label='Yaw Angle [deg]', color='orange')
    axs[1].set_ylabel('Yaw [deg]')
    axs[1].set_title('Yaw Angle Response to OEI and Rudder Input')
    axs[1].legend()
    axs[1].grid()

    plt.show()


def plot_fuel_imbalance():
    T, ail_defl, roll = simulate_fuel_imbalance()
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    # Elevator deflection
    axs[0].plot(T, np.degrees(ail_defl), label='Aileron Deflection [deg]', color='blue')
    axs[0].set_ylabel('Deflection [deg]')
    axs[0].set_title('Aileron Deflection vs Time')
    axs[0].legend()
    axs[0].grid()

    # Pitch angle
    axs[1].plot(T, np.degrees(roll), label='Aileron Angle [deg]', color='orange')
    axs[1].set_ylabel('Aileron [deg]')
    axs[1].set_title('Aileron Angle Response to Fuel Imbalance and Aileron Input')
    axs[1].legend()
    axs[1].grid()


    plt.show()
    

def plot_CoG_shift():
    T, elev_defl, pitch = simulate_CoG_shift()
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    
    # Elevator deflection
    axs[0].plot(T, np.degrees(elev_defl), label='Elevator Deflection [deg]', color='blue')
    axs[0].set_ylabel('Deflection [deg]')
    axs[0].set_title('Elevator Deflection vs Time')
    axs[0].legend()
    axs[0].grid()

    # Pitch angle
    axs[1].plot(T, np.degrees(pitch), label='Pitch Angle [deg]', color='orange')
    axs[1].set_ylabel('Pitch [deg]')
    axs[1].set_title('Pitch Angle Response to CG-shift and Elevator Input')
    axs[1].legend()
    axs[1].grid()

    plt.show()
    
    # axs[0].plot(T, np.degrees(elev_defl)); axs[0].set_title("Elevator Deflection")
    # axs[1].plot(T, np.degrees(pitch)); axs[1].set_title("Pitch Response to CoG Shift")
    # plt.show()

# ------------------------------
# BUTTON INTERFACE
# ------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
plt.subplots_adjust(bottom=0.35)
ax.set_axis_off()

# Top button (centered)
ax_btn_main = plt.axes([0.32, 0.25, 0.36, 0.15])  # x, y, width, height
btn_main = Button(ax_btn_main, "3-DOF-Simulation")

# Bottom row of 3 case buttons (centered)
btn_width = 0.28
btn_height = 0.12
btn_spacing = 0.04
total_width = 3*btn_width + 2*btn_spacing
start_x = (1 - total_width) / 2  # center the row

ax_btn_case1 = plt.axes([start_x, 0.05, btn_width, btn_height])
btn_case1 = Button(ax_btn_case1, "Case 1: OEI")

ax_btn_case2 = plt.axes([start_x + btn_width + btn_spacing, 0.05, btn_width, btn_height])
btn_case2 = Button(ax_btn_case2, "Case 2: Fuel Imbalance")

ax_btn_case3 = plt.axes([start_x + 2*(btn_width + btn_spacing), 0.05, btn_width, btn_height])
btn_case3 = Button(ax_btn_case3, "Case 3: CG Shift")

# Connect buttons to functions
btn_main.on_clicked(lambda event: plot_normal())
btn_case1.on_clicked(lambda event: plot_OEI())
btn_case2.on_clicked(lambda event: plot_fuel_imbalance())
btn_case3.on_clicked(lambda event: plot_CoG_shift())

plt.show()
