import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import tkinter as tk
from threading import Thread
import time


# ─── Minimal 6-DOF model (rigid body, Euler angles) ──────────────────────
g = 9.80665

def isa_rho(h_m):
    # Simple ISA density: ρ ≈ 1.225 * (1 - 2.25577e-5*h)^4.256   (h in meters)
    h = max(0.0, float(h_m))
    return 1.225 * (1 - 2.25577e-5*h)**4.256

# ——— Parameters from '747 parameters.md' (converted to SI) ———
# Helper conversions
FT2M = 0.3048
FT2_TO_M2 = 0.09290304
FPS2MPS = 0.3048
SLUGFT2_TO_KGM2 = 1.3558179483314004
LBF_TO_N = 4.4482216152605

# Raw condition data (from the .md file)
_MD = {
    "Approach": {
        "alt_ft": 3280.839895013123,  # 1000 m start altitude
        "U_fps": 221.0,
        "theta0_deg": 8.5,
        "S_ft2": 5500.0,
        "b_ft": 196.0,
        "c_ft": 27.3,
        "W_lbf": 564000.0,
        "Ixx_slugft2": 13.7e6,
        "Iyy_slugft2": 30.5e6,
        "Izz_slugft2": 43.1e6,
        # steady coeffs
        "CL1": 1.76, "CD1": 0.263, "CTx1": 0.263,
        # longitudinal derivatives
        "Cm_u": 0.071, "Cm_alpha": -1.45, "Cm_adot": -3.3, "Cm_q": -21.4,
        "CL_u": -0.22, "CL_alpha": 5.67, "CL_adot": 6.7, "CL_q": 5.65,
        "CD_u": 1.13, "CD_alpha": 0.0,
        "CL_de": 0.36, "CD_de": 0.0, "Cm_de": -1.40,
        # lateral-directional derivatives
        "Cl_beta": -0.281, "Cl_p": -0.502, "Cl_r": 0.195, "Cl_da": 0.053, "Cl_dr": 0.0,
        "Cn_beta": 0.184, "Cn_p": -0.222, "Cn_r": -0.36, "Cn_da": 0.0083, "Cn_dr": -0.113,
        "CY_beta": -1.08, "CY_dr": 0.179
    },
    "CruiseHigh": {
        "alt_ft": 40000.0,
        "U_fps": 871.0,
        "theta0_deg": 2.4,
        "S_ft2": 5500.0,
        "b_ft": 196.0,
        "c_ft": 27.3,
        "W_lbf": 636636.0,
        "Ixx_slugft2": 18.2e6,
        "Iyy_slugft2": 33.1e6,
        "Izz_slugft2": 49.7e6,
        "CL1": 0.52, "CD1": 0.045, "CTx1": 0.045,
        "Cm_u": -0.09, "Cm_alpha": -1.60, "Cm_adot": -9.0, "Cm_q": -25.5,
        "CL_u": -0.23, "CL_alpha": 5.5, "CL_adot": 8.0, "CL_q": 7.8,
        "CD_u": 0.50, "CD_alpha": 0.22,
        "CL_de": 0.30, "CD_de": 0.0, "Cm_de": -1.20,
        "Cl_beta": -0.095, "Cl_p": -0.320, "Cl_r": 0.200, "Cl_da": 0.014, "Cl_dr": 0.005,
        "Cn_beta": 0.210, "Cn_p": 0.020, "Cn_r": -0.33, "Cn_da": -0.0028, "Cn_dr": -0.095,
        "CY_beta": -0.90, "CY_dr": 0.060
    },
    "CruiseLow": {
        "alt_ft": 20000.0,
        "U_fps": 673.0,
        "theta0_deg": 2.5,
        "S_ft2": 5500.0,
        "b_ft": 196.0,
        "c_ft": 27.3,
        "W_lbf": 636636.0,
        "Ixx_slugft2": 18.2e6,
        "Iyy_slugft2": 33.1e6,
        "Izz_slugft2": 49.7e6,
        "CL1": 0.40, "CD1": 0.025, "CTx1": 0.025,
        "Cm_u": 0.013, "Cm_alpha": -1.00, "Cm_adot": -4.0, "Cm_q": -20.5,
        "CL_u": 0.13, "CL_alpha": 4.4, "CL_adot": 7.0, "CL_q": 6.6,
        "CD_u": 0.20, "CD_alpha": 0.0,
        "CL_de": 0.32, "CD_de": 0.0, "Cm_de": -1.30,
        "Cl_beta": -0.160, "Cl_p": -0.340, "Cl_r": 0.130, "Cl_da": 0.013, "Cl_dr": 0.008,
        "Cn_beta": 0.160, "Cn_p": -0.026, "Cn_r": -0.28, "Cn_da": 0.0018, "Cn_dr": -0.100,
        "CY_beta": -0.90, "CY_dr": 0.120
    }
}


def _build_params_from_md(name: str):
    d = _MD[name]
    # geometry & masses
    S = d["S_ft2"] * FT2_TO_M2
    b = d["b_ft"] * FT2M
    cbar = d["c_ft"] * FT2M
    W_N = d["W_lbf"] * LBF_TO_N
    m = W_N / g
    I = (d["Ixx_slugft2"] * SLUGFT2_TO_KGM2,
         d["Iyy_slugft2"] * SLUGFT2_TO_KGM2,
         d["Izz_slugft2"] * SLUGFT2_TO_KGM2)

    # aero coefficients from md
    CL1, CD1 = d["CL1"], d["CD1"]
    CLa, CLq = d["CL_alpha"], d["CL_q"]
    Cm_a, Cmq, Cm_de = d["Cm_alpha"], d["Cm_q"], d["Cm_de"]  # note: file uses Cm_de<0 (TE-down → nose-down)
    CL_de = d["CL_de"]

    # estimate alpha0 from initial theta (gamma≈0 initially)
    alpha0 = np.radians(d["theta0_deg"])  # rad
    # aspect ratio & induced drag factor (assume e≈0.8)
    AR = b*b / S
    k_ind = 1.0/(np.pi*0.8*AR)
    # set CL0 so CL(alpha0) ≈ CL1; and CD0 so CD(CL1) ≈ CD1
    CL0 = CL1 - CLa*alpha0
    CD0 = max(1e-3, CD1 - k_ind*(CL1**2))

    # lateral-directional maps
    aero = dict(
        CL0=CL0, CLalpha=CLa, CLq=CLq, CLde=CL_de,
        Cm0=0.0, Cmalpha=Cm_a, Cmq=Cmq, Cmde=abs(Cm_de),  # GUI convention: positive elevator → positive pitch rate
        CD0=CD0, k_ind=k_ind,
        CYbeta=d.get("CY_beta", 0.0), CYdr=d.get("CY_dr", 0.0),
        Clbeta=d.get("Cl_beta", 0.0), Clp=d.get("Cl_p", 0.0), Clr=d.get("Cl_r", 0.0), Clda=d.get("Cl_da", 0.0),
        Cnbeta=d.get("Cn_beta", 0.0), Cnp=d.get("Cn_p", 0.0), Cnr=d.get("Cn_r", 0.0), Cnda=d.get("Cn_da", 0.0), Cndr=d.get("Cn_dr", 0.0),
    )

    # thrust: scale so 50% throttle gives trim thrust at start condition
    U0 = d["U_fps"] * FPS2MPS
    h0 = d["alt_ft"] * FT2M
    rho0 = isa_rho(h0)
    qbar0 = 0.5*rho0*U0*U0
    T_trim = qbar0 * S * d["CTx1"]
    T_max = 2.0 * T_trim  # 50% throttle → T_trim

    params = dict(m=m, I=I, S=S, b=b, cbar=cbar, T_max=T_max)

    init = dict(
        u0=U0,
        h0=h0,
        theta0_rad=np.radians(d["theta0_deg"]),
    )
    return params, aero, init

# Build default (Approach)
PARAMS, AERO, INIT0 = _build_params_from_md("Approach")

def euler_rates(phi, theta, p, q, r):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    tth = np.tan(theta)
    phidot   = p + tth*(q*sphi + r*cphi)
    thetadot = q*cphi - r*sphi
    psidot   = (q*sphi + r*cphi)/cth
    return phidot, thetadot, psidot

def forces_moments(Vb, p, q, r, u_cmd, qbar, params, aero):
    """
    Vb: body air-relative velocity [u,v,w] (m/s)
    u_cmd: [da, de, dr, throttle] (rad, rad, rad, 0..1)
    Returns X,Y,Z (N) and L,M,N (N·m) in BODY axes.
    """
    u,v,w = Vb
    V = np.linalg.norm(Vb) + 1e-6
    alpha = np.arctan2(w, u)
    beta  = np.arcsin(np.clip(v / V, -1.0, 1.0))
    da,de,dr,thr = u_cmd

    S,b,c = params["S"], params["b"], params["cbar"]
    pb2V, qb2V, rb2V = p*b/(2*V), q*c/(2*V), r*b/(2*V)

    # Coefficients
    CL = aero["CL0"] + aero["CLalpha"]*alpha + aero["CLq"]*qb2V + aero["CLde"]*de
    CD = aero["CD0"] + aero["k_ind"]*CL*CL
    Cm = aero["Cm0"] + aero["Cmalpha"]*alpha + aero["Cmq"]*qb2V + aero["Cmde"]*de

    CY = aero["CYbeta"]*beta + aero["CYdr"]*dr
    Cl = aero["Clbeta"]*beta + aero["Clp"]*pb2V + aero["Clr"]*rb2V + aero["Clda"]*da
    Cn = aero["Cnbeta"]*beta + aero["Cnp"]*pb2V + aero["Cnr"]*rb2V + aero["Cndr"]*dr

    # Wind-axis forces: Xw = -D, Yw = Y, Zw = -L  (Zw positive DOWN)
    L = CL * qbar * S
    D = CD * qbar * S
    Y = CY * qbar * S
    Fw = np.array([-D, Y, -L])

    # Convert wind→body: R_w2b = R_y(-alpha) @ R_z(-beta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    Ry = np.array([[ ca, 0, -sa],
                   [  0, 1,   0],
                   [ sa, 0,  ca]])
    Rz = np.array([[ cb, sb, 0],
                   [-sb, cb, 0],
                   [  0,  0, 1]])
    Fb = Ry @ Rz @ Fw  # body forces (aero)

    # Moments in body axes
    Lb = Cl * qbar * S * b
    Mb = Cm * qbar * S * c
    Nb = Cn * qbar * S * b

    # Thrust (aligned +x_body)
    T = params["T_max"] * np.clip(thr, 0.0, 1.0)
    Fb += np.array([T, 0.0, 0.0])

    return Fb[0], Fb[1], Fb[2], Lb, Mb, Nb

def sixdof_rhs(x, t, u_cmd, params=PARAMS, aero=AERO, wind_ned=np.zeros(3)):
    # State: [u,v,w,p,q,r, phi,theta,psi, N,E,D]
    u,v,w,p,q,r,phi,theta,psi,N,E,D = x
    m = params["m"]; Ixx,Iyy,Izz = params["I"]

    # Air-relative velocity in body (ignore wind for now)
    V_air_b = np.array([u,v,w])

    h = -D
    rho = isa_rho(h)
    V = np.linalg.norm(V_air_b)
    qbar = 0.5*rho*V*V

    X,Y,Z,L,M,Nm = forces_moments(V_air_b, p,q,r, u_cmd, qbar, params, aero)

    # Translational dynamics
    udot = r*v - q*w + X/m - g*np.sin(theta)
    vdot = p*w - r*u + Y/m + g*np.sin(phi)*np.cos(theta)
    wdot = q*u - p*v + Z/m + g*np.cos(phi)*np.cos(theta)

    # Rotational dynamics (principal inertia)
    pdot = (L + (Iyy - Izz)*q*r)/Ixx
    qdot = (M + (Izz - Ixx)*p*r)/Iyy
    rdot = (Nm + (Ixx - Iyy)*p*q)/Izz

    # Attitude kinematics
    phidot, thetadot, psidot = euler_rates(phi, theta, p, q, r)

    # Position kinematics: [Ndot,Edot,Ddot] = R_b2n * [u,v,w]
    c,s = np.cos, np.sin
    cph, sph = c(phi), s(phi)
    cps, sps = c(psi), s(psi)
    cth = c(theta)
    sth = -s(theta)   # flip sign so negative pitch angle (nose-down) → positive Down velocity
    Rb2n = np.array([
        [ cth*cps,                cth*sps,               -sth ],
        [ sph*sth*cps - cph*sps,  sph*sth*sps + cph*cps, sph*cth ],
        [ cph*sth*cps + sph*sps,  cph*sth*sps - sph*cps, cph*cth ],
    ])
    ned_dot = Rb2n @ np.array([u, v, w])
    # Invert heading's effect on East: flip the E component sign
    Ndot, Edot, Ddot = ned_dot[0], -ned_dot[1], ned_dot[2]

    return np.array([udot,vdot,wdot, pdot,qdot,rdot,
                     phidot,thetadot,psidot, Ndot,Edot,Ddot])
# ─────────────────────────────────────────────────────────────────────────


class FlightSimulatorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("747 Flight Dynamics Simulator")
        self.root.configure(bg="#f0f0f0")

        # time bookkeeping
        self.dt, self.t_max, self.t = 0.1, 60.0, [0.0]

        # 12-state: [u,v,w,p,q,r, phi,theta,psi, N,E,D]
        u0 = INIT0['u0']
        h0 = INIT0['h0']
        th0 = INIT0['theta0_rad']
        self.x = np.array([[u0, 0.0, 0.0,  0.0,0.0,0.0,  0.0,th0,0.0,  0.0,0.0,-h0]]).T

        # inputs (δa, δe, δr, throttle)
        self.u = np.array([0.0, 0.0, 0.0, 0.5])

        # history buffers
        self.hist = {k: [0.0] for k in (
            "u","v","w",
            "p","q","r",
            "phi","theta","psi",
            "N","E","D",
            "V","h",
            "delta_a","delta_e","delta_r","throttle"
        )}

        self.nav_window_m = 80.0
        self.running, self.max_pts = False, int(self.t_max / self.dt)

        self._create_control_panel()
        self._create_plots()
        Thread(target=self._simulate, daemon=True).start()

    # ───────────────────────── GUI CONTROLS ────────────────────────────────
    def _on_fc_change(self, name):
        self._apply_condition(name)

    def _apply_condition(self, name):
        global PARAMS, AERO, INIT0
        PARAMS, AERO, INIT0 = _build_params_from_md(name)
        # reset state and histories to new condition
        u0 = INIT0['u0']; h0 = INIT0['h0']; th0 = INIT0['theta0_rad']
        self.x = np.array([[u0,0,0, 0,0,0, 0,th0,0, 0,0,-h0]]).T
        self.t = [0.0]
        # histories
        self.hist['u']=[u0]; self.hist['v']=[0.0]; self.hist['w']=[0.0]
        self.hist['p']=[0.0]; self.hist['q']=[0.0]; self.hist['r']=[0.0]
        self.hist['phi']=[0.0]; self.hist['theta']=[th0]; self.hist['psi']=[0.0]
        self.hist['N']=[0.0]; self.hist['E']=[0.0]; self.hist['D']=[-h0]
        self.hist['V']=[u0];  self.hist['h']=[h0]
        self.hist['delta_a']=[0.0]; self.hist['delta_e']=[0.0]; self.hist['delta_r']=[0.0]
        self.hist['throttle']=[50.0]
        # reset controls
        try:
            for sld in (self.aileron, self.elevator, self.rudder): sld.set(0)
            self.u[3] = 0.5
            self.throttle.set(50)
        except Exception:
            pass
        # rebuild 3D geometry scale (depends on b)
        try:
            for s in getattr(self, 'surfs', []):
                s.remove()
        except Exception:
            pass
        self.surfs = []
        self.base_coords = []
        self.ctrl_idx = {}
        try:
            self._setup_plane_model()
        except Exception:
            pass
        self._update_plots()

    # ───────────────────────── GUI CONTROLS ────────────────────────────────
    def _create_control_panel(self):
        lf, bf = ("Arial", 12), ("Arial", 12)
        top = tk.Frame(self.root, bg="#f0f0f0"); top.pack(fill="x", pady=2)

        # two columns next to each other
        left  = tk.Frame(top, bg="#f0f0f0"); left.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        right = tk.Frame(top, bg="#f0f0f0"); right.grid(row=0, column=1, sticky="nsew", padx=(5,0))
        top.columnconfigure(0, weight=1); top.columnconfigure(1, weight=1)

        def slider(parent, row, txt, vmin, vmax, res):
            tk.Label(parent, text=txt, font=lf, bg="#f0f0f0") \
                .grid(row=row, column=0, padx=5, pady=2, sticky="w")
            s = tk.Scale(parent, from_=vmin, to=vmax, resolution=res,
                         orient=tk.HORIZONTAL, length=300,
                         bg="#f0f0f0", highlightthickness=0)
            s.grid(row=row, column=1, padx=5, pady=2, sticky="ew")
            parent.columnconfigure(1, weight=1)
            return s

        # LEFT: Aileron, Elevator, Rudder
        self.aileron  = slider(left,  0, "Aileron (δ_a) [deg]",  -10, 10, 0.1)
        self.elevator = slider(left,  1, "Elevator (δ_e) [deg]", -10, 10, 0.1)
        self.rudder   = slider(left,  2, "Rudder (δ_r) [deg]",   -5, 5, 0.1)

        # RIGHT: Flight Condition, Throttle and Tracking window
        tk.Label(right, text="Flight condition", font=lf, bg="#f0f0f0") \
            .grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.fc_var = tk.StringVar(value="Approach")
        self.fc_menu = tk.OptionMenu(right, self.fc_var, "Approach", "CruiseHigh", "CruiseLow", command=lambda v: self._on_fc_change(v))
        self.fc_menu.configure(bg="#f0f0f0")
        self.fc_menu.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(right, text="Throttle [%]", font=lf, bg="#f0f0f0") \
            .grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.throttle = tk.Scale(right, from_=0, to=100, resolution=1,
                                 orient=tk.HORIZONTAL, length=300,
                                 bg="#f0f0f0", highlightthickness=0)
        self.u[3] = 0.5
        self.throttle.set(50)
        self.throttle.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        right.columnconfigure(1, weight=1)

        tk.Label(right, text="Tracking window [m] (half-size)", font=lf, bg="#f0f0f0") \
            .grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.navwin = tk.Scale(right, from_=40, to=200, resolution=10,
                               orient=tk.HORIZONTAL, length=300,
                               bg="#f0f0f0", highlightthickness=0,
                               command=lambda v: setattr(self, 'nav_window_m', float(v)))
        self.navwin.set(int(self.nav_window_m))
        self.navwin.grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # Buttons on the RIGHT
        btn = tk.Frame(right, bg="#f0f0f0"); btn.grid(row=3, column=0, columnspan=2, pady=2, sticky="e")
        self.start_b = tk.Button(btn, text="Start Simulation", command=self._start_sim,
                                 font=bf, padx=10, pady=5, bg="#4CAF50", fg="white")
        self.stop_b  = tk.Button(btn, text="Stop Simulation",  command=self._stop_sim,
                                 font=bf, padx=10, pady=5, bg="#F44336", fg="white",
                                 state="disabled")
        self.reset_b = tk.Button(btn, text="Reset", command=self._reset_sim,
                                 font=bf, padx=10, pady=5, bg="#2196F3", fg="white")
        for b in (self.start_b, self.stop_b, self.reset_b):
            b.pack(side=tk.LEFT, padx=5)

    # ───────────────────────── PLOTS ──────────────────────────────────────
    def _create_plots(self):
        self.fig = plt.figure(figsize=(20, 20))
        gs = self.fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1],
                                   width_ratios=[1, 1], hspace=0.4)

        # Left column: 4 time-series plots
        self.ax1, self.ax2, self.ax3, self.ax4 = (self.fig.add_subplot(gs[i,0]) for i in range(4))

        # Right column: 3D view spanning all rows
        self.ax3d = self.fig.add_subplot(gs[:,1], projection='3d')

        # Plot 1: Translational body velocities u, v, w
        self.l_u, = self.ax1.plot([], [], color='k',            label='Forward (u) [m/s]', lw=2)
        self.l_v, = self.ax1.plot([], [], color='m',            label='Right (v) [m/s]',   lw=2)
        self.l_w, = self.ax1.plot([], [], color='tab:orange',   label='Down (w) [m/s]',    lw=2)
        self.ax1.set(title='Body Velocities', ylabel='[m/s]', xlim=(0, self.t_max))
        self.ax1.grid(True); self.ax1.legend(fontsize=8)

        # Plot 2: Body Rates (deg/s)
        self.l_p,  = self.ax2.plot([], [], color='b', label='Roll rate (p) [deg/s]',  lw=2)
        self.l_q,  = self.ax2.plot([], [], color='g', label='Pitch rate (q) [deg/s]', lw=2)
        self.l_r,  = self.ax2.plot([], [], color='r', label='Yaw rate (r) [deg/s]',   lw=2)
        self.ax2.set(title='Body Rates', ylabel='[deg/s]', xlim=(0, self.t_max))
        self.ax2.grid(True); self.ax2.legend(fontsize=8)

        # Plot 3: Euler Angles (deg)
        self.l_phi,   = self.ax3.plot([], [], color='b', label='Roll angle (φ) [deg]',    lw=2)
        self.l_theta, = self.ax3.plot([], [], color='g', label='Pitch angle (θ) [deg]',   lw=2)
        self.l_psi,   = self.ax3.plot([], [], color='r', label='Heading (ψ) [deg]',       lw=2)
        self.ax3.set(title='Euler Angles', ylabel='[deg]', xlim=(0, self.t_max))
        self.ax3.grid(True); self.ax3.legend(fontsize=8)

        # Plot 4: Position N, E, h (Altitude h = -D)
        self.l_N, = self.ax4.plot([], [], color='tab:olive', label='North (N) [m]',    lw=2)
        self.l_E, = self.ax4.plot([], [], color='tab:cyan',  label='East (E) [m]',     lw=2)
        self.l_h, = self.ax4.plot([], [], color='tab:gray',  label='Altitude (h) [m]', lw=2)
        self.ax4.set(title='Position (NEh)', xlabel='Time [s]', ylabel='[m]', xlim=(0, self.t_max))
        self.ax4.grid(True); self.ax4.legend(fontsize=8)

        # 3D view setup
        self._setup_plane_model()
        self.nav_window_m = getattr(self, 'nav_window_m', 80.0)  # initial view window half-size in meters
        self.ax3d.set(title='747 Position & Orientation (N, E, h)', xlabel='North [m]', ylabel='East [m]')
        self.ax3d.set_zlabel('Altitude h [m]')
        w = self.nav_window_m
        self.ax3d.set(xlim=(-w, w), ylim=(w, -w), zlim=(0, 2*w))
        try:
            self.ax3d.set_box_aspect([1,1,1])
        except Exception:
            pass
        self.ax3d.grid(True, alpha=0.3)
        self.ax3d.view_init(elev=20, azim=45)
        self.ax3d.set_facecolor('#f5f5f5')
        self.ax3d.dist = 7
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

    # ───────────── GEOMETRY ─────────────────────────────
    def _setup_plane_model(self):
        # Scale geometry so 1.6*s ≈ wing span (b), i.e., s = b/1.6 → geometry units are meters
        s = PARAMS['b']/1.6
        sign = -1.0
        self.base_coords, self.surfs, self.ctrl_idx = [], [], {}

        def add(X, Y, Z, color, alpha, name=None):
            surf = self.ax3d.plot_surface(X, Y, Z, color=color,
                                          alpha=alpha, linewidth=0)
            self.surfs.append(surf); self.base_coords.append((X, Y, Z))
            if name: self.ctrl_idx[name] = len(self.surfs) - 1

        # fuselage
        th = np.linspace(0, 2*np.pi, 20); z_lin = np.linspace(-1, 1, 20)*s
        TH, Z = np.meshgrid(th, z_lin); r = 0.05*s
        add(sign*Z, sign*r*np.cos(TH), r*np.sin(TH), 'silver', 0.8)

        # wings
        span, sweep, dihedral = .8*s, .3*s, .1*s
        Xw = np.array([[0, sweep]]*4) - .2*s
        Yw = np.array([[-span, -span], [span, span]]*2)
        Zw = np.array([[0, dihedral], [0, dihedral],
                       [dihedral, dihedral], [0, 0]])
        add(sign*Xw, Yw, Zw, 'gray', 0.7)

        # horizontal tail
        hspan = .3*s
        Xh = np.array([[0.8, 0.9]]*4)*s
        Yh = np.array([[-hspan, -hspan], [hspan, hspan]]*2)
        Zh = np.zeros((4,2))
        add(sign*Xh, Yh, Zh, 'gray', 0.7)

        # vertical tail
        vht = .3*s
        Xv = np.array([[0.8, 0.9]]*4)*s
        Yv = np.zeros((4,2))
        Zv = np.array([[0,0],[0,0],[vht,vht],[vht,vht]])
        add(sign*Xv, Yv, Zv, 'gray', 0.7)

        # control-surface helper
        def plate(x_le, chord, y0, y1, z_off=0):
            X = np.array([[x_le, x_le+chord]]*2)
            Y = np.array([[y0, y0], [y1, y1]])
            return X, Y, np.full_like(X, z_off)

        # ailerons (blue) – hinge about y-axis (trailing-edge up/down)
        chord_a, x_a = .12*s, -0.1*s + sweep
        y_out, y_in = span, .5*span
        add(*plate(sign*x_a, chord_a, -y_out, -y_in, dihedral), 'blue', 0.9, 'ail_L')
        add(*plate(sign*x_a, chord_a,  y_in,  y_out, dihedral), 'blue', 0.9, 'ail_R')

        # elevators (green) – hinge about y-axis (pitch)
        chord_e, x_e = .12*s, .99*s
        ye_out, ye_in = hspan, .1*hspan
        add(*plate(sign*x_e, chord_e, -ye_out, -ye_in), 'green', 0.9, 'elev_L')
        add(*plate(sign*x_e, chord_e,  ye_in,  ye_out), 'green', 0.9, 'elev_R')

        # rudder (red)
        chord_r, x_r = .12*s, .88*s
        z_top = vht
        Xr = np.array([[x_r, x_r+chord_r]]*2)
        Yr = np.zeros_like(Xr)
        Zr = np.array([[0,0],[z_top,z_top]])
        add(sign*Xr, Yr, Zr, 'red', 0.9, 'rudder')

    # ───────────── BODY & DEFLECTION TRANSFORMS ───────────────────────────
    @staticmethod
    def _R_body(phi, th, ps):
        cφ, sφ, cθ, sθ, cψ, sψ = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(ps), np.sin(ps)
        Rr = np.array([[1,0,0],[0,cφ,-sφ],[0,sφ,cφ]])
        Rp = np.array([[cθ,0,sθ],[0,1,0],[-sθ,0,cθ]])
        Ry = np.array([[cψ,-sψ,0],[sψ,cψ,0],[0,0,1]])
        return Ry @ Rp @ Rr

    @staticmethod
    def _rotate_pts(pts, angle, axis):
        if angle == 0.0: return pts
        c, s = np.cos(angle), np.sin(angle)
        cx, cy, cz = pts.mean(1, keepdims=True)
        shifted = pts - np.vstack((cx, cy, cz))
        if axis == 'x':  R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 'y':R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:             R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        return R @ shifted + np.vstack((cx, cy, cz))

    def _update_plane_model(self, phi, θ, ψ, N, E, h, da, de, dr):
        # Use -θ so a negative pitch angle (nose-down) visually tilts the nose downward
        Rb = self._R_body(-phi, -θ, ψ)
        defl = {'ail_L': -da, 'ail_R': da, 'elev_L': de,
                'elev_R': de, 'rudder': dr}
        axis = {'ail_L': 'y', 'ail_R': 'y',
                'elev_L': 'y', 'elev_R': 'y',
                'rudder': 'z'}

        for surf in self.surfs: surf.remove()
        self.surfs = []

        for i, (X0, Y0, Z0) in enumerate(self.base_coords):
            pts = np.vstack((X0.flatten(), Y0.flatten(), Z0.flatten()))
            name = next((n for n, j in self.ctrl_idx.items() if j == i), None)
            if name: pts = self._rotate_pts(pts, defl[name], axis[name])
            pts = Rb @ pts  # body attitude
            # translate to position (N,E,h) in meters
            pts = pts + np.array([[N],[E],[h]])

            Xr, Yr, Zr = (pts[k].reshape(X0.shape) for k in range(3))
            col = {'ail_L':'blue','ail_R':'blue',
                   'elev_L':'green','elev_R':'green',
                   'rudder':'red'}.get(name, 'silver' if i==0 else 'gray')
            alpha = 0.9 if name else 0.8 if i==0 else 0.7
            self.surfs.append(
                self.ax3d.plot_surface(Xr, Yr, Zr, color=col, alpha=alpha, linewidth=0)
            )

    # ───────────── SIMULATION LOOP ─────────────────────────────────────────
    def _simulate(self):
        while True:
            if self.running and self.t[-1] < self.t_max:
                da = np.radians(self.aileron.get())
                de = np.radians(self.elevator.get())
                dr = np.radians(self.rudder.get())
                thr = 0.01 * self.throttle.get()
                self.u = np.array([da, de, dr, thr])

                sol = odeint(sixdof_rhs, self.x[:, -1],
                             [self.t[-1], self.t[-1]+self.dt], args=(self.u,))
                self.x = np.hstack((self.x, sol[-1, :].reshape(-1,1)))
                self.t.append(self.t[-1]+self.dt)

                u_b, v_b, w_b = self.x[0:3, -1]
                p, q, r       = self.x[3:6, -1]
                phi, th, ps   = self.x[6:9, -1]
                N, E          = self.x[9:11, -1]
                D             = self.x[11, -1]
                V = float(np.linalg.norm([u_b, v_b, w_b]))
                h = float(-D)

                # histories
                self.hist['u'].append(u_b); self.hist['v'].append(v_b); self.hist['w'].append(w_b)
                self.hist['p'].append(p);   self.hist['q'].append(q);   self.hist['r'].append(r)
                self.hist['phi'].append(phi); self.hist['theta'].append(th); self.hist['psi'].append(ps)
                self.hist['N'].append(N);  self.hist['E'].append(E);    self.hist['D'].append(D)
                self.hist['V'].append(V);  self.hist['h'].append(h)
                for k,v in zip(('delta_a','delta_e','delta_r'), (da,de,dr)):
                    self.hist[k].append(np.degrees(v))
                self.hist['throttle'].append(100.0*thr)

                self._update_plots()
            time.sleep(self.dt/2)

    # ───────────── PLOT/3-D UPDATE ────────────────────────────────────────
    def _update_plots(self):
        s = max(0, len(self.t)-self.max_pts)

        # Plot 1: u, v, w
        self.l_u.set_data(self.t[s:], self.hist['u'][s:])
        self.l_v.set_data(self.t[s:], self.hist['v'][s:])
        self.l_w.set_data(self.t[s:], self.hist['w'][s:])
        self.ax1.relim(); self.ax1.autoscale_view()

        # Plot 2: p, q, r (deg/s)
        self.l_p.set_data(self.t[s:], np.degrees(self.hist['p'][s:]))
        self.l_q.set_data(self.t[s:], np.degrees(self.hist['q'][s:]))
        self.l_r.set_data(self.t[s:], np.degrees(self.hist['r'][s:]))
        self.ax2.relim(); self.ax2.autoscale_view()

        # Plot 3: φ, θ, ψ (deg)
        self.l_phi  .set_data(self.t[s:], np.degrees(self.hist['phi'  ][s:]))
        self.l_theta.set_data(self.t[s:], np.degrees(self.hist['theta'][s:]))
        self.l_psi  .set_data(self.t[s:], np.degrees(self.hist['psi'  ][s:]))
        self.ax3.relim(); self.ax3.autoscale_view()

        # Plot 4: N, E, h
        self.l_N.set_data(self.t[s:], self.hist['N'][s:])
        self.l_E.set_data(self.t[s:], self.hist['E'][s:])
        self.l_h.set_data(self.t[s:], self.hist['h'][s:])
        self.ax4.relim(); self.ax4.autoscale_view()

        # Update 3D model
        self._update_plane_model(self.hist['phi'][-1],
                                 self.hist['theta'][-1],
                                 self.hist['psi'][-1],
                                 self.hist['N'][-1],
                                 self.hist['E'][-1],
                                 self.hist['h'][-1],
                                 np.radians(self.hist['delta_a'][-1]),
                                 np.radians(self.hist['delta_e'][-1]),
                                 np.radians(self.hist['delta_r'][-1]))
        # Center the 3D view window around current N,E,h
        try:
            w = getattr(self, 'nav_window_m', 100.0)
            N0, E0, h0 = self.hist['N'][-1], self.hist['E'][-1], self.hist['h'][-1]
            self.ax3d.set(xlim=(N0-w, N0+w), ylim=(E0+w, E0-w), zlim=(max(0.0, h0-w), h0+w))
            self.ax3d.set_box_aspect([1,1,1])
        except Exception:
            pass
        self.canvas.draw_idle(); self.canvas.flush_events()

    # ───────────── BUTTONS ────────────────────────────────────────────────
    def _start_sim(self): self.running=True;  self.start_b.config(state='disabled'); self.stop_b.config(state='normal')
    def _stop_sim(self):  self.running=False; self.start_b.config(state='normal');  self.stop_b.config(state='disabled')
    def _reset_sim(self):
        # Reset to the current flight condition's start parameters
        self.running = False
        try:
            mode = self.fc_var.get()
        except Exception:
            mode = "Approach"
        self._apply_condition(mode)


# ───────────── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    FlightSimulatorGUI(root)
    root.mainloop()
