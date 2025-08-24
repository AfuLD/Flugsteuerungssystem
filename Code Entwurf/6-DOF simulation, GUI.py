"""
747 6‑DOF Flight Dynamics — MVC-style refactor

- SixDOFSimulator: physics, parameters, ODE, state/history, stepping
- FlightSimulatorGUI: tkinter UI + matplotlib plots/3D model

Run: python this_file.py
"""

from __future__ import annotations

import time
import tkinter as tk
from threading import Thread
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────────────────────────────────
# Constants & helpers
# ──────────────────────────────────────────────────────────────────────────

g = 9.80665  # m/s²

FT2M = 0.3048
FT2_TO_M2 = 0.09290304
FPS2MPS = 0.3048
SLUGFT2_TO_KGM2 = 1.3558179483314004
LBF_TO_N = 4.4482216152605


def isa_rho(h_m: float) -> float:
    """
    Very simple ISA density model:
      ρ ≈ 1.225 * (1 - 2.25577e-5*h)^4.256   for h in meters (truncate h < 0)
    """
    h = max(0.0, float(h_m))
    return 1.225 * (1 - 2.25577e-5 * h) ** 4.256


# ——— Parameters from “747 parameters.md” (converted to SI) ———
_MD: Dict[str, Dict[str, float]] = {
    "Approach": {
        "alt_ft": 3280.839895013123,  # ≈1000 m
        "U_fps": 221.0,
        "theta0_deg": 8.5,
        "S_ft2": 5500.0,
        "b_ft": 196.0,
        "c_ft": 27.3,
        "W_lbf": 564000.0,
        "Ixx_slugft2": 13.7e6,
        "Iyy_slugft2": 30.5e6,
        "Izz_slugft2": 43.1e6,
        "CL1": 1.76,
        "CD1": 0.263,
        "CTx1": 0.263,
        "Cm_u": 0.071,
        "Cm_alpha": -1.45,
        "Cm_adot": -3.3,
        "Cm_q": -21.4,
        "CL_u": -0.22,
        "CL_alpha": 5.67,
        "CL_adot": 6.7,
        "CL_q": 5.65,
        "CD_u": 1.13,
        "CD_alpha": 0.0,
        "CL_de": 0.36,
        "CD_de": 0.0,
        "Cm_de": -1.40,
        "Cl_beta": -0.281,
        "Cl_p": -0.502,
        "Cl_r": 0.195,
        "Cl_da": 0.053,
        "Cl_dr": 0.0,
        "Cn_beta": 0.184,
        "Cn_p": -0.222,
        "Cn_r": -0.36,
        "Cn_da": 0.0083,
        "Cn_dr": -0.113,
        "CY_beta": -1.08,
        "CY_dr": 0.179,
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
        "CL1": 0.52,
        "CD1": 0.045,
        "CTx1": 0.045,
        "Cm_u": -0.09,
        "Cm_alpha": -1.60,
        "Cm_adot": -9.0,
        "Cm_q": -25.5,
        "CL_u": -0.23,
        "CL_alpha": 5.5,
        "CL_adot": 8.0,
        "CL_q": 7.8,
        "CD_u": 0.50,
        "CD_alpha": 0.22,
        "CL_de": 0.30,
        "CD_de": 0.0,
        "Cm_de": -1.20,
        "Cl_beta": -0.095,
        "Cl_p": -0.320,
        "Cl_r": 0.200,
        "Cl_da": 0.014,
        "Cl_dr": 0.005,
        "Cn_beta": 0.210,
        "Cn_p": 0.020,
        "Cn_r": -0.33,
        "Cn_da": -0.0028,
        "Cn_dr": -0.095,
        "CY_beta": -0.90,
        "CY_dr": 0.060,
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
        "CL1": 0.40,
        "CD1": 0.025,
        "CTx1": 0.025,
        "Cm_u": 0.013,
        "Cm_alpha": -1.00,
        "Cm_adot": -4.0,
        "Cm_q": -20.5,
        "CL_u": 0.13,
        "CL_alpha": 4.4,
        "CL_adot": 7.0,
        "CL_q": 6.6,
        "CD_u": 0.20,
        "CD_alpha": 0.0,
        "CL_de": 0.32,
        "CD_de": 0.0,
        "Cm_de": -1.30,
        "Cl_beta": -0.160,
        "Cl_p": -0.340,
        "Cl_r": 0.130,
        "Cl_da": 0.013,
        "Cl_dr": 0.008,
        "Cn_beta": 0.160,
        "Cn_p": -0.026,
        "Cn_r": -0.28,
        "Cn_da": 0.0018,
        "Cn_dr": -0.100,
        "CY_beta": -0.90,
        "CY_dr": 0.120,
    },
}


def _build_params_from_md(name: str) -> Tuple[Dict, Dict, Dict]:
    """Create (params, aero, init) dictionaries from the raw MD block."""
    d = _MD[name]

    # Geometry & masses
    S = d["S_ft2"] * FT2_TO_M2
    b = d["b_ft"] * FT2M
    c_mac = d["c_ft"] * FT2M  # mean aerodynamic chord
    W_N = d["W_lbf"] * LBF_TO_N
    m = W_N / g
    I = (
        d["Ixx_slugft2"] * SLUGFT2_TO_KGM2,
        d["Iyy_slugft2"] * SLUGFT2_TO_KGM2,
        d["Izz_slugft2"] * SLUGFT2_TO_KGM2,
    )

    # Aero from MD
    CL1, CD1 = d["CL1"], d["CD1"]
    CLa, CLq = d["CL_alpha"], d["CL_q"]
    Cm_a, Cmq, Cm_de = d["Cm_alpha"], d["Cm_q"], d["Cm_de"]
    CL_de = d["CL_de"]

    # Estimate alpha0 from initial theta (assume γ≈0 initially)
    alpha0 = np.radians(d["theta0_deg"])
    # Aspect ratio & induced drag factor (assume e≈0.8)
    AR = b * b / S
    k_ind = 1.0 / (np.pi * 0.8 * AR)

    # Trim force balance at specified condition
    U0 = d["U_fps"] * FPS2MPS
    h0 = d["alt_ft"] * FT2M
    rho0 = isa_rho(h0)
    qbar0 = 0.5 * rho0 * U0 * U0

    # Set CL so that L=W at t=0 (level trim)
    CLtrim = W_N / (qbar0 * S)
    CL0 = CLtrim - CLa * alpha0

    # Set CD so that D = T_trim at t=0; use CTx1 as reference
    CD0 = max(1e-3, d["CTx1"] - k_ind * (CLtrim**2))
    Cm0 = -Cm_a * alpha0

    # Lateral-directional maps
    aero = dict(
        CL0=CL0,
        CLalpha=CLa,
        CLq=CLq,
        CLde=CL_de,
        CLu=d.get("CL_u", 0.0),
        Cm0=Cm0,
        Cmalpha=Cm_a,
        Cmq=Cmq,
        Cmde=abs(Cm_de),
        Cmu=d.get("Cm_u", 0.0),
        CD0=CD0,
        k_ind=k_ind,
        CDu=d.get("CD_u", 0.0),
        CYbeta=d.get("CY_beta", 0.0),
        CYdr=d.get("CY_dr", 0.0),
        Clbeta=d.get("Cl_beta", 0.0),
        Clp=d.get("Cl_p", 0.0),
        Clr=d.get("Cl_r", 0.0),
        Clda=d.get("Cl_da", 0.0),
        Cnbeta=d.get("Cn_beta", 0.0),
        Cnp=d.get("Cn_p", 0.0),
        Cnr=d.get("Cn_r", 0.0),
        Cnda=d.get("Cn_da", 0.0),
        Cndr=d.get("Cn_dr", 0.0),
    )

    # Thrust: scale so 50% throttle gives trim thrust at start condition
    T_trim = qbar0 * S * d["CTx1"]
    T_max = 2.0 * T_trim  # 50% throttle → T_trim

    # Velocity components consistent with alpha0
    u0_b = U0 * np.cos(alpha0)
    w0_b = U0 * np.sin(alpha0)

    params = dict(
        m=m,
        I=I,
        S=S,
        b=b,
        c_mac=c_mac,
        T_max=T_max,
        V=U0,
        u_trim=u0_b,
        alpha0=alpha0,
        CT_ref=d["CTx1"],
    )

    init = dict(
        u0=u0_b,
        w0=w0_b,
        h0=h0,
        theta0_rad=np.radians(d["theta0_deg"]),
    )
    return params, aero, init


# ──────────────────────────────────────────────────────────────────────────
# Simulation (MODEL)
# ──────────────────────────────────────────────────────────────────────────

class SixDOFSimulator:
    """
    Minimal rigid-body 6‑DOF simulator with Euler angles.

    State x: [u,v,w,p,q,r, phi,theta,psi, N,E,D]  (D positive down)
    Controls u: [δa, δe, δr, throttle]  (rad, rad, rad, 0..1)
    """

    def __init__(self, condition: str = "Approach", dt: float = 0.10, t_max: float = 120.0):
        self.dt = float(dt)
        self.t_max = float(t_max)
        self.params, self.aero, self.init = _build_params_from_md(condition)
        self.condition = condition

        # initial state
        u0 = self.init["u0"]
        w0 = self.init["w0"]
        h0 = self.init["h0"]
        th0 = self.init["theta0_rad"]
        self.x = np.array([[u0, 0.0, w0, 0.0, 0.0, 0.0, 0.0, th0, 0.0, 0.0, 0.0, -h0]]).T
        self.t_hist = [0.0]

        # controls
        self.u_cmd = np.array([0.0, 0.0, 0.0, 0.5])

        # history buffers
        self.hist: Dict[str, list] = {
            k: [0.0]
            for k in (
                "u",
                "v",
                "w",
                "p",
                "q",
                "r",
                "phi",
                "theta",
                "psi",
                "N",
                "E",
                "D",
                "V",
                "h",
                "delta_a",
                "delta_e",
                "delta_r",
                "throttle",
            )
        }
        self._seed_hist()

    # ——— public API ————————————————————————————————————————————————

    def set_controls(self, da_rad: float, de_rad: float, dr_rad: float, throttle_01: float) -> None:
        self.u_cmd = np.array([da_rad, de_rad, dr_rad, float(np.clip(throttle_01, 0.0, 1.0))])

    def step(self) -> None:
        """Advance one fixed time step using Radau."""
        t0 = self.t_hist[-1]
        x0 = self.x[:, -1]

        sol = solve_ivp(
            lambda t, y: self._rhs(y, t),
            (t0, t0 + self.dt),
            x0,
            method="RK23",
            rtol=1e-6,
            atol=1e-8,
            max_step=self.dt,
        )
        x1 = sol.y[:, -1]
        self.x = np.hstack((self.x, x1.reshape(-1, 1)))
        self.t_hist.append(t0 + self.dt)

        # book-keeping
        self._update_history()

    def reset(self, condition: str | None = None) -> None:
        """Reset to starting condition (optionally change condition)."""
        if condition is not None and condition != self.condition:
            self.params, self.aero, self.init = _build_params_from_md(condition)
            self.condition = condition

        # reset state and histories
        u0 = self.init["u0"]
        w0 = self.init["w0"]
        h0 = self.init["h0"]
        th0 = self.init["theta0_rad"]
        self.x = np.array([[u0, 0.0, w0, 0.0, 0.0, 0.0, 0.0, th0, 0.0, 0.0, 0.0, -h0]]).T
        self.t_hist = [0.0]
        self.u_cmd = np.array([0.0, 0.0, 0.0, 0.5])
        self.hist = {k: [0.0] for k in self.hist.keys()}
        self._seed_hist()

    # ——— getters ————————————————————————————————————————————————

    def latest(self) -> Dict[str, float]:
        """Convenience accessor for most recent values."""
        u_b, v_b, w_b = self.x[0:3, -1]
        p, q, r = self.x[3:6, -1]
        phi, th, ps = self.x[6:9, -1]
        N, E = self.x[9:11, -1]
        D = self.x[11, -1]
        V = float(np.linalg.norm([u_b, v_b, w_b]))
        h = float(-D)
        return dict(u=u_b, v=v_b, w=w_b, p=p, q=q, r=r, phi=phi, theta=th, psi=ps, N=N, E=E, h=h, V=V)

    # ——— dynamics ————————————————————————————————————————————————

    @staticmethod
    def _euler_rates(phi: float, theta: float, p: float, q: float, r: float) -> Tuple[float, float, float]:
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        tth = np.tan(theta)
        phidot = p + tth * (q * sphi + r * cphi)
        thetadot = q * cphi - r * sphi
        psidot = (q * sphi + r * cphi) / cth
        return phidot, thetadot, psidot

    def _forces_moments(self, Vb: np.ndarray, p: float, q: float, r: float) -> Tuple[float, float, float, float, float, float]:
        """
        Compute body forces/moments given body-air velocity Vb and body rates.
        """
        u, v, w = Vb
        V = np.linalg.norm(Vb) + 1e-6
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / V, -1.0, 1.0))
        da, de, dr, thr = self.u_cmd
        S, b, c = self.params["S"], self.params["b"], self.params["c_mac"]

        # nondimensional rates
        pb2V, qc2V, rb2V = p * b / (2 * V), q * c / (2 * V), r * b / (2 * V)

        # coefficients
        CL = self.aero["CL0"] + self.aero["CLalpha"] * alpha + self.aero["CLq"] * qc2V + self.aero["CLde"] * de
        du_nd = (u - self.params.get("u_trim", V * np.cos(self.params.get("alpha0", 0.0)))) / V
        CL += self.aero.get("CLu", 0.0) * du_nd
        CD = self.aero["CD0"] + self.aero["k_ind"] * CL * CL + self.aero.get("CDu", 0.0) * du_nd
        Cm = self.aero["Cm0"] + self.aero["Cmalpha"] * alpha + self.aero["Cmq"] * qc2V + self.aero["Cmde"] * de

        CY = self.aero["CYbeta"] * beta + self.aero["CYdr"] * dr
        Cl = self.aero["Clbeta"] * beta + self.aero["Clp"] * pb2V + self.aero["Clr"] * rb2V + self.aero["Clda"] * da
        Cn = self.aero["Cnbeta"] * beta + self.aero["Cnp"] * pb2V + self.aero["Cnr"] * rb2V + self.aero["Cndr"] * dr

        # dynamic pressure at current altitude/speed
        state = self.latest()
        rho = isa_rho(state["h"])
        qbar = 0.5 * rho * V * V

        # wind-axis forces -> body
        L = CL * qbar * S
        D = CD * qbar * S
        Y = CY * qbar * S
        Fw = np.array([-D, Y, -L])

        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        Ry = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
        Rz = np.array([[cb, sb, 0], [-sb, cb, 0], [0, 0, 1]])
        Fb = Ry @ Rz @ Fw

        # moments in body
        Lb = Cl * qbar * S * b
        Mb = Cm * qbar * S * c
        Nb = Cn * qbar * S * b

        # thrust along +x_body
        T = self.params["T_max"] * np.clip(thr, 0.0, 1.0)
        Fb += np.array([T, 0.0, 0.0])

        return Fb[0], Fb[1], Fb[2], Lb, Mb, Nb

    def _rhs(self, x: np.ndarray, t: float) -> np.ndarray:
        """6‑DOF rigid body equations with Euler angles and NED position."""
        u, v, w, p, q, r, phi, theta, psi, N, E, D = x
        m = self.params["m"]
        Ixx, Iyy, Izz = self.params["I"]

        V_air_b = np.array([u, v, w])
        h = -D

        X, Y, Z, Lb, Mb, Nb = self._forces_moments(V_air_b, p, q, r)

        # translational
        udot = r * v - q * w + X / m - g * np.sin(theta)
        vdot = p * w - r * u + Y / m + g * np.sin(phi) * np.cos(theta)
        wdot = q * u - p * v + Z / m + g * np.cos(phi) * np.cos(theta)

        # rotational (principal inertia)
        pdot = (Lb + (Iyy - Izz) * q * r) / Ixx
        qdot = (Mb + (Izz - Ixx) * p * r) / Iyy
        rdot = (Nb + (Ixx - Iyy) * p * q) / Izz

        # attitude kinematics
        phidot, thetadot, psidot = self._euler_rates(phi, theta, p, q, r)

        # position kinematics: R_b2n * V_body
        c, s = np.cos, np.sin
        cph, sph = c(phi), s(phi)
        cth, sth = c(theta), s(theta)
        cps, sps = c(psi), s(psi)
        Rb2n = np.array(
            [
                [cps * cth, cps * sth * sph - sps * cph, cps * sth * cph + sps * sph],
                [sps * cth, sps * sth * sph + cps * cph, sps * sth * cph - cps * sph],
                [-sth, cth * sph, cth * cph],
            ]
        )
        Ndot, Edot, Ddot = Rb2n @ np.array([u, v, w])

        return np.array([udot, vdot, wdot, pdot, qdot, rdot, phidot, thetadot, psidot, Ndot, Edot, Ddot])

    # ——— internals ————————————————————————————————————————————————

    def _seed_hist(self) -> None:
        """Initialize history with starting state."""
        st = self.latest()
        self.hist["u"] = [st["u"]]
        self.hist["v"] = [0.0]
        self.hist["w"] = [st["w"]]
        self.hist["p"] = [0.0]
        self.hist["q"] = [0.0]
        self.hist["r"] = [0.0]
        self.hist["phi"] = [0.0]
        self.hist["theta"] = [self.init["theta0_rad"]]
        self.hist["psi"] = [0.0]
        self.hist["N"] = [0.0]
        self.hist["E"] = [0.0]
        self.hist["D"] = [-self.init["h0"]]
        self.hist["V"] = [float(np.hypot(st["u"], st["w"]))]
        self.hist["h"] = [self.init["h0"]]
        self.hist["delta_a"] = [0.0]
        self.hist["delta_e"] = [0.0]
        self.hist["delta_r"] = [0.0]
        self.hist["throttle"] = [50.0]

    def _update_history(self) -> None:
        st = self.latest()
        self.hist["u"].append(st["u"])
        self.hist["v"].append(self.x[1, -1])
        self.hist["w"].append(st["w"])
        self.hist["p"].append(self.x[3, -1])
        self.hist["q"].append(self.x[4, -1])
        self.hist["r"].append(self.x[5, -1])
        self.hist["phi"].append(self.x[6, -1])
        self.hist["theta"].append(self.x[7, -1])
        self.hist["psi"].append(self.x[8, -1])
        self.hist["N"].append(self.x[9, -1])
        self.hist["E"].append(self.x[10, -1])
        self.hist["D"].append(self.x[11, -1])
        self.hist["V"].append(st["V"])
        self.hist["h"].append(st["h"])
        da, de, dr, thr = self.u_cmd
        self.hist["delta_a"].append(np.degrees(da))
        self.hist["delta_e"].append(np.degrees(de))
        self.hist["delta_r"].append(np.degrees(dr))
        self.hist["throttle"].append(100.0 * thr)


# ──────────────────────────────────────────────────────────────────────────
# GUI (VIEW + CONTROLLER)
# ──────────────────────────────────────────────────────────────────────────

class FlightSimulatorGUI:
    """
    Tkinter UI for the 747 dynamics. Uses a SixDOFSimulator instance.
    """

    def __init__(self, root: tk.Tk, sim: SixDOFSimulator):
        self.root = root
        self.sim = sim

        self.root.title("747 Flight Dynamics Simulator")
        self.root.configure(bg="#f0f0f0")

        # settings
        self.nav_window_m = 80.0
        self.running = False
        self.max_pts = int(self.sim.t_max / self.sim.dt)

        # UI widgets + plots
        self._create_control_panel()
        self._create_plots()

        # background integration thread
        Thread(target=self._loop, daemon=True).start()

    # ——— UI: controls ————————————————————————————————————————————————

    def _create_control_panel(self) -> None:
        lf, bf = ("Arial", 12), ("Arial", 12)
        top = tk.Frame(self.root, bg="#f0f0f0")
        top.pack(fill="x", pady=2)

        left = tk.Frame(top, bg="#f0f0f0")
        right = tk.Frame(top, bg="#f0f0f0")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        def slider(parent, row, txt, vmin, vmax, res) -> tk.Scale:
            tk.Label(parent, text=txt, font=lf, bg="#f0f0f0").grid(row=row, column=0, padx=5, pady=2, sticky="w")
            s = tk.Scale(
                parent,
                from_=vmin,
                to=vmax,
                resolution=res,
                orient=tk.HORIZONTAL,
                length=300,
                bg="#f0f0f0",
                highlightthickness=0,
            )
            s.grid(row=row, column=1, padx=5, pady=2, sticky="ew")
            parent.columnconfigure(1, weight=1)
            return s

        # LEFT: aileron/elevator/rudder
        self.aileron = slider(left, 0, "Aileron (δ_a) [deg]", -10, 10, 0.1)
        self.elevator = slider(left, 1, "Elevator (δ_e) [deg]", -10, 10, 0.1)
        self.rudder = slider(left, 2, "Rudder (δ_r) [deg]", -5, 5, 0.1)

        # RIGHT: flight condition, throttle, window
        tk.Label(right, text="Flight condition", font=lf, bg="#f0f0f0").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.fc_var = tk.StringVar(value=self.sim.condition)
        self.fc_menu = tk.OptionMenu(
            right, self.fc_var, "Approach", "CruiseHigh", "CruiseLow", command=lambda v: self._on_condition(v)
        )
        self.fc_menu.configure(bg="#f0f0f0")
        self.fc_menu.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(right, text="Throttle [%]", font=lf, bg="#f0f0f0").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.throttle = tk.Scale(
            right, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, length=300, bg="#f0f0f0", highlightthickness=0
        )
        self.throttle.set(50)
        self.throttle.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        right.columnconfigure(1, weight=1)

        tk.Label(right, text="Tracking window [m] (half-size)", font=lf, bg="#f0f0f0").grid(
            row=2, column=0, padx=5, pady=2, sticky="w"
        )
        self.navwin = tk.Scale(
            right,
            from_=40,
            to=200,
            resolution=10,
            orient=tk.HORIZONTAL,
            length=300,
            bg="#f0f0f0",
            highlightthickness=0,
            command=lambda v: setattr(self, "nav_window_m", float(v)),
        )
        self.navwin.set(int(self.nav_window_m))
        self.navwin.grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # buttons
        btn = tk.Frame(right, bg="#f0f0f0")
        btn.grid(row=3, column=0, columnspan=2, pady=2, sticky="e")
        self.start_b = tk.Button(
            btn, text="Start Simulation", command=self._start, font=bf, padx=10, pady=5, bg="#4CAF50", fg="white"
        )
        self.stop_b = tk.Button(
            btn,
            text="Stop Simulation",
            command=self._stop,
            font=bf,
            padx=10,
            pady=5,
            bg="#F44336",
            fg="white",
            state="disabled",
        )
        self.reset_b = tk.Button(
            btn, text="Reset", command=self._reset, font=bf, padx=10, pady=5, bg="#2196F3", fg="white"
        )
        for b in (self.start_b, self.stop_b, self.reset_b):
            b.pack(side=tk.LEFT, padx=5)

    # ——— UI: plotting ————————————————————————————————————————————————

    def _create_plots(self) -> None:
        self.fig = plt.figure(figsize=(20, 20))
        gs = self.fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1], hspace=0.4)

        # left: 4 time series
        self.ax1, self.ax2, self.ax3, self.ax4 = (self.fig.add_subplot(gs[i, 0]) for i in range(4))
        # right: 3D
        self.ax3d = self.fig.add_subplot(gs[:, 1], projection="3d")

        # plot 1: u,v,w
        self.l_u, = self.ax1.plot([], [], color="k", label="Forward (u) [m/s]", lw=2)
        self.l_v, = self.ax1.plot([], [], color="m", label="Right (v) [m/s]", lw=2)
        self.l_w, = self.ax1.plot([], [], color="tab:orange", label="Down (w) [m/s]", lw=2)
        self.ax1.set(title="Body Velocities", ylabel="[m/s]", xlim=(0, self.sim.t_max))
        self.ax1.grid(True)
        self.ax1.legend(fontsize=8)

        # plot 2: p,q,r
        self.l_p, = self.ax2.plot([], [], color="b", label="Roll rate (p) [deg/s]", lw=2)
        self.l_q, = self.ax2.plot([], [], color="g", label="Pitch rate (q) [deg/s]", lw=2)
        self.l_r, = self.ax2.plot([], [], color="r", label="Yaw rate (r) [deg/s]", lw=2)
        self.ax2.set(title="Body Rates", ylabel="[deg/s]", xlim=(0, self.sim.t_max))
        self.ax2.grid(True)
        self.ax2.legend(fontsize=8)

        # plot 3: euler
        self.l_phi, = self.ax3.plot([], [], color="b", label="Roll (φ) [deg]", lw=2)
        self.l_theta, = self.ax3.plot([], [], color="g", label="Pitch (θ) [deg]", lw=2)
        self.l_psi, = self.ax3.plot([], [], color="r", label="Heading (ψ) [deg]", lw=2)
        self.ax3.set(title="Euler Angles", ylabel="[deg]", xlim=(0, self.sim.t_max))
        self.ax3.grid(True)
        self.ax3.legend(fontsize=8)

        # plot 4: position
        self.l_N, = self.ax4.plot([], [], color="tab:olive", label="North (N) [m]", lw=2)
        self.l_E, = self.ax4.plot([], [], color="tab:cyan", label="East (E) [m]", lw=2)
        self.l_h, = self.ax4.plot([], [], color="tab:gray", label="Altitude (h) [m]", lw=2)
        self.ax4.set(title="Position (NEh)", xlabel="Time [s]", ylabel="[m]", xlim=(0, self.sim.t_max))
        self.ax4.grid(True)
        self.ax4.legend(fontsize=8)

        # 3D view
        self._setup_plane_model()
        w = self.nav_window_m
        self.ax3d.set(title="747 Position & Orientation (N, E, h)", xlabel="North [m]", ylabel="East [m]")
        self.ax3d.set_zlabel("Altitude h [m]")
        self.ax3d.set(xlim=(-w, w), ylim=(w, -w), zlim=(0, 2 * w))
        try:
            self.ax3d.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        self.ax3d.grid(True, alpha=0.3)
        self.ax3d.view_init(elev=20, azim=45)
        self.ax3d.set_facecolor("#f5f5f5")
        self.ax3d.dist = 7

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

    # ——— geometry helpers ——————————————————————————————————————————————

    def _setup_plane_model(self) -> None:
        # scale geometry so 1.6*s ≈ b → s = b/1.6
        s = self.sim.params["b"] / 1.6
        sign = -1.0
        self.base_coords, self.surfs, self.ctrl_idx = [], [], {}

        def add(X, Y, Z, color, alpha, name=None):
            surf = self.ax3d.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)
            self.surfs.append(surf)
            self.base_coords.append((X, Y, Z))
            if name:
                self.ctrl_idx[name] = len(self.surfs) - 1

        # fuselage
        th = np.linspace(0, 2 * np.pi, 20)
        z_lin = np.linspace(-1, 1, 20) * s
        TH, Z = np.meshgrid(th, z_lin)
        r = 0.05 * s
        add(sign * Z, sign * r * np.cos(TH), r * np.sin(TH), "silver", 0.8)

        # wings
        span, sweep, dihedral = 0.8 * s, 0.3 * s, 0.1 * s
        Xw = np.array([[0, sweep]] * 4) - 0.2 * s
        Yw = np.array([[-span, -span], [span, span]] * 2)
        Zw = np.array([[0, dihedral], [0, dihedral], [dihedral, dihedral], [0, 0]])
        add(sign * Xw, Yw, Zw, "gray", 0.7)

        # horizontal tail
        hspan = 0.3 * s
        Xh = np.array([[0.8, 0.9]] * 4) * s
        Yh = np.array([[-hspan, -hspan], [hspan, hspan]] * 2)
        Zh = np.zeros((4, 2))
        add(sign * Xh, Yh, Zh, "gray", 0.7)

        # vertical tail
        vht = 0.3 * s
        Xv = np.array([[0.8, 0.9]] * 4) * s
        Yv = np.zeros((4, 2))
        Zv = np.array([[0, 0], [0, 0], [vht, vht], [vht, vht]])
        add(sign * Xv, Yv, Zv, "gray", 0.7)

        # control-surface helper
        def plate(x_le, chord, y0, y1, z_off=0):
            X = np.array([[x_le, x_le + chord]] * 2)
            Y = np.array([[y0, y0], [y1, y1]])
            return X, Y, np.full_like(X, z_off)

        # ailerons (blue)
        chord_a, x_a = 0.12 * s, -0.1 * s + sweep
        y_out, y_in = span, 0.5 * span
        add(*plate(sign * x_a, chord_a, -y_out, -y_in, dihedral), "blue", 0.9, "ail_L")
        add(*plate(sign * x_a, chord_a, y_in, y_out, dihedral), "blue", 0.9, "ail_R")

        # elevators (green)
        chord_e, x_e = 0.12 * s, 0.99 * s
        ye_out, ye_in = hspan, 0.1 * hspan
        add(*plate(sign * x_e, chord_e, -ye_out, -ye_in), "green", 0.9, "elev_L")
        add(*plate(sign * x_e, chord_e, ye_in, ye_out), "green", 0.9, "elev_R")

        # rudder (red)
        chord_r, x_r = 0.12 * s, 0.88 * s
        z_top = vht
        Xr = np.array([[x_r, x_r + chord_r]] * 2)
        Yr = np.zeros_like(Xr)
        Zr = np.array([[0, 0], [z_top, z_top]])
        add(sign * Xr, Yr, Zr, "red", 0.9, "rudder")

    @staticmethod
    def _R_body(phi: float, th: float, ps: float) -> np.ndarray:
        cφ, sφ, cθ, sθ, cψ, sψ = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(ps), np.sin(ps)
        Rr = np.array([[1, 0, 0], [0, cφ, -sφ], [0, sφ, cφ]])
        Rp = np.array([[cθ, 0, sθ], [0, 1, 0], [-sθ, 0, cθ]])
        Ry = np.array([[cψ, -sψ, 0], [sψ, cψ, 0], [0, 0, 1]])
        return Ry @ Rp @ Rr

    @staticmethod
    def _rotate_pts(pts: np.ndarray, angle: float, axis: str) -> np.ndarray:
        if angle == 0.0:
            return pts
        c, s = np.cos(angle), np.sin(angle)
        cx, cy, cz = pts.mean(1, keepdims=True)
        shifted = pts - np.vstack((cx, cy, cz))
        if axis == "x":
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return R @ shifted + np.vstack((cx, cy, cz))

    def _update_plane_model(self, phi, θ, ψ, N, E, h, da, de, dr) -> None:
        Rb = self._R_body(-phi, -θ, ψ)
        defl = {"ail_L": -da, "ail_R": da, "elev_L": de, "elev_R": de, "rudder": dr}
        axis = {"ail_L": "y", "ail_R": "y", "elev_L": "y", "elev_R": "y", "rudder": "z"}

        for surf in getattr(self, "surfs", []):
            surf.remove()
        self.surfs = []

        for i, (X0, Y0, Z0) in enumerate(self.base_coords):
            pts = np.vstack((X0.flatten(), Y0.flatten(), Z0.flatten()))
            name = next((n for n, j in self.ctrl_idx.items() if j == i), None)
            if name:
                pts = self._rotate_pts(pts, defl[name], axis[name])
            pts = Rb @ pts
            pts = pts + np.array([[N], [E], [h]])

            Xr, Yr, Zr = (pts[k].reshape(X0.shape) for k in range(3))
            col = {
                "ail_L": "blue",
                "ail_R": "blue",
                "elev_L": "green",
                "elev_R": "green",
                "rudder": "red",
            }.get(name, "silver" if i == 0 else "gray")
            alpha = 0.9 if name else 0.8 if i == 0 else 0.7
            self.surfs.append(self.ax3d.plot_surface(Xr, Yr, Zr, color=col, alpha=alpha, linewidth=0))

    # ——— main loop & events ————————————————————————————————————————————

    def _loop(self) -> None:
        while True:
            while self.running and self.sim.t_hist[-1] < self.sim.t_max:
                # controls from UI
                da = np.radians(self.aileron.get())
                de = np.radians(self.elevator.get())
                dr = np.radians(self.rudder.get())
                thr = 0.01 * self.throttle.get()
                self.sim.set_controls(da, de, dr, thr)

                # integrate one step
                self.sim.step()

                # refresh plots
                self._update_plots()
            time.sleep(0.1)

    def _on_condition(self, name: str) -> None:
        self.sim.reset(name)
        # reset UI controls
        for sld in (self.aileron, self.elevator, self.rudder):
            sld.set(0)
        self.throttle.set(50)
        # rebuild 3D geometry (b may have changed)
        try:
            for s in getattr(self, "surfs", []):
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

    # ——— plotting updates ——————————————————————————————————————————————

    def _update_plots(self) -> None:
        s = max(0, len(self.sim.t_hist) - self.max_pts)
        t = self.sim.t_hist

        # plot 1: u,v,w
        self.l_u.set_data(t[s:], self.sim.hist["u"][s:])
        self.l_v.set_data(t[s:], self.sim.hist["v"][s:])
        self.l_w.set_data(t[s:], self.sim.hist["w"][s:])
        self.ax1.relim()
        self.ax1.autoscale_view()

        # plot 2: p,q,r (deg/s)
        self.l_p.set_data(t[s:], np.degrees(self.sim.hist["p"][s:]))
        self.l_q.set_data(t[s:], np.degrees(self.sim.hist["q"][s:]))
        self.l_r.set_data(t[s:], np.degrees(self.sim.hist["r"][s:]))
        self.ax2.relim()
        self.ax2.autoscale_view()

        # plot 3: φ,θ,ψ
        self.l_phi.set_data(t[s:], np.degrees(self.sim.hist["phi"][s:]))
        self.l_theta.set_data(t[s:], np.degrees(self.sim.hist["theta"][s:]))
        self.l_psi.set_data(t[s:], np.degrees(self.sim.hist["psi"][s:]))
        self.ax3.relim()
        self.ax3.autoscale_view()

        # plot 4: N,E,h
        self.l_N.set_data(t[s:], self.sim.hist["N"][s:])
        self.l_E.set_data(t[s:], self.sim.hist["E"][s:])
        self.l_h.set_data(t[s:], self.sim.hist["h"][s:])
        self.ax4.relim()
        self.ax4.autoscale_view()

        # 3D model
        phi = self.sim.hist["phi"][-1]
        th = self.sim.hist["theta"][-1]
        ps = self.sim.hist["psi"][-1]
        N = self.sim.hist["N"][-1]
        E = self.sim.hist["E"][-1]
        h = self.sim.hist["h"][-1]
        da = np.radians(self.sim.hist["delta_a"][-1])
        de = np.radians(self.sim.hist["delta_e"][-1])
        dr = np.radians(self.sim.hist["delta_r"][-1])

        self._update_plane_model(phi, th, ps, N, E, h, da, de, dr)

        # center nav window
        try:
            w = self.nav_window_m
            self.ax3d.set(xlim=(N - w, N + w), ylim=(E + w, E - w), zlim=(max(0.0, h - w), h + w))
            self.ax3d.set_box_aspect([1, 1, 1])
        except Exception:
            pass

        self.canvas.draw_idle()
        self.canvas.flush_events()

    # ——— buttons ————————————————————————————————————————————————————

    def _start(self) -> None:
        self.running = True
        self.start_b.config(state="disabled")
        self.stop_b.config(state="normal")

    def _stop(self) -> None:
        self.running = False
        self.start_b.config(state="normal")
        self.stop_b.config(state="disabled")

    def _reset(self) -> None:
        self.running = False
        self._on_condition(self.fc_var.get())


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    sim = SixDOFSimulator(condition="Approach", dt=0.10, t_max=120.0)
    FlightSimulatorGUI(root, sim)
    root.mainloop()
