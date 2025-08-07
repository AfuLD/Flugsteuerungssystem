import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import tkinter as tk
from threading import Thread
import time


# ─── 747 (Roskam, power-approach) state-space model ─────────────────────────
A = np.array([[-1.020, 0, 0.396],
              [0, -0.380, 0],
              [-0.143, 0, -0.232]])
B = np.array([[0.243, 0, 0],
              [0, -0.402, 0],
              [0.012, 0, -0.165]])

def flight_dynamics(x, t, u):
    return A @ x + B @ u
# -----------------------------------------------------------------------------


class FlightSimulatorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("747 Flight Dynamics Simulator")
        self.root.configure(bg="#f0f0f0")

        # time bookkeeping
        self.dt, self.t_max, self.t = 0.1, 30, [0]

        # states (p q r) and inputs (δa δe δr)
        self.x = np.array([[0, 0, 0]]).T
        self.u = np.array([0, 0, 0])

        # history buffers
        self.hist = {k: [0] for k in
                     ("p", "q", "r", "phi", "theta", "psi",
                      "delta_a", "delta_e", "delta_r")}
        self.running, self.max_pts = False, int(self.t_max / self.dt)

        self._create_control_panel()
        self._create_plots()
        Thread(target=self._simulate, daemon=True).start()

    # ───────────────────────── GUI CONTROLS ────────────────────────────────
    def _create_control_panel(self):
        lf, bf = ("Arial", 12), ("Arial", 12)
        frame = tk.Frame(self.root, bg="#f0f0f0"); frame.pack(fill="x", pady=2)

        def slider(row, txt):
            tk.Label(frame, text=txt, font=lf, bg="#f0f0f0") \
                .grid(row=row, column=0, padx=5, pady=2, sticky="w")
            s = tk.Scale(frame, from_=-10, to=10, resolution=0.1,
                         orient=tk.HORIZONTAL, length=300,
                         bg="#f0f0f0", highlightthickness=0)
            s.grid(row=row, column=1, padx=5, pady=2, sticky="ew")
            return s
        self.aileron, self.elevator, self.rudder = \
            slider(0, "Aileron (δ_a) [deg]"), slider(1, "Elevator (δ_e) [deg]"), slider(2, "Rudder (δ_r) [deg]")

        # buttons
        btn = tk.Frame(frame, bg="#f0f0f0"); btn.grid(row=3, column=0, columnspan=2, pady=2)
        self.start_b = tk.Button(btn, text="Start Simulation", command=self._start_sim,
                                 font=bf, padx=10, pady=5, bg="#4CAF50", fg="white")
        self.stop_b  = tk.Button(btn, text="Stop Simulation",  command=self._stop_sim,
                                 font=bf, padx=10, pady=5, bg="#F44336", fg="white",
                                 state="disabled")
        self.reset_b = tk.Button(btn, text="Reset", command=self._reset_sim,
                                 font=bf, padx=10, pady=5, bg="#2196F3", fg="white")
        for b in (self.start_b, self.stop_b, self.reset_b):
            b.pack(side=tk.LEFT, padx=5)
        frame.columnconfigure(1, weight=1)

    # ───────────────────────── PLOTS ──────────────────────────────────────
    def _create_plots(self):
        self.fig = plt.figure(figsize=(20, 20))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1],
                                   width_ratios=[1, 1], hspace=0.4)
        self.ax1, self.ax2, self.ax3 = (self.fig.add_subplot(gs[i,0]) for i in range(3))
        self.ax4 = self.fig.add_subplot(gs[:,1], projection='3d')

        # traces
        self.l_da, = self.ax1.plot([], [], 'b', label='Aileron (δ_a)', lw=2)
        self.l_de, = self.ax1.plot([], [], 'g', label='Elevator (δ_e)', lw=2)
        self.l_dr, = self.ax1.plot([], [], 'r', label='Rudder (δ_r)', lw=2)
        self.ax1.set(title='Control Surface Deflections', ylabel='Deflection [deg]',
                     xlim=(0, self.t_max)); self.ax1.grid(True); self.ax1.legend(fontsize=8)

        self.l_p,  = self.ax2.plot([], [], 'b', label='Roll Rate (p)',  lw=2)
        self.l_q,  = self.ax2.plot([], [], 'g', label='Pitch Rate (q)', lw=2)
        self.l_r,  = self.ax2.plot([], [], 'r', label='Yaw Rate (r)',   lw=2)
        self.ax2.set(title='Angular Rates', ylabel='Rate [deg/s]',
                     xlim=(0, self.t_max)); self.ax2.grid(True); self.ax2.legend(fontsize=8)

        self.l_phi,   = self.ax3.plot([], [], 'b', label='Roll Angle (φ)',  lw=2)
        self.l_theta, = self.ax3.plot([], [], 'g', label='Pitch Angle (θ)', lw=2)
        self.l_psi,   = self.ax3.plot([], [], 'r', label='Yaw Angle (ψ)',   lw=2)
        self.ax3.set(title='Euler Angles', xlabel='Time [s]', ylabel='Angle [deg]',
                     xlim=(0, self.t_max)); self.ax3.grid(True); self.ax3.legend(fontsize=8)

        self._setup_plane_model()
        limit = 4     # because scale is larger now
        self.ax4.set(title='747 Orientation', xlabel='X', ylabel='Y')
        self.ax4.set_zlabel('Z')
        self.ax4.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
        self.ax4.grid(True, alpha=0.3)
        self.ax4.view_init(elev=20, azim=45)
        self.ax4.set_facecolor('#f5f5f5')
        self.ax4.dist = 7
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

    # ───────────── GEOMETRY ─────────────────────────────
    def _setup_plane_model(self):
        s, sign = 4, -1.0
        self.base_coords, self.surfs, self.ctrl_idx = [], [], {}

        def add(X, Y, Z, color, alpha, name=None):
            surf = self.ax4.plot_surface(X, Y, Z, color=color,
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

        # elevators (green) – hinge about x-axis (pitch)
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

    def _update_plane_model(self, phi, θ, ψ, da, de, dr):
        Rb = self._R_body(phi, θ, ψ)
        defl = {'ail_L': da, 'ail_R': -da, 'elev_L': de,
                'elev_R': de, 'rudder': dr}
        axis = {'ail_L': 'y', 'ail_R': 'y',   # ← rotate about y-axis
                'elev_L': 'y', 'elev_R': 'y',
                'rudder': 'z'}

        for surf in self.surfs: surf.remove()
        self.surfs = []

        for i, (X0, Y0, Z0) in enumerate(self.base_coords):
            pts = np.vstack((X0.flatten(), Y0.flatten(), Z0.flatten()))
            name = next((n for n, j in self.ctrl_idx.items() if j == i), None)
            if name: pts = self._rotate_pts(pts, defl[name], axis[name])
            pts = Rb @ pts  # body attitude

            Xr, Yr, Zr = (pts[k].reshape(X0.shape) for k in range(3))
            col = {'ail_L':'blue','ail_R':'blue',
                   'elev_L':'green','elev_R':'green',
                   'rudder':'red'}.get(name, 'silver' if i==0 else 'gray')
            alpha = 0.9 if name else 0.8 if i==0 else 0.7
            self.surfs.append(
                self.ax4.plot_surface(Xr, Yr, Zr, color=col, alpha=alpha, linewidth=0)
            )

    # ───────────── SIMULATION LOOP ─────────────────────────────────────────
    def _simulate(self):
        while True:
            if self.running and self.t[-1] < self.t_max:
                da = np.radians(self.aileron.get())
                de = np.radians(self.elevator.get())
                dr = np.radians(self.rudder.get())
                self.u = np.array([da, de, dr])

                sol = odeint(flight_dynamics, self.x[:, -1],
                             [self.t[-1], self.t[-1]+self.dt], args=(self.u,))
                self.x = np.hstack((self.x, sol[-1, :].reshape(-1,1)))
                self.t.append(self.t[-1]+self.dt)

                p,q,r = self.x[:,-1]
                self.hist['p'].append(p);  self.hist['q'].append(q);  self.hist['r'].append(r)
                self.hist['phi'  ].append(self.hist['phi'  ][-1] + .5*(self.hist['p'][-2]+p)*self.dt)
                self.hist['theta'].append(self.hist['theta'][-1] + .5*(self.hist['q'][-2]+q)*self.dt)
                self.hist['psi'  ].append(self.hist['psi'  ][-1] + .5*(self.hist['r'][-2]+r)*self.dt)
                for k,v in zip(('delta_a','delta_e','delta_r'), (da,de,dr)):
                    self.hist[k].append(np.degrees(v))

                self._update_plots()
            time.sleep(self.dt/2)

    # ───────────── PLOT/3-D UPDATE ────────────────────────────────────────
    def _update_plots(self):
        s = max(0, len(self.t)-self.max_pts)
        # control surfaces
        self.l_da.set_data(self.t[s:], self.hist['delta_a'][s:])
        self.l_de.set_data(self.t[s:], self.hist['delta_e'][s:])
        self.l_dr.set_data(self.t[s:], self.hist['delta_r'][s:])
        self.ax1.relim(); self.ax1.autoscale_view()

        # body rates
        self.l_p.set_data(self.t[s:], np.degrees(self.hist['p'][s:]))
        self.l_q.set_data(self.t[s:], np.degrees(self.hist['q'][s:]))
        self.l_r.set_data(self.t[s:], np.degrees(self.hist['r'][s:]))
        self.ax2.relim(); self.ax2.autoscale_view()

        # Euler angles
        self.l_phi  .set_data(self.t[s:], np.degrees(self.hist['phi'  ][s:]))
        self.l_theta.set_data(self.t[s:], np.degrees(self.hist['theta'][s:]))
        self.l_psi  .set_data(self.t[s:], np.degrees(self.hist['psi'  ][s:]))
        self.ax3.relim(); self.ax3.autoscale_view()

        self._update_plane_model(self.hist['phi']  [-1],
                                 self.hist['theta'][-1],
                                 self.hist['psi']  [-1],
                                 np.radians(self.hist['delta_a'][-1]),
                                 np.radians(self.hist['delta_e'][-1]),
                                 np.radians(self.hist['delta_r'][-1]))
        self.canvas.draw_idle(); self.canvas.flush_events()

    # ───────────── BUTTONS ────────────────────────────────────────────────
    def _start_sim(self): self.running=True;  self.start_b.config(state='disabled'); self.stop_b.config(state='normal')
    def _stop_sim(self):  self.running=False; self.start_b.config(state='normal');  self.stop_b.config(state='disabled')
    def _reset_sim(self):
        self.running=False; self.t=[0]; self.x=np.array([[0,0,0]]).T
        for k in self.hist: self.hist[k]=[0]
        for sld in (self.aileron, self.elevator, self.rudder): sld.set(0)
        self._update_plots()


# ───────────── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk(); FlightSimulatorGUI(root); root.mainloop()
