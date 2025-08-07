import threading

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D, art3d
import tkinter as tk
from threading import Thread
import time

# Set modern plot style
plt.style.use('seaborn-v0_8')

# Define state-space matrices for 747 (Power Approach)
A = np.array([[-1.020, 0, 0.396],
              [0, -0.380, 0],
              [-0.143, 0, -0.232]])
B = np.array([[0.243, 0, 0],
              [0, -0.402, 0],
              [0.012, 0, -0.165]])

# Flight dynamics function
def flight_dynamics(x, t, u):
    return A @ x + B @ u

# GUI class
class FlightSimulatorGUI:
    def __init__(self, root):
        self.lock = threading.Lock()
        self.root = root
        self.root.title("747 Flight Dynamics Simulator")
        self.root.configure(bg='#f0f0f0')

        # Simulation parameters
        self.dt = 0.1
        self.t_max = 30
        self.t = [0]
        self.x = np.array([[0, 0, 0]]).T
        self.u = np.array([0, 0, 0])
        self.history = {'p': [0], 'q': [0], 'r': [0],
                        'phi': [0], 'theta': [0], 'psi': [0],
                        'delta_a': [0], 'delta_e': [0], 'delta_r': [0]}
        self.running = False
        self.max_points = int(self.t_max / self.dt)

        # Create GUI elements
        self.create_control_panel()
        self.create_plots()

        # Start simulation thread
        self.thread = Thread(target=self.simulate)
        self.thread.daemon = True
        self.thread.start()

    def create_control_panel(self):
        label_font = ('Arial', 14)
        button_font = ('Arial', 14)

        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(fill='x', pady=5)

        tk.Label(control_frame, text="Aileron (δ_a) [deg]", font=label_font, bg='#f0f0f0').grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.aileron_slider = tk.Scale(control_frame, from_=-10, to=10, resolution=0.1, orient=tk.HORIZONTAL, length=350, bg='#f0f0f0', highlightthickness=0)
        self.aileron_slider.grid(row=0, column=1, padx=10, pady=5, sticky='ew')

        tk.Label(control_frame, text="Elevator (δ_e) [deg]", font=label_font, bg='#f0f0f0').grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.elevator_slider = tk.Scale(control_frame, from_=-10, to=10, resolution=0.1, orient=tk.HORIZONTAL, length=350, bg='#f0f0f0', highlightthickness=0)
        self.elevator_slider.grid(row=1, column=1, padx=10, pady=5, sticky='ew')

        tk.Label(control_frame, text="Rudder (δ_r) [deg]", font=label_font, bg='#f0f0f0').grid(row=2, column=0, padx=10, pady=5, sticky='w')
        self.rudder_slider = tk.Scale(control_frame, from_=-10, to=10, resolution=0.1, orient=tk.HORIZONTAL, length=350, bg='#f0f0f0', highlightthickness=0)
        self.rudder_slider.grid(row=2, column=1, padx=10, pady=5, sticky='ew')

        button_frame = tk.Frame(control_frame, bg='#f0f0f0')
        button_frame.grid(row=3, column=0, columnspan=2, pady=5)

        self.start_button = tk.Button(button_frame, text="Start Simulation", command=self.start_simulation, font=button_font, padx=15, pady=10, bg='#4CAF50', fg='white')
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(button_frame, text="Stop Simulation", command=self.stop_simulation, font=button_font, padx=15, pady=10, bg='#F44336', fg='white')
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset_simulation, font=button_font, padx=15, pady=10, bg='#2196F3', fg='white')
        self.reset_button.pack(side=tk.LEFT, padx=10)

        control_frame.columnconfigure(1, weight=1)

    def create_plots(self):
        self.fig = plt.figure(figsize=(18, 16))
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1.5, 1], hspace=0.4)

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.fig.add_subplot(gs[2, 0])
        self.ax4 = self.fig.add_subplot(gs[:, 1], projection='3d')

        self.line_da, = self.ax1.plot([], [], 'b', label='Aileron (δ_a)', linewidth=2.5)
        self.line_de, = self.ax1.plot([], [], 'g', label='Elevator (δ_e)', linewidth=2.5)
        self.line_dr, = self.ax1.plot([], [], 'r', label='Rudder (δ_r)', linewidth=2.5)
        self.ax1.set_title('Control Surface Deflections', fontsize=14, pad=15)
        self.ax1.set_ylabel('Deflection [deg]', fontsize=12)
        self.ax1.set_xlim(0, self.t_max)
        self.ax1.grid(True)
        self.ax1.legend(fontsize=10)

        self.line_p, = self.ax2.plot([], [], 'b', label='Roll Rate (p)', linewidth=2.5)
        self.line_q, = self.ax2.plot([], [], 'g', label='Pitch Rate (q)', linewidth=2.5)
        self.line_r, = self.ax2.plot([], [], 'r', label='Yaw Rate (r)', linewidth=2.5)
        self.ax2.set_title('Angular Rates', fontsize=14, pad=15)
        self.ax2.set_ylabel('Rate [deg/s]', fontsize=12)
        self.ax2.set_xlim(0, self.t_max)
        self.ax2.grid(True)
        self.ax2.legend(fontsize=10)

        self.line_phi, = self.ax3.plot([], [], 'b', label='Roll Angle (φ)', linewidth=2.5)
        self.line_theta, = self.ax3.plot([], [], 'g', label='Pitch Angle (θ)', linewidth=2.5)
        self.line_psi, = self.ax3.plot([], [], 'r', label='Yaw Angle (ψ)', linewidth=2.5)
        self.ax3.set_title('Euler Angles', fontsize=14, pad=15)
        self.ax3.set_xlabel('Time [s]', fontsize=12)
        self.ax3.set_ylabel('Angle [deg]', fontsize=12)
        self.ax3.set_xlim(0, self.t_max)
        self.ax3.grid(True)
        self.ax3.legend(fontsize=10)

        self.setup_plane_model()
        self.ax4.set_title('747 Orientation', fontsize=14, pad=15)
        self.ax4.set_xlabel('X', fontsize=12)
        self.ax4.set_ylabel('Y', fontsize=12)
        self.ax4.set_zlabel('Z', fontsize=12)
        self.ax4.set_xlim(-5, 5)
        self.ax4.set_ylim(-5, 5)
        self.ax4.set_zlim(-5, 5)
        self.ax4.grid(True, alpha=0.3)
        self.ax4.view_init(elev=20, azim=45)
        self.ax4.set_facecolor('#f5f5f5')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

    def setup_plane_model(self):
        scale = 2.5
        # Fuselage: approximate as a thin cylinder
        theta = np.linspace(0, 2 * np.pi, 20)
        z_fuselage = np.linspace(-1, 1, 20) * scale
        theta, z = np.meshgrid(theta, z_fuselage)
        r = 0.05 * scale
        x_fuselage = z
        y_fuselage = r * np.cos(theta)
        z_fuselage = r * np.sin(theta)

        # Wings: swept, dihedral
        wing_span = 0.8 * scale
        wing_sweep = 0.3 * scale
        wing_dihedral = 0.1 * scale
        x_wing = np.array([[0, wing_sweep], [0, wing_sweep], [0, wing_sweep], [0, wing_sweep]]) - 0.2 * scale
        y_wing = np.array([[-wing_span, -wing_span], [wing_span, wing_span], [wing_span, wing_span], [-wing_span, -wing_span]])
        z_wing = np.array([[0, wing_dihedral], [0, wing_dihedral], [wing_dihedral, wing_dihedral], [0, 0]])

        # Horizontal tail
        htail_span = 0.3 * scale
        x_htail = np.array([[0.8, 0.9], [0.8, 0.9], [0.8, 0.9], [0.8, 0.9]]) * scale
        y_htail = np.array([[-htail_span, -htail_span], [htail_span, htail_span], [htail_span, htail_span], [-htail_span, -htail_span]])
        z_htail = np.zeros((4, 2))

        # Vertical tail
        vtail_height = 0.4 * scale  # Increased from 0.3 to 0.4 for taller rudder
        x_vtail = np.array([[0.8, 0.9], [0.8, 0.9], [0.8, 0.9], [0.8, 0.9]]) * scale
        y_vtail = np.zeros((4, 2))
        z_vtail = np.array([[0, 0], [0, 0], [vtail_height, vtail_height], [vtail_height, vtail_height]])

        # Ailerons (blue) on wings backside, bigger
        aileron_span = 0.3 * scale  # Kept bigger, tilt removed
        x_aileron_left = np.array([wing_sweep - 0.2 * scale, wing_sweep - 0.2 * scale, wing_sweep - 0.2 * scale, wing_sweep - 0.2 * scale])
        y_aileron_left = np.array([-wing_span, -wing_span + aileron_span, -wing_span + aileron_span, -wing_span])
        z_aileron_left = np.array([wing_dihedral, wing_dihedral, wing_dihedral + 0.05, wing_dihedral + 0.05])  # Tilt removed
        x_aileron_right = np.array([wing_sweep - 0.2 * scale, wing_sweep - 0.2 * scale, wing_sweep - 0.2 * scale, wing_sweep - 0.2 * scale])
        y_aileron_right = np.array([wing_span - aileron_span, wing_span, wing_span, wing_span - aileron_span])
        z_aileron_right = np.array([wing_dihedral, wing_dihedral, wing_dihedral + 0.05, wing_dihedral + 0.05])  # Tilt removed
        self.aileron_left = art3d.Poly3DCollection([list(zip(x_aileron_left, y_aileron_left, z_aileron_left))], color='blue', alpha=0.9)
        self.aileron_right = art3d.Poly3DCollection([list(zip(x_aileron_right, y_aileron_right, z_aileron_right))], color='blue', alpha=0.9)

        # Elevators (green) on horizontal stabilizer backside, bigger
        elevator_span = 0.22 * scale
        x_elevator = np.array([0.9, 0.9, 0.9, 0.9]) * scale
        y_elevator_left = np.array([-htail_span, -htail_span + elevator_span, -htail_span + elevator_span, -htail_span])
        y_elevator_right = np.array([htail_span - elevator_span, htail_span, htail_span, htail_span - elevator_span])
        z_elevator = np.array([0.05, 0.05, 0, 0])
        self.elevator_left = art3d.Poly3DCollection([list(zip(x_elevator, y_elevator_left, z_elevator))], color='green', alpha=0.9)
        self.elevator_right = art3d.Poly3DCollection([list(zip(x_elevator, y_elevator_right, z_elevator))], color='green', alpha=0.9)

        # Rudder (red) on vertical stabilizer backside, taller
        rudder_width = 0.12 * scale  # Kept width the same
        x_rudder = np.array([0.9, 0.9, 0.9, 0.9]) * scale
        y_rudder = np.array([-rudder_width/2, rudder_width/2, rudder_width/2, -rudder_width/2])
        z_rudder = np.array([vtail_height - 0.15, vtail_height - 0.15, vtail_height, vtail_height])  # Extended height
        self.rudder = art3d.Poly3DCollection([list(zip(x_rudder, y_rudder, z_rudder))], color='red', alpha=0.9)

        self.plane_surfaces = [
            self.ax4.plot_surface(x_fuselage, y_fuselage, z_fuselage, color='lightgray', alpha=0.8),
            self.ax4.plot_surface(x_wing, y_wing, z_wing, color='gray', alpha=0.7),
            self.ax4.plot_surface(x_htail, y_htail, z_htail, color='gray', alpha=0.7),
            self.ax4.plot_surface(x_vtail, y_vtail, z_vtail, color='gray', alpha=0.7),
            self.rudder,
            self.aileron_left,
            self.aileron_right,
            self.elevator_left,
            self.elevator_right
        ]
        self.plane_coords = [
            (x_fuselage, y_fuselage, z_fuselage),
            (x_wing, y_wing, z_wing),
            (x_htail, y_htail, z_htail),
            (x_vtail, y_vtail, z_vtail),
            list(zip(x_rudder, y_rudder, z_rudder)),
            list(zip(x_aileron_left, y_aileron_left, z_aileron_left)),
            list(zip(x_aileron_right, y_aileron_right, z_aileron_right)),
            list(zip(x_elevator, y_elevator_left, z_elevator)),
            list(zip(x_elevator, y_elevator_right, z_elevator))
        ]
        for collection in [self.rudder, self.aileron_left, self.aileron_right, self.elevator_left, self.elevator_right]:
            self.ax4.add_collection3d(collection)

    def update_plane_model(self, phi, theta, psi):
        with self.lock:  # Protect against concurrent access
            delta_a = np.radians(self.aileron_slider.get())
            delta_e = np.radians(self.elevator_slider.get())
            delta_r = np.radians(self.rudder_slider.get())

            cphi, sphi = np.cos(phi), np.sin(phi)
            ctheta, stheta = np.cos(theta), np.sin(theta)
            cpsi, spsi = np.cos(psi), np.sin(psi)

            R_roll = np.array([[1, 0, 0],
                               [0, cphi, -sphi],
                               [0, sphi, cphi]])
            R_pitch = np.array([[ctheta, 0, stheta],
                                [0, 1, 0],
                                [-stheta, 0, ctheta]])
            R_yaw = np.array([[cpsi, -spsi, 0],
                              [spsi, cpsi, 0],
                              [0, 0, 1]])

            R_body = R_yaw @ R_pitch @ R_roll

            # Control surface rotations
            cda, sda = np.cos(delta_a), np.sin(delta_a)
            R_aileron_left = np.array([[cda, 0, -sda],
                                       [0, 1, 0],
                                       [sda, 0, cda]])
            R_aileron_right = np.array([[cda, 0, sda],
                                        [0, 1, 0],
                                        [-sda, 0, cda]])

            cde, sde = np.cos(delta_e), np.sin(delta_e)
            R_elevator = np.array([[cde, 0, -sde],
                                   [0, 1, 0],
                                   [sde, 0, cde]])

            cdr, sdr = np.cos(delta_r), np.sin(delta_r)
            R_rudder = np.array([[cdr, sdr, 0],
                                 [-sdr, cdr, 0],
                                 [0, 0, 1]])

            for i, coords in enumerate(self.plane_coords):
                try:
                    self.plane_surfaces[i].remove()
                    if i < 4:  # Body parts (x, y, z arrays)
                        x, y, z = coords
                        points = np.vstack((np.array(x).flatten(), np.array(y).flatten(), np.array(z).flatten()))
                        if points.size == 0:
                            continue
                        rotated_points = R_body @ points
                        x_rot = rotated_points[0].reshape(len(x), -1)
                        y_rot = rotated_points[1].reshape(len(y), -1)
                        z_rot = rotated_points[2].reshape(len(z), -1)
                    else:  # Control surfaces (list of points)
                        points = np.array(coords).T
                        if points.size == 0:
                            continue
                        if i == 4:  # Rudder
                            rotated_points = R_body @ (R_rudder @ points)
                        elif i in [5, 6]:  # Ailerons
                            rotated_points = R_body @ (R_aileron_left @ points if i == 5 else R_aileron_right @ points)
                        elif i in [7, 8]:  # Elevators
                            rotated_points = R_body @ (R_elevator @ points)
                        x_rot = rotated_points[0]
                        y_rot = rotated_points[1]
                        z_rot = rotated_points[2]

                    if i == 0:
                        self.plane_surfaces[i] = self.ax4.plot_surface(x_rot, y_rot, z_rot, color='lightgray', alpha=0.8)
                    elif i in [1, 2, 3]:
                        self.plane_surfaces[i] = self.ax4.plot_surface(x_rot, y_rot, z_rot, color='gray', alpha=0.7)
                    elif i == 4:
                        self.plane_surfaces[i] = art3d.Poly3DCollection([list(zip(x_rot, y_rot, z_rot))], color='red', alpha=0.9)
                        self.ax4.add_collection3d(self.plane_surfaces[i])
                    elif i in [5, 6]:
                        self.plane_surfaces[i] = art3d.Poly3DCollection([list(zip(x_rot, y_rot, z_rot))], color='blue', alpha=0.9)
                        self.ax4.add_collection3d(self.plane_surfaces[i])
                    elif i in [7, 8]:
                        self.plane_surfaces[i] = art3d.Poly3DCollection([list(zip(x_rot, y_rot, z_rot))], color='green', alpha=0.9)
                        self.ax4.add_collection3d(self.plane_surfaces[i])
                except Exception as e:
                    print(f"Error at surface {i}: {e}")
                    continue

    def start_simulation(self):
        self.running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')

    def stop_simulation(self):
        self.running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def reset_simulation(self):
        self.running = False
        self.t = [0]
        self.x = np.array([[0, 0, 0]]).T
        self.u = np.array([0, 0, 0])
        self.history = {'p': [0], 'q': [0], 'r': [0],
                        'phi': [0], 'theta': [0], 'psi': [0],
                        'delta_a': [0], 'delta_e': [0], 'delta_r': [0]}
        self.aileron_slider.set(0)
        self.elevator_slider.set(0)
        self.rudder_slider.set(0)
        self.update_plots()

    def simulate(self):
        while True:
            if self.running and self.t[-1] < self.t_max:
                delta_a = np.radians(self.aileron_slider.get())
                delta_e = np.radians(self.elevator_slider.get())
                delta_r = np.radians(self.rudder_slider.get())
                self.u = np.array([delta_a, delta_e, delta_r])

                t_span = [self.t[-1], self.t[-1] + self.dt]
                sol = odeint(flight_dynamics, self.x[:, -1], t_span, args=(self.u,))
                self.x = np.hstack((self.x, sol[-1, :].reshape(-1, 1)))

                self.t.append(self.t[-1] + self.dt)
                self.history['p'].append(self.x[0, -1])
                self.history['q'].append(self.x[1, -1])
                self.history['r'].append(self.x[2, -1])
                self.history['phi'].append(self.history['phi'][-1] + 0.5 * (self.history['p'][-2] + self.history['p'][-1]) * self.dt)
                self.history['theta'].append(self.history['theta'][-1] + 0.5 * (self.history['q'][-2] + self.history['q'][-1]) * self.dt)
                self.history['psi'].append(self.history['psi'][-1] + 0.5 * (self.history['r'][-2] + self.history['r'][-1]) * self.dt)
                self.history['delta_a'].append(np.degrees(delta_a))
                self.history['delta_e'].append(np.degrees(delta_e))
                self.history['delta_r'].append(np.degrees(delta_r))

                self.update_plots()

            time.sleep(self.dt / 2)

    def update_plots(self):
        start_idx = max(0, len(self.t) - self.max_points)

        self.line_da.set_data(self.t[start_idx:], self.history['delta_a'][start_idx:])
        self.line_de.set_data(self.t[start_idx:], self.history['delta_e'][start_idx:])
        self.line_dr.set_data(self.t[start_idx:], self.history['delta_r'][start_idx:])
        y_data1 = np.concatenate([self.history['delta_a'][start_idx:],
                                  self.history['delta_e'][start_idx:],
                                  self.history['delta_r'][start_idx:]])
        if len(y_data1) > 0:
            y_min1, y_max1 = np.min(y_data1), np.max(y_data1)
            margin1 = 0.1 * (y_max1 - y_min1) if y_max1 != y_min1 else 1
            self.ax1.set_ylim(y_min1 - margin1, y_max1 + margin1)

        self.line_p.set_data(self.t[start_idx:], np.degrees(self.history['p'][start_idx:]))
        self.line_q.set_data(self.t[start_idx:], np.degrees(self.history['q'][start_idx:]))
        self.line_r.set_data(self.t[start_idx:], np.degrees(self.history['r'][start_idx:]))
        y_data2 = np.concatenate([np.degrees(self.history['p'][start_idx:]),
                                  np.degrees(self.history['q'][start_idx:]),
                                  np.degrees(self.history['r'][start_idx:])])
        if len(y_data2) > 0:
            y_min2, y_max2 = np.min(y_data2), np.max(y_data2)
            margin2 = 0.1 * (y_max2 - y_min2) if y_max2 != y_min2 else 0.1
            self.ax2.set_ylim(y_min2 - margin2, y_max2 + margin2)

        self.line_phi.set_data(self.t[start_idx:], np.degrees(self.history['phi'][start_idx:]))
        self.line_theta.set_data(self.t[start_idx:], np.degrees(self.history['theta'][start_idx:]))
        self.line_psi.set_data(self.t[start_idx:], np.degrees(self.history['psi'][start_idx:]))
        y_data3 = np.concatenate([np.degrees(self.history['phi'][start_idx:]),
                                  np.degrees(self.history['theta'][start_idx:]),
                                  np.degrees(self.history['psi'][start_idx:])])
        if len(y_data3) > 0:
            y_min3, y_max3 = np.min(y_data3), np.max(y_data3)
            margin3 = 0.1 * (y_max3 - y_min3) if y_max3 != y_min3 else 1
            self.ax3.set_ylim(y_min3 - margin3, y_max3 + margin3)

        if len(self.history['phi']) > start_idx:
            phi = self.history['phi'][-1]
            theta = self.history['theta'][-1]
            psi = self.history['psi'][-1]
            self.update_plane_model(phi, theta, psi)

        self.canvas.draw()
        self.canvas.flush_events()

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = FlightSimulatorGUI(root)
    root.mainloop()