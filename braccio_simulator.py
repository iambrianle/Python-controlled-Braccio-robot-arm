import tkinter as tk
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class BraccioArm:
    def __init__(self, port=None, baudrate=115200):
        self.joints = {
            'base': 90,
            'shoulder': 90,
            'elbow': 90,
            'wrist_ver': 90,
            'wrist_rot': 90,
            'gripper': 73
        }
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        if self.port:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                time.sleep(2)
            except serial.SerialException as e:
                print(f"Error opening serial port: {e}")
                self.ser = None

    def set_joint(self, joint, angle):
        if joint in self.joints:
            self.joints[joint] = int(angle)
        else:
            print(f"Error: Invalid joint {joint}")

    def get_joint_angles(self):
        return self.joints

    def send_to_robot(self, speed=150):
        if self.ser:
            shoulder_angle = 180 - self.joints['shoulder']
            command = f"P{self.joints['base']},{shoulder_angle},{self.joints['elbow']},{self.joints['wrist_ver']},{self.joints['wrist_rot']},{self.joints['gripper']},{speed}\n"
            self.ser.write(command.encode())
            response = self.ser.readline().decode().strip()
            print(f"Robot response: {response}")
        else:
            print("Serial port not available. Cannot send command to robot.")

    def home(self):
        self.joints = {
            'base': 90,
            'shoulder': 90,
            'elbow': 90,
            'wrist_ver': 90,
            'wrist_rot': 90,
            'gripper': 73
        }
        self.send_to_robot()


class App:
    def __init__(self, root, arm):
        self.root = root
        self.arm = arm
        self.root.title("Braccio Arm Simulator")

        # Sliders and labels
        self.sliders = {}
        self.labels = {}

        controls_frame = tk.Frame(self.root)
        controls_frame.pack(side=tk.LEFT, padx=10, pady=10)

        row = 0
        for joint, angle in self.arm.get_joint_angles().items():
            label = tk.Label(controls_frame, text=f"{joint.capitalize()}: {angle}")
            label.grid(row=row, column=0, padx=10, pady=5)
            self.labels[joint] = label

            slider = tk.Scale(controls_frame, from_=0, to=180, orient=tk.HORIZONTAL,
                              command=lambda value, j=joint: self.update_joint(j, value))
            slider.set(angle)
            slider.grid(row=row, column=1, padx=10, pady=5)
            self.sliders[joint] = slider
            row += 1

        self.copy_button = tk.Button(controls_frame, text="Copy to Robot", command=self.copy_to_robot)
        self.copy_button.grid(row=row, column=0, columnspan=2, pady=10)

        self.home_button = tk.Button(controls_frame, text="Home", command=self.home_robot)
        self.home_button.grid(row=row+1, column=0, columnspan=2, pady=10)

        # 3D Plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.update_plot()

    def dh_matrix(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, angles):
        q1 = np.deg2rad(angles['base'])
        q2 = np.deg2rad(angles['shoulder'] - 90)
        q3 = np.deg2rad(angles['elbow'] - 90)
        q4 = np.deg2rad(angles['wrist_ver'] - 90)
        q5 = np.deg2rad(angles['wrist_rot'])

        # DH Parameters
        # j, theta, d, a, alpha, offset
        dh_params = [
            [q1, 71.5, 0, -np.pi/2, 0],
            [q2, 0, -125, 0, np.pi/2],
            [q3, 0, -125, 0, 0],
            [q4, 0, 0, np.pi/2, -np.pi/2],
            [q5, 192, 0, 0, 0]
        ]

        T = np.identity(4)
        points = [np.array([0, 0, 0])]
        for params in dh_params:
            T_i = self.dh_matrix(params[0] + params[4], params[1], params[2], params[3])
            T = np.dot(T, T_i)
            points.append(T[:3, 3])

        return points

    def update_plot(self):
        points = self.forward_kinematics(self.arm.get_joint_angles())
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        z = [p[2] for p in points]

        self.ax.clear()
        self.ax.plot(x, y, z, 'o-')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim([-300, 300])
        self.ax.set_ylim([-300, 300])
        self.ax.set_zlim([0, 400])
        self.canvas.draw()

    def update_joint(self, joint, angle):
        self.arm.set_joint(joint, angle)
        self.labels[joint].config(text=f"{joint.capitalize()}: {angle}")
        self.update_plot()

    def copy_to_robot(self):
        self.arm.send_to_robot()

    def home_robot(self):
        self.arm.home()
        for joint, angle in self.arm.get_joint_angles().items():
            self.sliders[joint].set(angle)
        self.update_plot()


if __name__ == '__main__':
    SERIAL_PORT = '/dev/cu.usbmodem1301'

    root = tk.Tk()
    arm = BraccioArm(port=SERIAL_PORT)
    app = App(root, arm)
    root.mainloop()
