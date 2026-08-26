#!/usr/bin/env python3
"""
Self-contained Braccio pick-and-place script with interactive calibration.

Detects a workspace defined by 4 ArUco corner markers (DICT_4X4_50, IDs 1-4)
and picks up a target tagged with a DICT_6X6_50 marker.

Inverse kinematics and pickup sequence ported from the original project's
solverNNA.move_to_position_cart and braccio_control_python.pick_up so the
arm uses the same joint angles that were tuned to the official Arduino
Braccio's mechanical assembly.

Run:
    python3 braccio_pick.py

Calibration is saved next to the script in 'braccio_calib.json' and reloaded
on next run, so you only have to dial it in once.

Workspace coordinates: origin at marker 1, x toward marker 4, y toward marker 2.

Mouse on the WORKSPACE window:
    Left click          Set robot base position to that point
    Shift+Left click    Set robot heading to point AT that location from base
    G then click        Move arm above the clicked workspace point (verify calibration)

Keys:
    H   home arm
    T   safe forward-leaning test pose
    O   open gripper          C   close gripper
    I   invert gripper open/close values (fixes reversed grip)
    M   move above target only (preview, does not descend or grip)
    P   pick up the target and drop at the configured drop location
    G   arm-the next workspace click to "go above clicked point"
    F   show/hide tuning window (manual dashboard)
    [   nudge heading -5 deg  ]   nudge heading +5 deg
    ,   nudge base offset -5  .   nudge base offset +5
    -   shrink workspace 5mm  =   grow workspace 5mm
    R   show/hide reach overlay
    S   save calibration      L   reload calibration
    D   print full state to terminal
    Q / Esc   quit

Tuning window (drag sliders, mouse-wheel for 0.1°):
    IK knobs, plus a zoomable base-backlash editor.
    Set the angle range and step (2/5/10/20°) to put more
    anchors where you need them. Right-click an anchor to remove it.
    Changes apply to the next move; press [P], [M] or click-to-go to test.

Environment:
    BRACCIO_SERIAL_PORT  default /dev/cu.usbmodem2101
    BRACCIO_CAMERA       URL or webcam index (default IP cam below)
    BRACCIO_GRIPPER_INVERT=1  if your gripper open/close is reversed at startup
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import serial


# ----------------------------- configuration -----------------------------

SERIAL_PORT = os.environ.get("BRACCIO_SERIAL_PORT", "/dev/cu.usbmodem2101")
SERIAL_BAUD = 115200

_CAMERA_RAW = os.environ.get("BRACCIO_CAMERA", "http://172.20.10.3:8080/video")
try:
    CAMERA_SOURCE: object = int(_CAMERA_RAW)
except ValueError:
    CAMERA_SOURCE = _CAMERA_RAW

# arm geometry (mm) used by the original solverNNA
L_BASE = 71.5
L1 = 125.0
L2 = 125.0
L3 = 60.0 + 132.0   # wrist + gripper, fully extended (192) — original convention

SHOULDER_MIN, SHOULDER_MAX = 15, 165

# Backlash table: sparse anchors (turn_deg -> extra_deg). The dashboard lets
# you zoom into any angle range and drop anchors as densely as 2°.
BACKLASH_ANGLE_MIN = 0
BACKLASH_ANGLE_MAX = 180
BACKLASH_COMP_MAX = 30.0
BACKLASH_STEPS = (2, 5, 10, 20)
BACKLASH_MAX_ROWS = 16
_OLD_BACKLASH_RE = re.compile(r"^base_backlash_(ccw|cw)_at_(\d+)deg$")


def _as_backlash_dict(raw) -> Dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, float] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out

DEFAULT_GRIPPER_OPEN = 73
DEFAULT_GRIPPER_CLOSED = 15
if int(os.environ.get("BRACCIO_GRIPPER_INVERT", "0")):
    DEFAULT_GRIPPER_OPEN, DEFAULT_GRIPPER_CLOSED = (
        DEFAULT_GRIPPER_CLOSED, DEFAULT_GRIPPER_OPEN)

# Pick-up sequence heights — matches the original braccio_control_python.pick_up
APPROACH_Z = 100.0
PICK_Z = 10.0
LIFT_Z = 200.0
# Drop position in robot frame (x, y, z)
DROP_TARGET_ROBOT = (145.2, 37.3, 120.0)

TRAVEL_SPEED = 80
PICK_SPEED = 30
# Inter-step delay used by the original code to wait for moves to finish
STEP_DELAY = 2.0

CORNER_DICT = cv2.aruco.DICT_4X4_50
TARGET_DICT = cv2.aruco.DICT_6X6_50

WIN_LIVE = "Braccio - camera"
WIN_WORK = "Braccio - workspace"
WIN_TUNE = "Braccio - tuning"
PANEL_W = 380
TUNE_W = 740

# warped workspace render size in pixels
WARP_PX_W = 600
WARP_PX_H = 800

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "braccio_calib.json")


# ----------------------------- calibration -----------------------------

@dataclass
class Calibration:
    # workspace
    workspace_w_mm: float = 280.0
    workspace_h_mm: float = 360.0
    # robot placement on workspace
    base_x_mm: float = 30.0
    base_y_mm: float = 330.0
    heading_deg: float = 315.0
    base_offset_deg: float = 0.0
    # IK tuning (defaults match the original solverNNA constants)
    radial_compensation: float = 1.02
    # Two-point linear reach calibration. Each value is "extra mm of reach the
    # IK should add when the target is at that horizontal distance from the
    # base." The IK linearly interpolates / extrapolates between the two.
    # Tune them independently: dial reach_at_100mm with a close target, then
    # reach_at_300mm with a far target.
    reach_at_100mm: float = 0.0
    reach_at_300mm: float = 0.0
    z_offset_mm: float = 15.0
    elbow_offset_deg: float = 5.0
    wrist_offset_deg: float = 5.0
    # Sparse backlash table: extra degrees added when the base servo turns
    # this many degrees in the given direction. Keys are integers (degrees).
    # Linearly interpolated between anchors; scales from 0 below the first
    # anchor; clamps to the last anchor above it.
    base_backlash_ccw: Dict[int, float] = field(default_factory=dict)
    base_backlash_cw: Dict[int, float] = field(default_factory=dict)
    # gripper servo positions
    gripper_open: int = DEFAULT_GRIPPER_OPEN
    gripper_closed: int = DEFAULT_GRIPPER_CLOSED

    def load(self) -> bool:
        try:
            with open(CALIB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            ccw = dict(self.base_backlash_ccw)
            cw = dict(self.base_backlash_cw)
            for k, v in data.items():
                m = _OLD_BACKLASH_RE.match(str(k))
                if m:
                    dest = ccw if m.group(1) == "ccw" else cw
                    dest[int(m.group(2))] = float(v)
                    continue
                if k in ("base_backlash_ccw", "base_backlash_cw"):
                    dest = ccw if k.endswith("ccw") else cw
                    dest.update(_as_backlash_dict(v))
                    continue
                if hasattr(self, k):
                    cur = getattr(self, k)
                    cls = type(cur)
                    try:
                        setattr(self, k, cls(v))
                    except (TypeError, ValueError):
                        pass
            self.base_backlash_ccw = ccw
            self.base_backlash_cw = cw
            print(f"[INFO] Loaded calibration from {CALIB_PATH}")
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not load calibration: {exc}")
            return False

    def save(self, quiet: bool = False) -> None:
        try:
            data = asdict(self)
            for name in ("base_backlash_ccw", "base_backlash_cw"):
                d = getattr(self, name)
                data[name] = {
                    str(int(k)): round(float(v), 2)
                    for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))
                }
            with open(CALIB_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if not quiet:
                print(f"[INFO] Saved calibration to {CALIB_PATH}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not save calibration: {exc}")


# ----------------------------- arm control -----------------------------

class Braccio:
    def __init__(self, port: str = SERIAL_PORT, baud: int = SERIAL_BAUD) -> None:
        print(f"[INFO] Opening serial port {port} @ {baud} ...")
        self.serial = serial.Serial(port, baud, timeout=2)
        time.sleep(2.0)
        while self.serial.in_waiting:
            self.serial.read(self.serial.in_waiting)
        self.lock = threading.Lock()
        self.last_pose: Tuple[int, int, int, int, int, int] = \
            (90, 90, 90, 90, 90, DEFAULT_GRIPPER_OPEN)
        # Last *uncompensated* base servo we asked for, used to decide the
        # direction for backlash compensation on the next move.
        self.prev_base_target: Optional[float] = None
        # Backlash compensation tables: list of (turn_deg, extra_deg) pairs,
        # set by the main loop from Calibration.
        self.backlash_ccw_table: List[Tuple[float, float]] = []
        self.backlash_cw_table: List[Tuple[float, float]] = []
        # Gripper open/closed servo values (set from Calibration).
        self.gripper_open: int = DEFAULT_GRIPPER_OPEN
        self.gripper_closed: int = DEFAULT_GRIPPER_CLOSED
        print("[INFO] Connected.")

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> int:
        return int(round(max(lo, min(hi, v))))

    @staticmethod
    def _interp_backlash(abs_delta: float,
                         table: List[Tuple[float, float]]) -> float:
        """Linear interpolation between adjacent (turn_deg, extra_deg) anchors.
        Below the smallest anchor: scale linearly from origin.
        Above the largest anchor: clamp to the largest anchor's value."""
        if not table or abs_delta <= 0.0:
            return 0.0
        if table[0][0] > 0 and abs_delta < table[0][0]:
            return table[0][1] * (abs_delta / table[0][0])
        if abs_delta >= table[-1][0]:
            return table[-1][1]
        for i in range(len(table) - 1):
            a0, v0 = table[i]
            a1, v1 = table[i + 1]
            if a0 <= abs_delta <= a1:
                if a1 == a0:
                    return v0
                t = (abs_delta - a0) / (a1 - a0)
                return v0 + t * (v1 - v0)
        return 0.0

    def _apply_backlash(self, base: float) -> float:
        if self.prev_base_target is None:
            return base
        delta = base - self.prev_base_target
        if abs(delta) < 1.0:
            return base
        table = self.backlash_ccw_table if delta > 0 else self.backlash_cw_table
        extra = self._interp_backlash(abs(delta), table)
        if extra < 0.0:
            extra = 0.0
        return base + extra if delta > 0 else base - extra

    def send(self, base: float, shoulder: float, elbow: float, wrist: float,
             wrist_rot: float, gripper: float, speed: int = TRAVEL_SPEED
             ) -> Tuple[int, int, int, int, int, int]:
        compensated_base = self._apply_backlash(base)
        # remember the uncompensated target for the next call
        self.prev_base_target = float(base)
        b = self._clamp(compensated_base, 0, 180)
        s = self._clamp(shoulder, SHOULDER_MIN, SHOULDER_MAX)
        e = self._clamp(elbow, 0, 180)
        w = self._clamp(wrist, 0, 180)
        wr = self._clamp(wrist_rot, 0, 180)
        g = self._clamp(gripper, 0, 90)
        sp = self._clamp(speed, 10, 200)
        cmd = f"P{b},{s},{e},{w},{wr},{g},{sp}\n"
        with self.lock:
            self.serial.write(cmd.encode("ascii"))
            self.serial.flush()
        self.last_pose = (b, s, e, w, wr, g)
        return self.last_pose

    def home(self) -> None:
        b = self.last_pose[0]
        self.send(b, 90, 90, 90, 90, self.gripper_open, speed=60)

    def safe_test_pose(self) -> None:
        self.send(90, 60, 110, 100, 90, self.gripper_open, speed=60)

    def open_gripper(self) -> None:
        b, s, e, w, wr, _ = self.last_pose
        self.send(b, s, e, w, wr, self.gripper_open, speed=60)

    def close_gripper(self) -> None:
        b, s, e, w, wr, _ = self.last_pose
        self.send(b, s, e, w, wr, self.gripper_closed, speed=60)


# ----------------------------- inverse kinematics -----------------------------
# Ported from solverNNA.move_to_position_cart in the original project. All the
# tuning knobs (radial compensation, z offset, elbow/wrist offsets, etc.) live
# on the Calibration so they can be adjusted live from the tuning window.


def solve_ik(x: float, y: float, z: float, calib: "Calibration"
             ) -> Optional[Tuple[float, float, float, float, float]]:
    """Original-style IK using the live calibration values.
    Returns (base_servo, shoulder, elbow, wrist, phi_deg). phi_deg is 0."""
    z_eff = z + calib.z_offset_mm
    r_hor = math.hypot(x, y)
    if r_hor < 1e-6 and abs(z_eff - L_BASE) < 1e-6:
        return None
    # Two-point linear reach correction. (100mm, e_close) -> (300mm, e_far),
    # interpolated for r_hor in between and extrapolated outside that range.
    # Setting both to 0 disables the correction.
    e_close = calib.reach_at_100mm
    e_far = calib.reach_at_300mm
    slope = (e_far - e_close) / 200.0  # per mm
    distance_correction = e_close + slope * (r_hor - 100.0)
    r = (math.hypot(r_hor, z_eff - L_BASE) * calib.radial_compensation
         + distance_correction)

    # Use atan2 (not atan) so the IK knows the target is behind the robot
    # rather than folding y<0 onto the y>0 hemisphere. For valid forward
    # targets (y >= 0) atan2 returns the same values as atan(x/y), so this
    # preserves the original IK exactly. For y<0 the resulting base_servo
    # falls outside [0, 180] and the bounds check below rejects the pose
    # cleanly instead of pointing the arm in the wrong direction.
    theta_base = 90.0 - math.degrees(math.atan2(x, y))

    arg = (r - L2) / (L1 + L3)
    if not (-1.0 <= arg <= 1.0):
        return None
    alpha1 = math.acos(arg)
    alpha1_deg = math.degrees(alpha1)

    sin_a3 = math.sin(alpha1) * (L3 - L1) / L2
    if abs(sin_a3) > 1.0:
        sin_a3 = max(-1.0, min(1.0, sin_a3))
    alpha3 = math.asin(sin_a3)
    theta_shoulder = alpha1_deg
    theta_elbow = (90.0 - alpha1_deg) + math.degrees(alpha3)
    theta_wrist = (90.0 - alpha1_deg) - math.degrees(alpha3)

    if theta_wrist <= 0.0:
        sin_inner = (L3 - L1) / r if r > 1e-6 else 0.0
        if abs(sin_inner) > 1.0:
            sin_inner = max(-1.0, min(1.0, sin_inner))
        theta_shoulder = math.degrees(alpha1 + math.asin(sin_inner))
        theta_elbow = 90.0 - alpha1_deg
        theta_wrist = 90.0 - alpha1_deg

    if abs(z_eff - L_BASE) > 1e-6 and r > 1e-6:
        theta_shoulder += math.degrees(math.atan((z_eff - L_BASE) / r))

    theta_elbow += calib.elbow_offset_deg
    theta_wrist += calib.wrist_offset_deg

    base_servo = 180.0 - theta_base + calib.base_offset_deg

    # Clamp all servo angles — send() enforces the same limits in hardware.
    # Only the radial geometry check (arg) above is a hard rejection.
    base_servo     = max(0.0,              min(180.0,              base_servo))
    theta_shoulder = max(float(SHOULDER_MIN), min(float(SHOULDER_MAX), theta_shoulder))
    theta_elbow    = max(0.0,              min(180.0,              theta_elbow))
    theta_wrist    = max(0.0,              min(180.0,              theta_wrist))

    return base_servo, theta_shoulder, theta_elbow, theta_wrist, 0.0


def _ik_at_tilt(x: float, y: float, z: float, phi_deg: float,
                calib: "Calibration"
                ) -> Optional[Tuple[float, float, float, float]]:
    """Compatibility shim — the original IK has no tilt parameter."""
    sol = solve_ik(x, y, z, calib)
    if sol is None:
        return None
    return sol[:4]


# ----------------------------- coordinate transforms -----------------------------

def workspace_to_robot(x_w: float, y_w: float, calib: Calibration
                       ) -> Tuple[float, float]:
    """Map workspace mm (origin marker 1) to robot frame mm (x right, y forward)."""
    dx = x_w - calib.base_x_mm
    dy = y_w - calib.base_y_mm
    h = math.radians(calib.heading_deg)
    fwd_x, fwd_y = math.cos(h), math.sin(h)
    right_x, right_y = math.sin(h), -math.cos(h)
    y_r = dx * fwd_x + dy * fwd_y
    x_r = dx * right_x + dy * right_y
    return x_r, y_r


def warped_pixel_to_workspace_mm(px: float, py: float, w: int, h: int,
                                 calib: Calibration) -> Tuple[float, float]:
    x_mm = (px / max(1, w - 1)) * calib.workspace_w_mm
    y_mm = ((h - 1 - py) / max(1, h - 1)) * calib.workspace_h_mm
    return x_mm, y_mm


def workspace_mm_to_warped_pixel(x_mm: float, y_mm: float, w: int, h: int,
                                 calib: Calibration) -> Tuple[int, int]:
    px = int(round((x_mm / max(1.0, calib.workspace_w_mm)) * (w - 1)))
    py = int(round((h - 1) -
                   (y_mm / max(1.0, calib.workspace_h_mm)) * (h - 1)))
    return px, py


# ----------------------------- vision -----------------------------

def make_detector(dict_id: int) -> "cv2.aruco.ArucoDetector":
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dict_id),
        cv2.aruco.DetectorParameters(),
    )


def warp_workspace(frame: np.ndarray, corners_by_id: dict
                   ) -> Optional[np.ndarray]:
    """Warp the rectangle (markers 1..4) to a top-down view.
        marker 1 -> bottom-left  (0, H-1)
        marker 2 -> top-left     (0, 0)
        marker 3 -> top-right    (W-1, 0)
        marker 4 -> bottom-right (W-1, H-1)
    """
    if any(corners_by_id.get(i) is None for i in (1, 2, 3, 4)):
        return None
    src = np.array(
        [corners_by_id[1], corners_by_id[2], corners_by_id[3], corners_by_id[4]],
        dtype="float32",
    )
    dst = np.array(
        [[0, WARP_PX_H - 1], [0, 0],
         [WARP_PX_W - 1, 0], [WARP_PX_W - 1, WARP_PX_H - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (WARP_PX_W, WARP_PX_H))


# ----------------------------- UI helpers -----------------------------

def render_panel(width: int, lines) -> np.ndarray:
    height = max(560, len(lines) * 24 + 50)
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.putText(panel, "BRACCIO PICK & PLACE", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(panel, (16, 42), (width - 16, 42), (80, 80, 80), 1)
    y = 70
    for text, color in lines:
        cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, color, 1, cv2.LINE_AA)
        y += 22
    return panel


def compose(canvas_left: np.ndarray, panel: np.ndarray) -> np.ndarray:
    h = max(canvas_left.shape[0], panel.shape[0])
    w = canvas_left.shape[1] + panel.shape[1]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[: canvas_left.shape[0], : canvas_left.shape[1]] = canvas_left
    out[: panel.shape[0],
        canvas_left.shape[1]:canvas_left.shape[1] + panel.shape[1]] = panel
    return out


def _draw_banner(view: np.ndarray, lines, color=(0, 220, 255)) -> None:
    h, w = view.shape[:2]
    bar_h = 30 * len(lines) + 20
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, view, 0.45, 0, dst=view)
    y = 30
    for line in lines:
        cv2.putText(view, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        y += 28


def draw_workspace_overlay(view: np.ndarray, calib: Calibration,
                           target_ws: Optional[Tuple[float, float]],
                           setup_step: int = 2,
                           goto_armed: bool = False) -> None:
    h, w = view.shape[:2]
    bx, by = workspace_mm_to_warped_pixel(
        calib.base_x_mm, calib.base_y_mm, w, h, calib)
    if 0 <= bx < w and 0 <= by < h:
        cv2.circle(view, (bx, by), 14, (0, 255, 255), 2)
        cv2.circle(view, (bx, by), 3, (0, 255, 255), -1)
        cv2.putText(view, "robot", (bx + 16, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        h_rad = math.radians(calib.heading_deg)
        tip_x_mm = calib.base_x_mm + 80.0 * math.cos(h_rad)
        tip_y_mm = calib.base_y_mm + 80.0 * math.sin(h_rad)
        tx, ty = workspace_mm_to_warped_pixel(tip_x_mm, tip_y_mm, w, h, calib)
        cv2.arrowedLine(view, (bx, by), (tx, ty), (0, 255, 255), 2,
                        tipLength=0.25)

    if target_ws is not None:
        tx, ty = workspace_mm_to_warped_pixel(
            target_ws[0], target_ws[1], w, h, calib)
        cv2.line(view, (bx, by), (tx, ty), (255, 200, 80), 1)

    cv2.circle(view, (5, h - 5), 6, (255, 200, 0), -1)
    cv2.putText(view, "1 (0,0)", (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    if goto_armed:
        _draw_banner(view, ["CLICK A POINT - the arm will go above it"],
                     color=(80, 220, 255))
    elif setup_step == 0:
        _draw_banner(view,
                     ["STEP 1: click on the robot's base in this view",
                      "(the heading auto-aims toward the workspace center)"],
                     color=(0, 255, 255))
    elif setup_step == 1:
        _draw_banner(view,
                     ["STEP 2: SHIFT+click in the direction the arm faces",
                      "press [ ] to nudge - press S to save when it looks right"],
                     color=(0, 255, 255))


# ----------------------------- tuning window -----------------------------

# (field, label, min, max, precision, format)
IK_SPECS = [
    ("reach_at_100mm",      "reach @100mm",  -50.0, 200.0, 0.5,   "{:+.1f}"),
    ("reach_at_300mm",      "reach @300mm",  -50.0, 300.0, 0.5,   "{:+.1f}"),
    ("z_offset_mm",         "z offset mm",   -20.0,  40.0, 0.5,   "{:+.1f}"),
    ("elbow_offset_deg",    "elbow deg",     -30.0,  30.0, 0.1,   "{:+.1f}"),
    ("wrist_offset_deg",    "wrist deg",     -30.0,  30.0, 0.1,   "{:+.1f}"),
    ("radial_compensation", "radial *",       0.95,  1.20, 0.001, "{:.3f}"),
    ("base_offset_deg",     "base off deg",  -45.0,  45.0, 0.1,   "{:+.1f}"),
]

_COL_BG = (28, 28, 28)
_COL_TRACK = (52, 52, 52)
_COL_TEXT = (235, 235, 235)
_COL_MUTED = (150, 150, 150)
_COL_ACCENT = (0, 210, 255)
_COL_IK = (90, 170, 80)
_COL_CCW = (220, 180, 70)
_COL_CW = (70, 140, 255)
_COL_KNOB = (240, 240, 240)
_COL_BTN = (58, 58, 58)
_COL_BTN_ON = (70, 110, 40)

_tune_hits: list = []
_tune_drag: Optional[dict] = None


def _snap(val: float, prec: float, lo: float, hi: float) -> float:
    if prec <= 0:
        prec = 0.1
    v = round(val / prec) * prec
    if abs(v) < prec * 0.5:
        v = 0.0
    return max(lo, min(hi, float(v)))


def _val_from_x(x: int, x0: int, x1: int, lo: float, hi: float,
                prec: float) -> float:
    span = max(1, x1 - x0)
    t = (x - x0) / span
    return _snap(lo + t * (hi - lo), prec, lo, hi)


def _hit_at(x: int, y: int) -> Optional[dict]:
    for h in _tune_hits:
        if h["x0"] <= x <= h["x1"] and h["y0"] <= y <= h["y1"]:
            return h
    return None


def _visible_backlash_angles(state: "State") -> List[int]:
    _clamp_backlash_window(state)
    lo, hi, step = (state.backlash_range_lo, state.backlash_range_hi,
                    state.backlash_step)
    return list(range(lo, hi + 1, step))


def _clamp_backlash_window(state: "State") -> None:
    step = int(state.backlash_step)
    if step not in BACKLASH_STEPS:
        step = 10
        state.backlash_step = step
    lo = int(round(state.backlash_range_lo / step) * step)
    hi = int(round(state.backlash_range_hi / step) * step)
    lo = max(BACKLASH_ANGLE_MIN, min(BACKLASH_ANGLE_MAX, lo))
    hi = max(BACKLASH_ANGLE_MIN, min(BACKLASH_ANGLE_MAX, hi))
    if hi <= lo:
        hi = min(BACKLASH_ANGLE_MAX, lo + step)
    max_span = step * (BACKLASH_MAX_ROWS - 1)
    if hi - lo > max_span:
        mid = (lo + hi) / 2.0
        lo = int(round((mid - max_span / 2.0) / step) * step)
        hi = lo + max_span
        if hi > BACKLASH_ANGLE_MAX:
            hi = int(round(BACKLASH_ANGLE_MAX / step) * step)
            lo = max(BACKLASH_ANGLE_MIN, hi - max_span)
        if lo < BACKLASH_ANGLE_MIN:
            lo = BACKLASH_ANGLE_MIN
            hi = min(BACKLASH_ANGLE_MAX, lo + max_span)
    state.backlash_range_lo = int(lo)
    state.backlash_range_hi = int(hi)


def _center_backlash_window(state: "State", angle: float) -> None:
    span = max(state.backlash_step,
               state.backlash_range_hi - state.backlash_range_lo)
    lo = angle - span / 2.0
    hi = angle + span / 2.0
    state.backlash_range_lo = int(round(lo))
    state.backlash_range_hi = int(round(hi))
    _clamp_backlash_window(state)


def _set_backlash_span(state: "State", span: int, step: int) -> None:
    mid = (state.backlash_range_lo + state.backlash_range_hi) / 2.0
    state.backlash_step = step
    state.backlash_range_lo = int(round(mid - span / 2.0))
    state.backlash_range_hi = int(round(mid + span / 2.0))
    _clamp_backlash_window(state)


def _backlash_dict(calib: "Calibration", direction: str) -> Dict[int, float]:
    return (calib.base_backlash_ccw if direction == "ccw"
            else calib.base_backlash_cw)


def _apply_tune_drag(state: "State", x: int) -> None:
    drag = _tune_drag
    if drag is None:
        return
    val = _val_from_x(x, drag["x0"], drag["x1"],
                      drag["lo"], drag["hi"], drag["prec"])
    kind = drag["kind"]
    if kind == "ik":
        setattr(state.calib, drag["field"], val)
    elif kind == "backlash":
        table = _backlash_dict(state.calib, drag["direction"])
        table[int(drag["angle"])] = val
    elif kind == "range_lo":
        state.backlash_range_lo = int(round(val))
        if state.backlash_range_lo >= state.backlash_range_hi:
            state.backlash_range_lo = state.backlash_range_hi - state.backlash_step
        _clamp_backlash_window(state)
    elif kind == "range_hi":
        state.backlash_range_hi = int(round(val))
        if state.backlash_range_hi <= state.backlash_range_lo:
            state.backlash_range_hi = state.backlash_range_lo + state.backlash_step
        _clamp_backlash_window(state)


def _nudge_hit(state: "State", hit: dict, sign: int) -> None:
    kind = hit["kind"]
    if kind == "ik":
        cur = float(getattr(state.calib, hit["field"]))
        setattr(state.calib, hit["field"],
                _snap(cur + sign * hit["prec"], hit["prec"],
                      hit["lo"], hit["hi"]))
        state.calib.save(quiet=True)
    elif kind == "backlash":
        table = _backlash_dict(state.calib, hit["direction"])
        ang = int(hit["angle"])
        cur = backlash_at(table, ang)
        table[ang] = _snap(cur + sign * hit["prec"], hit["prec"],
                           0.0, BACKLASH_COMP_MAX)
        state.calib.save(quiet=True)
    elif kind == "nudge":
        inner = dict(hit)
        inner["kind"] = hit["target"]
        _nudge_hit(state, inner, hit["sign"])


def _on_tune_mouse(event: int, x: int, y: int, flags: int,
                   state: "State") -> None:
    global _tune_drag
    if event == cv2.EVENT_LBUTTONDOWN:
        hit = _hit_at(x, y)
        if hit is None:
            return
        kind = hit["kind"]
        if kind == "step":
            state.backlash_step = int(hit["step"])
            _clamp_backlash_window(state)
        elif kind == "preset":
            name = hit["preset"]
            if name == "wide":
                state.backlash_step = 20
                state.backlash_range_lo = 0
                state.backlash_range_hi = 160
            elif name == "zoom":
                _set_backlash_span(state, 40, 5)
            elif name == "fine":
                _set_backlash_span(state, 20, 2)
            _clamp_backlash_window(state)
        elif kind == "graph":
            ang = _val_from_x(x, hit["x0"], hit["x1"],
                              BACKLASH_ANGLE_MIN, BACKLASH_ANGLE_MAX, 1.0)
            _center_backlash_window(state, ang)
        elif kind == "nudge":
            _nudge_hit(state, hit, hit["sign"])
        elif kind in ("ik", "backlash", "range_lo", "range_hi"):
            _tune_drag = {
                "kind": kind,
                "x0": hit["x0"], "x1": hit["x1"],
                "lo": hit["lo"], "hi": hit["hi"],
                "prec": hit["prec"],
                "field": hit.get("field", ""),
                "angle": hit.get("angle", 0),
                "direction": hit.get("direction", ""),
            }
            _apply_tune_drag(state, x)
    elif event == cv2.EVENT_RBUTTONDOWN:
        hit = _hit_at(x, y)
        if hit is None or hit["kind"] != "backlash":
            return
        table = _backlash_dict(state.calib, hit["direction"])
        table.pop(int(hit["angle"]), None)
        state.calib.save(quiet=True)
    elif event == cv2.EVENT_MOUSEMOVE:
        if _tune_drag is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            _apply_tune_drag(state, x)
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        if _tune_drag is not None:
            _tune_drag = None
            state.calib.save(quiet=True)
    elif getattr(cv2, "EVENT_MOUSEWHEEL", None) is not None and \
            event == cv2.EVENT_MOUSEWHEEL:
        hit = _hit_at(x, y)
        if hit is None:
            return
        if hasattr(cv2, "getMouseWheelDelta"):
            delta = cv2.getMouseWheelDelta(flags)
        else:
            delta = flags >> 16
            if delta >= 1 << 15:
                delta -= 1 << 16
        sign = 1 if delta > 0 else -1
        if hit["kind"] in ("ik", "backlash"):
            _nudge_hit(state, hit, sign)
        elif hit["kind"] == "nudge":
            _nudge_hit(state, hit, hit["sign"])


def _put(img: np.ndarray, text: str, xy: Tuple[int, int],
         color=_COL_TEXT, scale: float = 0.42, thick: int = 1) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def _fill_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               color) -> None:
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)


def _draw_button(img: np.ndarray, x: int, y: int, w: int, h: int,
                 label: str, on: bool) -> Tuple[int, int, int, int]:
    color = _COL_BTN_ON if on else _COL_BTN
    _fill_rect(img, x, y, x + w, y + h, color)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
    tw = int(len(label) * 7 * 0.85)
    _put(img, label, (x + max(4, (w - tw) // 2), y + h - 7),
         (255, 255, 255) if on else _COL_TEXT, 0.38)
    return x, y, x + w, y + h


def _draw_track(img: np.ndarray, x: int, y: int, w: int, h: int,
                val: float, lo: float, hi: float, fill) -> Tuple[int, int, int, int]:
    _fill_rect(img, x, y, x + w, y + h, _COL_TRACK)
    span = hi - lo
    t = 0.0 if span == 0 else (val - lo) / span
    t = max(0.0, min(1.0, t))
    fw = int(round(t * w))
    if fw > 0:
        _fill_rect(img, x, y, x + fw, y + h, fill)
    kx = x + fw
    cv2.circle(img, (kx, y + h // 2), max(5, h // 2 + 2), _COL_KNOB, -1)
    return x, y - 6, x + w, y + h + 6


def make_tuning_window(state: "State") -> None:
    global _tune_drag
    _tune_drag = None
    cv2.namedWindow(WIN_TUNE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TUNE, TUNE_W, 980)
    cv2.setMouseCallback(
        WIN_TUNE,
        lambda e, x, y, f, _p: _on_tune_mouse(e, x, y, f, state),
    )


def update_tuning_window(state: "State") -> None:
    global _tune_hits
    _clamp_backlash_window(state)
    angles = _visible_backlash_angles(state)
    n_rows = len(angles)
    height = 52 + 30 * len(IK_SPECS) + 210 + 28 * n_rows + 36
    canvas = np.full((height, TUNE_W, 3), _COL_BG, dtype=np.uint8)
    hits: list = []
    c = state.calib
    y = 8
    _put(canvas, "MANUAL TUNING", (14, y + 16), _COL_TEXT, 0.58, 1)
    _put(canvas, "drag  |  wheel or +/-  |  right-click removes a backlash anchor",
         (200, y + 16), _COL_MUTED, 0.36)
    y += 28
    cv2.line(canvas, (14, y), (TUNE_W - 14, y), (70, 70, 70), 1)
    y += 10

    # --- IK sliders ---
    track_x = 150
    track_w = 470
    for field, label, lo, hi, prec, fmt in IK_SPECS:
        val = float(getattr(c, field))
        _put(canvas, label, (14, y + 16), _COL_TEXT, 0.42)
        bx0, by0, bx1, by1 = _draw_button(canvas, track_x - 22, y + 2, 18, 18, "-", False)
        hits.append({"kind": "nudge", "target": "ik", "sign": -1,
                     "x0": bx0, "y0": by0, "x1": bx1, "y1": by1,
                     "field": field, "lo": lo, "hi": hi, "prec": prec})
        hx0, hy0, hx1, hy1 = _draw_track(
            canvas, track_x, y + 6, track_w, 10, val, lo, hi, _COL_IK)
        hits.append({"kind": "ik", "field": field, "lo": lo, "hi": hi,
                     "prec": prec, "x0": hx0, "y0": hy0, "x1": hx1, "y1": hy1})
        px0, py0, px1, py1 = _draw_button(
            canvas, track_x + track_w + 6, y + 2, 18, 18, "+", False)
        hits.append({"kind": "nudge", "target": "ik", "sign": 1,
                     "x0": px0, "y0": py0, "x1": px1, "y1": py1,
                     "field": field, "lo": lo, "hi": hi, "prec": prec})
        _put(canvas, fmt.format(val), (track_x + track_w + 30, y + 16),
             _COL_ACCENT, 0.42)
        y += 28

    y += 6
    cv2.line(canvas, (14, y), (TUNE_W - 14, y), (70, 70, 70), 1)
    y += 20
    _put(canvas, "BASE BACKLASH  —  angle range", (14, y), _COL_ACCENT, 0.50)
    y += 10

    # range sliders
    _put(canvas, "from", (14, y + 16), _COL_MUTED, 0.38)
    hx0, hy0, hx1, hy1 = _draw_track(
        canvas, 58, y + 6, 240, 10,
        float(state.backlash_range_lo),
        float(BACKLASH_ANGLE_MIN), float(BACKLASH_ANGLE_MAX), _COL_CCW)
    hits.append({"kind": "range_lo", "lo": float(BACKLASH_ANGLE_MIN),
                 "hi": float(BACKLASH_ANGLE_MAX), "prec": float(state.backlash_step),
                 "x0": hx0, "y0": hy0, "x1": hx1, "y1": hy1})
    _put(canvas, f"{state.backlash_range_lo:.0f}°", (304, y + 16), _COL_CCW, 0.42)

    _put(canvas, "to", (360, y + 16), _COL_MUTED, 0.38)
    hx0, hy0, hx1, hy1 = _draw_track(
        canvas, 390, y + 6, 240, 10,
        float(state.backlash_range_hi),
        float(BACKLASH_ANGLE_MIN), float(BACKLASH_ANGLE_MAX), _COL_CW)
    hits.append({"kind": "range_hi", "lo": float(BACKLASH_ANGLE_MIN),
                 "hi": float(BACKLASH_ANGLE_MAX), "prec": float(state.backlash_step),
                 "x0": hx0, "y0": hy0, "x1": hx1, "y1": hy1})
    _put(canvas, f"{state.backlash_range_hi:.0f}°", (638, y + 16), _COL_CW, 0.42)
    y += 28

    _put(canvas, "step", (14, y + 16), _COL_MUTED, 0.38)
    bx = 58
    for step in BACKLASH_STEPS:
        w = 44
        x0, y0, x1, y1 = _draw_button(
            canvas, bx, y + 2, w, 20, f"{step}°",
            state.backlash_step == step)
        hits.append({"kind": "step", "step": step,
                     "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        bx += w + 6
    bx += 10
    for preset, label in (("wide", "0-160 / 20°"),
                          ("zoom", "zoom 40°"),
                          ("fine", "fine 20° / 2°")):
        w = 100 if preset != "wide" else 110
        x0, y0, x1, y1 = _draw_button(canvas, bx, y + 2, w, 20, label, False)
        hits.append({"kind": "preset", "preset": preset,
                     "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        bx += w + 6
    y += 30

    # curve graph
    gx, gy, gw, gh = 14, y, TUNE_W - 28, 88
    _fill_rect(canvas, gx, gy, gx + gw, gy + gh, (22, 22, 22))
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), (70, 70, 70), 1)
    hits.append({"kind": "graph",
                 "x0": gx, "y0": gy, "x1": gx + gw, "y1": gy + gh})
    # highlight selected range
    rx0 = gx + int(round(state.backlash_range_lo / 180.0 * gw))
    rx1 = gx + int(round(state.backlash_range_hi / 180.0 * gw))
    overlay = canvas.copy()
    _fill_rect(overlay, rx0, gy, rx1, gy + gh, (50, 70, 40))
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)

    def _gxy(ang: float, extra: float) -> Tuple[int, int]:
        px = gx + int(round(ang / 180.0 * gw))
        py = gy + gh - 6 - int(round(
            max(0.0, min(BACKLASH_COMP_MAX, extra)) / BACKLASH_COMP_MAX
            * (gh - 12)))
        return px, py

    for direction, color in (("ccw", _COL_CCW), ("cw", _COL_CW)):
        table = _backlash_dict(c, direction)
        pts = [_gxy(float(a), backlash_at(table, float(a)))
               for a in range(0, 181, 2)]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)],
                          False, color, 1, cv2.LINE_AA)
        for ang, extra in table.items():
            cv2.circle(canvas, _gxy(float(ang), extra), 3, color, -1)
    for tick in (0, 45, 90, 135, 180):
        tx = gx + int(round(tick / 180.0 * gw))
        cv2.line(canvas, (tx, gy + gh - 4), (tx, gy + gh), (90, 90, 90), 1)
        _put(canvas, str(tick), (tx - 8, gy + gh + 14), _COL_MUTED, 0.32)
    _put(canvas, "CCW", (gx + 6, gy + 14), _COL_CCW, 0.36)
    _put(canvas, "CW", (gx + 46, gy + 14), _COL_CW, 0.36)
    _put(canvas, "click graph to center the edit window",
         (gx + 200, gy + 14), _COL_MUTED, 0.32)
    y = gy + gh + 22

    # per-angle sliders for the selected range
    _put(canvas, f"editing {state.backlash_range_lo}°–{state.backlash_range_hi}°"
         f"  every {state.backlash_step}°"
         f"  ({n_rows} pts, 0.1° steps)",
         (14, y), _COL_MUTED, 0.36)
    y += 16
    ccw_x, ccw_w = 70, 230
    cw_x = 430
    _put(canvas, "CCW extra °", (ccw_x, y), _COL_CCW, 0.36)
    _put(canvas, "CW extra °", (cw_x, y), _COL_CW, 0.36)
    y += 8
    for ang in angles:
        _put(canvas, f"{ang:>3}°", (14, y + 16), _COL_TEXT, 0.40)
        for direction, tx, color in (("ccw", ccw_x, _COL_CCW),
                                     ("cw", cw_x, _COL_CW)):
            table = _backlash_dict(c, direction)
            explicit = int(ang) in table
            val = backlash_at(table, float(ang))
            bx0, by0, bx1, by1 = _draw_button(
                canvas, tx - 22, y + 2, 18, 18, "-", False)
            hits.append({"kind": "nudge", "target": "backlash", "sign": -1,
                         "x0": bx0, "y0": by0, "x1": bx1, "y1": by1,
                         "direction": direction, "angle": int(ang),
                         "lo": 0.0, "hi": BACKLASH_COMP_MAX, "prec": 0.1})
            hx0, hy0, hx1, hy1 = _draw_track(
                canvas, tx, y + 6, ccw_w, 10, val, 0.0, BACKLASH_COMP_MAX,
                color)
            hits.append({"kind": "backlash", "direction": direction,
                         "angle": int(ang), "lo": 0.0, "hi": BACKLASH_COMP_MAX,
                         "prec": 0.1, "x0": hx0, "y0": hy0, "x1": hx1, "y1": hy1})
            px0, py0, px1, py1 = _draw_button(
                canvas, tx + ccw_w + 6, y + 2, 18, 18, "+", False)
            hits.append({"kind": "nudge", "target": "backlash", "sign": 1,
                         "x0": px0, "y0": py0, "x1": px1, "y1": py1,
                         "direction": direction, "angle": int(ang),
                         "lo": 0.0, "hi": BACKLASH_COMP_MAX, "prec": 0.1})
            vcol = color if explicit else _COL_MUTED
            mark = "*" if explicit else " "
            _put(canvas, f"{val:4.1f}{mark}", (tx + ccw_w + 28, y + 16),
                 vcol, 0.40)
        y += 28

    _put(canvas, "* = explicit anchor (right-click to drop back to interpolated)",
         (14, y + 12), _COL_MUTED, 0.34)

    _tune_hits = hits
    cv2.imshow(WIN_TUNE, canvas)


def backlash_table(d: Dict[int, float]) -> List[Tuple[float, float]]:
    return sorted((float(a), float(v)) for a, v in d.items())


def backlash_at(d: Dict[int, float], angle: float) -> float:
    return Braccio._interp_backlash(float(angle), backlash_table(d))


def sync_calib_to_arm(arm: "Braccio", calib: "Calibration") -> None:
    arm.backlash_ccw_table = backlash_table(calib.base_backlash_ccw)
    arm.backlash_cw_table = backlash_table(calib.base_backlash_cw)
    arm.gripper_open = int(calib.gripper_open)
    arm.gripper_closed = int(calib.gripper_closed)


# ----------------------------- main -----------------------------

@dataclass
class State:
    calib: Calibration = field(default_factory=Calibration)
    calib_loaded: bool = False
    busy: bool = False
    busy_msg: str = ""
    last_action: str = ""
    last_target_ws: Optional[Tuple[float, float]] = None
    target_seen_at: float = 0.0
    arm_goto_click: bool = False  # next click sends the arm to that point
    setup_step: int = 0  # 0 = need base, 1 = need heading, 2 = ready
    tuning_visible: bool = True
    backlash_range_lo: int = 0
    backlash_range_hi: int = 160
    backlash_step: int = 20


def main() -> None:
    state = State()
    state.calib_loaded = state.calib.load()
    state.setup_step = 2 if state.calib_loaded else 0

    arm = Braccio()
    sync_calib_to_arm(arm, state.calib)

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera: {CAMERA_SOURCE}", file=sys.stderr)
        sys.exit(1)

    corner_detector = make_detector(CORNER_DICT)
    target_detector = make_detector(TARGET_DICT)
    last_corner_pixels: dict = {}   # center of each marker (display)
    last_outer_corners: dict = {}   # outer corner point of each marker (warp)

    cv2.namedWindow(WIN_LIVE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_WORK, cv2.WINDOW_NORMAL)
    if state.tuning_visible:
        make_tuning_window(state)

    def on_workspace_mouse(event, x, y, flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x_mm, y_mm = warped_pixel_to_workspace_mm(
            x, y, WARP_PX_W, WARP_PX_H, state.calib)

        if state.arm_goto_click:
            state.arm_goto_click = False
            x_r, y_r = workspace_to_robot(x_mm, y_mm, state.calib)
            print(f"[GOTO] ws=({x_mm:.1f},{y_mm:.1f}) "
                  f"robot=({x_r:.1f},{y_r:.1f})")
            run_async("go to clicked",
                      lambda: go(x_r, y_r, APPROACH_Z, arm.last_pose[5]))
            return

        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            dx = x_mm - state.calib.base_x_mm
            dy = y_mm - state.calib.base_y_mm
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                state.calib.heading_deg = math.degrees(math.atan2(dy, dx))
                state.setup_step = 2
                print(f"[CALIB] heading -> {state.calib.heading_deg:.1f} deg")
        else:
            state.calib.base_x_mm = x_mm
            state.calib.base_y_mm = y_mm
            # auto-aim heading toward the center of the workspace so first
            # click already gives a sensible default before shift+click
            cx = state.calib.workspace_w_mm * 0.5
            cy = state.calib.workspace_h_mm * 0.5
            dx = cx - x_mm
            dy = cy - y_mm
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                state.calib.heading_deg = math.degrees(math.atan2(dy, dx))
            state.setup_step = max(state.setup_step, 1)
            print(f"[CALIB] base -> ({x_mm:.1f}, {y_mm:.1f}) mm  "
                  f"heading auto -> {state.calib.heading_deg:.1f}")

    cv2.setMouseCallback(WIN_WORK, on_workspace_mouse)

    def run_async(label: str, fn) -> None:
        if state.busy:
            print(f"[WARN] busy: {state.busy_msg}")
            return

        def runner() -> None:
            state.busy = True
            state.busy_msg = label
            try:
                fn()
                state.last_action = f"{label}: ok"
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] {label}: {exc}")
                state.last_action = f"{label}: ERROR ({exc})"
            finally:
                state.busy = False
                state.busy_msg = ""

        threading.Thread(target=runner, daemon=True).start()

    def go(x: float, y: float, z: float, gripper: int,
           speed: int = TRAVEL_SPEED) -> bool:
        sol = solve_ik(x, y, z, state.calib)
        if sol is None:
            print(f"[WARN] IK failed for robot=({x:.1f}, {y:.1f}, {z:.1f}) "
                  f"- check heading/base calibration")
            return False
        b, s, e, w, _ = sol
        arm.send(b, s, e, w, 90, gripper, speed=speed)
        return True

    def _resolve_target_robot() -> Tuple[float, float]:
        if state.last_target_ws is None or \
                time.time() - state.target_seen_at > 2.0:
            raise RuntimeError("No fresh target detection")
        x_w, y_w = state.last_target_ws
        return workspace_to_robot(x_w, y_w, state.calib)

    def do_preview() -> None:
        x_r, y_r = _resolve_target_robot()
        go(x_r, y_r, APPROACH_Z, arm.gripper_closed, speed=TRAVEL_SPEED)

    def do_pickup() -> None:
        x_r, y_r = _resolve_target_robot()
        print(f"[PICK] robot=({x_r:.1f}, {y_r:.1f})")
        # 1) home
        arm.home()
        time.sleep(STEP_DELAY)
        # 2) approach z=100 with gripper CLOSED
        go(x_r, y_r, APPROACH_Z, arm.gripper_closed, speed=TRAVEL_SPEED)
        time.sleep(STEP_DELAY)
        # 3) open gripper at approach
        go(x_r, y_r, APPROACH_Z, arm.gripper_open, speed=TRAVEL_SPEED)
        time.sleep(STEP_DELAY)
        # 4) descend to z=10 with gripper open
        print("[PICK] descending")
        go(x_r, y_r, PICK_Z, arm.gripper_open, speed=PICK_SPEED)
        time.sleep(STEP_DELAY)
        # 5) close gripper
        go(x_r, y_r, PICK_Z, arm.gripper_closed, speed=PICK_SPEED)
        time.sleep(STEP_DELAY)
        # 6) lift to z=200
        go(x_r, y_r, LIFT_Z, arm.gripper_closed, speed=TRAVEL_SPEED)
        time.sleep(STEP_DELAY)
        # 7) move to drop position
        dx, dy, dz = DROP_TARGET_ROBOT
        print(f"[PICK] moving to drop robot=({dx:.1f}, {dy:.1f})")
        go(dx, dy, LIFT_Z, arm.gripper_closed, speed=TRAVEL_SPEED)
        time.sleep(STEP_DELAY)
        go(dx, dy, dz, arm.gripper_closed, speed=TRAVEL_SPEED)
        time.sleep(STEP_DELAY)
        # 8) open gripper, home
        print("[PICK] releasing")
        go(dx, dy, dz, arm.gripper_open, speed=TRAVEL_SPEED)
        time.sleep(STEP_DELAY)
        print("[PICK] done")
        arm.home()

    while True:
        # keep arm-side gripper / backlash values in sync with calibration
        sync_calib_to_arm(arm, state.calib)
        if state.tuning_visible:
            update_tuning_window(state)

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[ERROR] Lost camera frame.")
            break
        display = frame.copy()

        corners, ids, _ = corner_detector.detectMarkers(frame)
        corners_by_id: dict = {}    # marker center points (for display)
        _raw_arrays: dict = {}      # raw 4-corner arrays from this frame
        if ids is not None:
            for c, raw_id in zip(corners, ids.flatten()):
                i = int(raw_id)
                if 1 <= i <= 4:
                    arr = c[0]  # shape (4, 2): TL, TR, BR, BL
                    cx = float(arr[:, 0].mean())
                    cy = float(arr[:, 1].mean())
                    corners_by_id[i] = (cx, cy)
                    last_corner_pixels[i] = (cx, cy)
                    _raw_arrays[i] = arr
        for i in range(1, 5):
            if i not in corners_by_id and i in last_corner_pixels:
                corners_by_id[i] = last_corner_pixels[i]
        all_corners = all(i in corners_by_id for i in (1, 2, 3, 4))

        # Compute outer corner of each workspace marker (farthest from centroid).
        # This makes the warp cover the full area up to the marker edges.
        outer_by_id: dict = {}
        if all_corners:
            cx_mean = sum(corners_by_id[i][0] for i in range(1, 5)) / 4.0
            cy_mean = sum(corners_by_id[i][1] for i in range(1, 5)) / 4.0
            centroid = np.array([cx_mean, cy_mean])
            for i in range(1, 5):
                if i in _raw_arrays:
                    arr = _raw_arrays[i]
                    dists = np.linalg.norm(arr - centroid, axis=1)
                    pt = arr[int(np.argmax(dists))]
                    outer_by_id[i] = (float(pt[0]), float(pt[1]))
                    last_outer_corners[i] = outer_by_id[i]
                elif i in last_outer_corners:
                    outer_by_id[i] = last_outer_corners[i]
                else:
                    outer_by_id[i] = corners_by_id[i]

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
        if all_corners:
            poly = np.array([corners_by_id[i] for i in (1, 2, 3, 4)],
                            dtype=np.int32)
            cv2.polylines(display, [poly], True, (0, 255, 255), 2)
            for i in range(1, 5):
                p = (int(corners_by_id[i][0]), int(corners_by_id[i][1]))
                cv2.circle(display, p, 7, (0, 255, 0), -1)
                cv2.putText(display, str(i), (p[0] + 10, p[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        target_ws_now: Optional[Tuple[float, float]] = None
        warped: Optional[np.ndarray] = None
        if all_corners and len(outer_by_id) == 4:
            warped = warp_workspace(frame, outer_by_id)
        if warped is not None:
            tcorners, tids, _ = target_detector.detectMarkers(warped)
            if tids is not None and len(tcorners) > 0:
                pts = tcorners[0][0]
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                target_ws_now = warped_pixel_to_workspace_mm(
                    cx, cy, warped.shape[1], warped.shape[0], state.calib)
                state.last_target_ws = target_ws_now
                state.target_seen_at = time.time()
                cv2.aruco.drawDetectedMarkers(warped, tcorners, tids)
                cv2.circle(warped, (int(cx), int(cy)), 14, (0, 0, 255), 2)
                cv2.line(warped, (int(cx), 0),
                         (int(cx), warped.shape[0]), (0, 0, 255), 1)
                cv2.line(warped, (0, int(cy)),
                         (warped.shape[1], int(cy)), (0, 0, 255), 1)

            draw_workspace_overlay(
                warped, state.calib,
                target_ws_now,
                setup_step=state.setup_step,
                goto_armed=state.arm_goto_click,
            )
            cv2.imshow(WIN_WORK, warped)

        # status panel
        target_age = time.time() - state.target_seen_at \
            if state.last_target_ws else None
        target_fresh = target_age is not None and target_age < 1.5

        lines = []
        lines.append((f"Workspace: {state.calib.workspace_w_mm:.0f} x "
                      f"{state.calib.workspace_h_mm:.0f} mm",
                      (210, 210, 210)))
        lines.append((f"Robot pos:  ({state.calib.base_x_mm:.0f}, "
                      f"{state.calib.base_y_mm:.0f}) mm",
                      (210, 210, 210)))
        lines.append((f"Heading: {state.calib.heading_deg:.0f} deg   "
                      f"Base offset: {state.calib.base_offset_deg:+.0f}",
                      (210, 210, 210)))
        lines.append((f"L3 (gripper): {L3:.0f} mm",
                      (160, 160, 160)))
        lines.append(("", (0, 0, 0)))
        if all_corners:
            lines.append(("Corners 1..4: OK", (0, 200, 0)))
        else:
            missing = [i for i in (1, 2, 3, 4) if i not in corners_by_id]
            lines.append((f"Corners missing: {missing}", (0, 100, 255)))

        if state.last_target_ws is None:
            lines.append(("Target: not detected", (0, 100, 255)))
        else:
            x_w, y_w = state.last_target_ws
            x_r, y_r = workspace_to_robot(x_w, y_w, state.calib)
            r = math.hypot(x_r, y_r)
            color = (0, 255, 0) if target_fresh else (0, 165, 255)
            lines.append((f"Target ws: ({x_w:.0f}, {y_w:.0f}) mm", color))
            lines.append((f"Target rb: ({x_r:.0f}, {y_r:.0f}) mm  "
                          f"r={r:.0f}", color))

        lines.append(("", (0, 0, 0)))
        if state.busy:
            lines.append((f"BUSY: {state.busy_msg}", (0, 200, 255)))
        elif state.last_action:
            lines.append((state.last_action, (180, 180, 180)))

        lines.append(("", (0, 0, 0)))
        lines.append((f"Gripper: open={state.calib.gripper_open}  "
                      f"closed={state.calib.gripper_closed}", (180, 220, 255)))
        def _fmt_backlash(d: dict) -> str:
            if not d:
                return "(none)"
            items = sorted((int(a), float(v)) for a, v in d.items())
            if len(items) > 6:
                items = items[:: max(1, len(items) // 6)]
            return " ".join(f"{a}:{v:.1f}" for a, v in items)

        lines.append(("Back CCW " + _fmt_backlash(state.calib.base_backlash_ccw),
                      (180, 220, 255)))
        lines.append(("Back  CW " + _fmt_backlash(state.calib.base_backlash_cw),
                      (180, 220, 255)))
        lines.append((f"Reach @100={state.calib.reach_at_100mm:+.0f}  "
                      f"@300={state.calib.reach_at_300mm:+.0f}  "
                      f"radial *{state.calib.radial_compensation:.3f}",
                      (180, 220, 255)))
        lines.append((f"z off={state.calib.z_offset_mm:+.0f}mm  "
                      f"elbow={state.calib.elbow_offset_deg:+.0f}deg  "
                      f"wrist={state.calib.wrist_offset_deg:+.0f}deg",
                      (180, 220, 255)))
        lines.append(("", (0, 0, 0)))
        lines.append(("Click workspace = set base", (200, 200, 200)))
        lines.append(("Shift+click = aim heading", (200, 200, 200)))
        lines.append(("[G]+click = arm goes above point", (200, 200, 200)))
        lines.append(("[H]ome  [T]est pose", (255, 255, 255)))
        lines.append(("[M] preview above target", (255, 255, 255)))
        lines.append(("[P] pick & place", (255, 255, 255)))
        lines.append(("[O] open  [C] close gripper", (255, 255, 255)))
        lines.append(("[I] invert gripper open<->close", (255, 255, 255)))
        lines.append(("[F] toggle tuning dashboard", (255, 255, 255)))
        lines.append(("[ ] heading   , . base offset", (255, 255, 255)))
        lines.append(("- = workspace size", (255, 255, 255)))
        lines.append(("[S]ave  [L]oad  [D]ump", (255, 255, 255)))
        lines.append(("[Q] quit", (255, 255, 255)))

        panel = render_panel(PANEL_W, lines)
        cv2.imshow(WIN_LIVE, compose(display, panel))

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue
        ch = chr(key) if 32 <= key < 127 else ""

        if key in (ord('q'), 27):
            break
        elif ch == 'h':
            run_async("home", arm.home)
        elif ch == 't':
            run_async("test pose", arm.safe_test_pose)
        elif ch == 'o':
            run_async("open gripper", arm.open_gripper)
        elif ch == 'c':
            run_async("close gripper", arm.close_gripper)
        elif ch == 'p':
            run_async("pick & place", do_pickup)
        elif ch == 'm':
            run_async("preview above target", do_preview)
        elif ch == 'g':
            state.arm_goto_click = not state.arm_goto_click
            print(f"[MODE] click-to-go: {state.arm_goto_click}")
        elif ch == 'i':
            state.calib.gripper_open, state.calib.gripper_closed = (
                state.calib.gripper_closed, state.calib.gripper_open)
            sync_calib_to_arm(arm, state.calib)
            print(f"[GRIPPER] swapped -> open={state.calib.gripper_open} "
                  f"closed={state.calib.gripper_closed}")
        elif ch == 'f':
            state.tuning_visible = not state.tuning_visible
            if state.tuning_visible:
                make_tuning_window(state)
            else:
                try:
                    cv2.destroyWindow(WIN_TUNE)
                except cv2.error:
                    pass
        elif ch == '[':
            state.calib.heading_deg = (state.calib.heading_deg - 5.0) % 360.0
        elif ch == ']':
            state.calib.heading_deg = (state.calib.heading_deg + 5.0) % 360.0
        elif ch == ',':
            state.calib.base_offset_deg -= 5.0
        elif ch == '.':
            state.calib.base_offset_deg += 5.0
        elif ch == '-':
            state.calib.workspace_w_mm = max(40.0, state.calib.workspace_w_mm - 5.0)
            state.calib.workspace_h_mm = max(40.0, state.calib.workspace_h_mm - 5.0)
        elif ch == '=' or ch == '+':
            state.calib.workspace_w_mm += 5.0
            state.calib.workspace_h_mm += 5.0
        elif ch == 's':
            state.calib.save()
        elif ch == 'l':
            state.calib.load()
        elif ch == 'd':
            print("\n--- STATE DUMP ---")
            print(json.dumps(asdict(state.calib), indent=2))
            if state.last_target_ws:
                xw, yw = state.last_target_ws
                xr, yr = workspace_to_robot(xw, yw, state.calib)
                print(f"target ws=({xw:.1f},{yw:.1f}) "
                      f"robot=({xr:.1f},{yr:.1f})")
                sol = solve_ik(xr, yr, PICK_Z, state.calib)
                print(f"ik={sol}")
            print(f"arm.last_pose={arm.last_pose}\n")

    try:
        arm.home()
        time.sleep(1.5)
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
