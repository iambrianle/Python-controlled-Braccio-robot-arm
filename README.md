# Python-controlled Braccio robot arm

Vision-guided pick-and-place for a [Tinkerkit Braccio](https://store.arduino.cc/products/tinkerkit-braccio-robot). The script to run is **`braccio_pick.py`**: it finds a workspace marked with four ArUco corners, locates a tagged object, and drives the arm over serial.

Demo of the original project: [YouTube](https://www.youtube.com/watch?v=SUgCcF-lWCI).

The inverse kinematics were written for one mechanical assembly of the official Braccio. If your arm was built differently, you will need to recalibrate (and possibly adjust joint offsets) before picks are accurate.

---

## What you need

**Hardware**

- Tinkerkit Braccio arm with its Arduino shield and **external 12 V power supply** (USB power is not enough for the servos)
- Arduino board the shield sits on (typically Uno)
- A camera looking down on the table: USB webcam **or** a phone running an IP webcam app
- A printer, to make the ArUco markers

**Software**

- [Arduino IDE](https://www.arduino.cc/en/software)
- Python **3.10 or newer**
- Packages in `requirements.txt` (installed in the Python step below)

**Arduino library (not pip)**

In Arduino IDE: **Sketch → Include Library → Manage Libraries…** and install **Braccio Robot** by Stefan Strömberg ([stefangs/arduino-library-braccio-robot](https://github.com/stefangs/arduino-library-braccio-robot)). The sketch uses `BraccioRobot.h` and `Position::setFromString`. Do **not** confuse this with the older official `Braccio` library.

---

## 1. Print the ArUco markers

Use a generator such as [chev.me/arucogen](https://chev.me/arucogen/). Print them large enough to stay readable from the camera (roughly 50–80 mm on a side works well).

| Role | Dictionary | IDs |
| --- | --- | --- |
| Workspace corners | **4x4** (50) | **1, 2, 3, 4** |
| Object to pick | **6x6** (50) | any ID (the script uses the first 6x6 marker it sees) |

Tape the four corner markers on the table in this layout, as seen from above:

```
  2 -------------------- 3
  |                      |
  |      workspace       |
  |                      |
  1 -------------------- 4
```

- Marker **1** is the origin (bottom-left of the workspace).
- **+X** runs toward marker **4**.
- **+Y** runs toward marker **2**.

Measure the rectangle between the **outer** corners of the markers (default in the script is **280 × 360 mm**). You can change that later with `-` / `=` in the UI, or by editing `braccio_calib.json`.

Put the arm so the camera can see all four corners **and** the gripper area. Typical placement is along one edge of the rectangle, not covering a marker.

---

## 2. Flash the Arduino

1. Connect the Arduino over USB. Plug in the Braccio’s **12 V** supply and set the shield power switch to **ON**.
2. Open `arduino_python_script3/arduino_python_script3.ino` in Arduino IDE.
3. Select your board and port, then **Upload**.
4. **Close the Serial Monitor and quit Arduino IDE** (or at least close the port). Only one program can own the serial port; if the IDE holds it, Python will fail to connect.
5. After upload the arm should move to a home pose (base 90°, other joints 90°, gripper open-ish).

The firmware talks at **115200 baud**. Commands look like `Pbase,shoulder,elbow,wrist,wristRot,gripper,speed` followed by a newline. `braccio_pick.py` sends those for you.

---

## 3. Install Python dependencies

From this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

That installs OpenCV, NumPy, and pySerial (plus SymPy for the older scripts).

---

## 4. Set the serial port and camera

Defaults in `braccio_pick.py` are a macOS USB-serial path and an IP-camera URL. Override them so they match **your** machine.

**Serial port**

```bash
# macOS
ls /dev/cu.usbmodem*

# Linux
ls /dev/ttyACM* /dev/ttyUSB*

# Windows: Device Manager → Ports (COM & LPT), e.g. COM3
```

**Camera**

- USB webcam: use the device index (`0` is usually the built-in or first camera; try `1` if that is wrong).
- Phone: install an IP webcam app, start the video server, and use the `/video` URL it shows (for example `http://192.168.1.20:8080/video`).

**Run (examples)**

```bash
# macOS / Linux — USB serial + USB webcam
export BRACCIO_SERIAL_PORT=/dev/cu.usbmodem2101
export BRACCIO_CAMERA=0
python3 braccio_pick.py

# IP camera instead of a webcam
export BRACCIO_CAMERA=http://192.168.1.20:8080/video
python3 braccio_pick.py
```

Windows (PowerShell):

```powershell
$env:BRACCIO_SERIAL_PORT="COM3"
$env:BRACCIO_CAMERA="0"
python braccio_pick.py
```

Optional: `BRACCIO_GRIPPER_INVERT=1` if open and close are swapped at startup (you can also press `I` in the UI).

---

## 5. First-run calibration

This repo includes a `braccio_calib.json` from one physical setup. **It will not match your table.** Either delete that file before the first run, or ignore the loaded numbers and click a new base/heading, then press `S`.

1. Start `braccio_pick.py`. You should get two OpenCV windows: **camera** and **workspace**. A **tuning** window opens as well (`F` hides it).
2. The camera overlay should show **Corners 1..4: OK**. If it lists missing IDs, fix lighting, marker print quality, or camera angle until all four are green.
3. In the **workspace** window:
   - **Left click** on the robot’s base (the center of the rotating shoulder/base).
   - **Shift + left click** on a point the arm’s “forward” axis should aim at (straight out from the Braccio, along the table).
4. Press **`T`** for a safe forward-leaning test pose, **`H`** to home, **`O`** / **`C`** to open and close the gripper. If close opens instead, press **`I`**.
5. Press **`G`**, then click a reachable spot on the workspace. The gripper should move **above** that point (not down onto the table). Repeat clicks and heading/base nudges until “above the click” is consistently above the click.
6. Put a 6x6 marker on the object you want to pick. When the workspace view draws a red cross on it, press **`M`** (preview: move above the target, no descend). If that looks good, press **`P`** to pick and drop.
7. Press **`S`** to save calibration to `braccio_calib.json` (next launch reloads it).

Useful extra keys: `[` `]` heading ±5°, `,` `.` base offset ±5°, `-` `=` workspace size, `R` reach overlay, `D` dump state to the terminal, `Q` or Esc quit.

The drop location is hard-coded in robot coordinates as `DROP_TARGET_ROBOT` near the top of `braccio_pick.py` (about 145, 37, 120 mm). Change that if you want a different release point.

---

## 6. Tuning if the gripper misses

Use the tuning window (or `D` to print values):

- **Reach @100 / @300 mm** — if the arm stops short or overshoots, especially far vs near targets.
- **Z offset** — if it hits the table or stays too high.
- **Base offset** and **backlash CCW/CW** — if left/right aiming depends on which way the base last turned.
- **Elbow / wrist offsets** — if the gripper pitch is wrong at the object.

Change a slider, then **`M`** or **`G`+click** to test. Save with **`S`** when it is good.

Keep clear of the arm. Start with **`M`** before **`P`**. **`P`** homes, approaches, opens, descends to ~10 mm, closes, lifts, moves to the drop pose, and opens.

---

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `could not open port` / serial busy | Close Arduino IDE Serial Monitor. Unplug/replug USB. Check `BRACCIO_SERIAL_PORT`. |
| `[ERROR] Could not open camera` | Wrong `BRACCIO_CAMERA`. For USB try `0` then `1`. For IP cam, open the URL in a browser first. |
| `Lost camera frame` | Phone/Wi‑Fi dropped, or the USB cam was unplugged. |
| Corners missing | Print larger, flatten the paper, add light, keep all four in view. Dictionaries must be **4x4 IDs 1–4**. |
| `No fresh target detection` on `P`/`M` | Need a visible **6x6** marker on the object. |
| `IK failed` | Target is behind the arm or out of reach; fix heading/base click, or move the object closer. |
| Gripper direction reversed | Press `I`, or start with `BRACCIO_GRIPPER_INVERT=1`. |
| Arm barely moves / jitters | 12 V supply not connected, shield switch off, or USB-only power. |
| Picks were accurate on someone else’s video but not yours | Recalibrate; joint wiring/assembly differs between kits. |

---

## Other scripts in this repo

`Aruco_detection_V2.py` + `braccio_control_python.py` are the older camera + IK pipeline. Prefer **`braccio_pick.py`**. If you do run the old pair, set `BRACCIO_SERIAL_PORT` (and `BRACCIO_CAMERA_URL` in `Aruco_detection_V2.py`) the same way, and keep both files in this folder.
