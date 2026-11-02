# WAPR pose tracker

## Hand-eye calibration data collection

Connect a RealSense camera and make the UR robot reachable over RTDE. Run on a
desktop with a display (the preview uses OpenCV):

```bash
uv sync
uv run python src/calibration/data_collection.py --save-dir data/hand_eye --num-samples 20
```

The default robot IP is `192.168.1.130`; override it with `--robot-ip`.
Use `--camera-serial` to select a specific RealSense when several are connected.
The color stream is **1280 × 720 at 30 FPS**.
The RealSense SDK rotation filter rotates both the preview and saved images
**180°** to match the upside-down camera mounting.

Move the robot manually and let it settle. With the preview window focused,
press **Space** to save one image/TCP pair. Repeat at different positions and
orientations while keeping the calibration target visible. Press **Q**, **Esc**,
or **Ctrl+C** to finish early, or close the preview. Omit `--num-samples` to keep
collecting until you quit. The script only receives robot state; it does not
enable freedrive or command motion.

Each run creates a new session without overwriting earlier captures:

```text
data/hand_eye/session_<UTC timestamp>_<id>/
├── 1.png
├── 2.png
├── ...
├── robot_poses.csv
└── camera_intrinsics.yaml
```

This matches `camera_calibration/calibration_gui/calibration_data/example_data`.
Images are lossless PNGs without preview overlays. `robot_poses.csv` has **no
header** and exactly six numeric columns: `x,y,z,rx,ry,rz` (translation in metres,
orientation as a rotation vector in radians). Row 1 belongs to `1.png`, row 2 to
`2.png`, and so on. The CSV is updated after every capture.

`camera_intrinsics.yaml` contains the image dimensions, camera matrix, distortion
coefficients, rectification matrix, and projection matrix in the example's YAML
structure. The principal point and tangential distortion coefficients are adjusted
for the 180° rotation, so the GUI can use the saved images directly. Calibration
results refer to the rotated optical frame (X and Y reversed, Z unchanged relative
to the native camera frame). Nonzero distortion models incompatible with OpenCV's
`plumb_bob` model are rejected rather than exported incorrectly.

Camera and robot reads are sequential, not hardware synchronized. Keep the robot
stationary when capturing. Keep the robot's configured TCP unchanged throughout
collection and subsequent calibration; the saved poses refer to that TCP in the
robot base frame. Previously recorded sessions retain their original format.

API references: [UR RTDE robot state](https://sdurobotics.gitlab.io/ur_rtde/pages/getting_started/quick_start.html)
and [RealSense OpenCV example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py).
