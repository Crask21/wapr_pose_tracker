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
├── session.json
├── poses.csv
└── images/
    ├── 000000.png
    └── 000001.png
```

`session.json` records camera intrinsics, the RealSense distortion model, stream
settings, robot IP, and pose conventions. Each row in `poses.csv` contains a
`sample_index`, its relative `image` path, the TCP pose columns `x,y,z,rx,ry,rz`
(metres and a rotation vector in radians), and camera/host timestamps. Columns
`T_base_tcp_00` through `T_base_tcp_33` store the 4×4 transform from the configured
TCP frame into the robot base, indexed by row and column. Images are lossless
PNGs without preview overlays. The CSV is updated after every capture.

Camera and robot reads are sequential, not hardware synchronized. Keep the robot
stationary when capturing. Keep the robot's configured TCP unchanged throughout
collection and subsequent calibration; the saved poses refer to that TCP.
RealSense distortion coefficients are stored with their native model and should
be interpreted using that model when calibrating.
Intrinsics remain in the native, unrotated color-camera frame. Before using them
for pose estimation, map detected points in the saved images back with
`u_native = 1279 - u_saved` and `v_native = 719 - v_saved`. This transform is also
stored as `camera.saved_pixel_to_native_pixel` in `session.json`. Alternatively,
rotate saved images 180° again before detecting calibration points.

API references: [UR RTDE robot state](https://sdurobotics.gitlab.io/ur_rtde/pages/getting_started/quick_start.html)
and [RealSense OpenCV example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py).
