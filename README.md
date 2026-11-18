# WAPR pose tracker

## Live single-object tracking

Connect a RealSense with RGB and depth streams and supply a triangle mesh of the
object. Run from a desktop session with an OpenCV display and a CUDA GPU:

```bash
uv sync
source .venv/bin/activate
python -m ensurepip --upgrade
python -m wapr.bootstrap --feature core dinov2 det2d sam2
wapr-pose-tracker --mesh /path/to/object.obj --mesh-scale 0.001
```

Use `--mesh-scale 0.001` for models authored in millimetres, or `1` for metres
(the default). Mesh texture/material files should remain beside the model.
The default diameter is the metric bounding-box diagonal; use `--diameter-m`
to supply a measured maximum object diameter. Wrong mesh scale causes wrong poses.
`python -m wapr_pose_tracker` is an equivalent entry point.

Click inside the object in the preview. WAPR renders CAD templates, detects and
segments the displayed frame, selects the highest-scoring matching mask covering
the click, and estimates its initial pose. Keep the camera and object stationary
during initialization, especially while first-use templates/engines are prepared.
Subsequent frames propagate a DINOv2 region, obtain a depth-based translation
candidate, and select among WAPR's six tracking hypotheses. No detector runs on
routine updates. This uses the WAPR APIs demonstrated by Example 08; it does not
copy that example's dataset-specific recovery search.

The preview draws the model bounding box and XYZ axes (red/green/blue), displays
translation in metres, and reports processing time including region tracking.
Green indicates a nonpositive WBPS group error; orange means uncertain and is
not a calibrated confidence estimate. Region/depth loss clears the pose and
requires another click. Press **R** to clear/reinitialize, **Q/Esc** to quit, or
close the window. Inspect the overlay: tracking and loss checks cannot guarantee
correct identity or pose, especially for symmetric or poorly observed objects.

```bash
wapr-pose-tracker --mesh models/object.ply --mesh-scale 0.001 \
  --camera-serial YOUR_SERIAL --width 640 --height 480 --fps 30 \
  --pose-log poses.jsonl
```

The requested RGB and depth resolution/rate must be supported by the camera.
Camera FPS is independent of inference throughput; capture queues are limited
and frames may be skipped when processing is slower. `--no-region-tracking`
disables DINOv2 propagation and uses previous-pose refinement only; this also
disables region/depth loss checks. CUDA-synchronized processing timing excludes
capture, drawing, UI, and logging. The displayed loop rate includes capture,
processing, overlay drawing, and pose logging; it excludes UI wait and
initialization, which is reported separately.

Pose output is **`T_camera_object`**, mapping original mesh coordinates (scaled
to metres, preserving the file's origin) into the **native color optical frame**:
X right, Y down, Z forward. Depth is aligned to color and converted using the
device's depth scale. Brown-Conrady color distortion is rectified for both RGB
and aligned depth; unsupported nonzero distortion models are rejected.
Tracking images are **not rotated**, unlike the hand-eye collection below.
Do not apply that rotated-frame calibration directly to these poses.

`--pose-log` creates a new JSONL file and refuses to overwrite an existing one.
Each successful initialization/update records the 4×4 matrix, status, quality,
frame number, camera timestamp and clock domain, host receipt time in Unix
nanoseconds, and processing time. Uncertain estimates are explicitly labeled;
no stale poses are emitted while waiting for initialization or after loss.
Camera timestamps are device timestamps, not ROS time. Host receipt time is
after alignment, not a hardware exposure timestamp. Console pose output is
limited to approximately once per second; the log records every pose update.

Camera capture (`camera.py`), source-independent pose updates (`tracker.py`), and
the display/logging runner (`cli.py`) are separate. The ROS adapter below feeds
aligned topic observations to the same tracker without opening the camera SDK.

Hardware-free checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## ROS 2 RealSense tracking

ROS 2 Jazzy and `realsense2_camera` are available on this computer. This package
also installs through colcon using `ament_cmake_python`; the existing uv package
and standalone command remain available. ROS Python messages come from the ROS
environment, while WAPR/CUDA dependencies come from the project virtualenv.

Build from the repository root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths . --packages-select wapr_pose_tracker --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash

ros2 launch wapr_pose_tracker realsense_tracking.launch.py \
  python_executable:="$PWD/.venv/bin/python" \
  mesh_path:="$PWD/assets/65mmCube.obj" mesh_scale:=0.001
```

If setting up a new environment, install the Python dependencies and prepare
WAPR as described above first. When updating an already bootstrapped environment,
use `uv sync --inexact` to preserve optional dependencies installed by WAPR.
Do not run the standalone camera command concurrently with the ROS camera driver.

The launch starts both nodes, requests synchronized 640×480 RGB-D at 30 Hz,
enables depth-to-color alignment in the native camera geometry, then rotates
the aligned RGB, depth, and intrinsics **180° inside the tracker**. The driver
rotation filter stays disabled. Click the object in the rotated preview to initialize. R
resets; Q/Esc exits both launched nodes. First initialization can take time while
templates and engines are prepared; keep the scene stationary until it completes.
No poses are published while waiting for initialization or after a lost track.

The tracker subscribes to:

| Topic | Type |
| --- | --- |
| `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` |
| `/camera/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` |

It publishes **`/wapr_pose_tracker/pose`**, type
**`geometry_msgs/msg/PoseStamped`**, with `header.frame_id = "world"` and the
source color-image timestamp. The position is in metres and the quaternion is
ordered XYZW. The object frame remains the model file's original origin after
unit conversion. No robot motion commands or world TF broadcaster are added.

```bash
# In another ROS-sourced terminal:
ros2 topic echo /wapr_pose_tracker/pose
ros2 service call /wapr_pose_tracker/reset std_srvs/srv/Trigger '{}'
```

The supplied YAML maps camera → world:
`p_world = T_world_camera @ p_camera`. Its direction was checked against a
measured object position; the previous world → camera labels were incorrect.
The node applies it directly: `T_world_object = T_world_camera @ T_camera_object`.
The calibrated
camera frame uses the **180° rotated images** from calibration collection.
With rotation disabled, the node inserts `diag(-1, -1, 1, 1)` between calibration
and observation frames. The YAML's full `transform` matrix is authoritative;
the duplicate translation/quaternion fields are informational.

The driver rotation filter precedes depth alignment, which makes sensor
extrinsics and rotated stream profiles interact during registration. The launch
keeps this filter disabled so alignment uses factory native sensor geometry,
matching the standalone path. The tracker rectifies native RGB and aligned depth
using the color CameraInfo, then rotates both arrays and sets
`cx_rotated = width - 1 - cx`, `cy_rotated = height - 1 - cy`; focal lengths stay
unchanged. Raw ROS camera topics and their TF frames remain native/unrotated.
The tracker preview and inference are rotated, and world-frame pose composition
accounts for that rotation. Effective tracking intrinsics are printed when the
camera profile changes. See the [driver filter order](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_camera/src/base_realsense_node.cpp).

Selected launch overrides (`ros2 launch ... --show-args` lists all):

| Argument | Default | Purpose |
| --- | --- | --- |
| `calibration_file` | packaged `configs/hand_eye_cam_1.yaml` | Extrinsic calibration |
| `calibration_direction` | `from_yaml` | Honors source/target; explicit overrides available |
| `rotate_180` | `true` | Rotate RGB, aligned depth, and K after native alignment/rectification |
| `calibration_camera_rotated` | `true` | Set false only for calibration from native images |
| `serial_no` | empty | Select a RealSense |
| `camera_profile` | `640x480x30` | RGB and depth profiles |
| `region_tracking` | `true` | DINOv2 region propagation |
| `display` | `true` | Click-to-initialize UI |
| `initial_u`, `initial_v` | `-1`, `-1` | Initial pixel coordinates for headless use |
| `max_frame_age` | `0.5` s | Suppress stale observations and finished poses |
| `publish_uncertain` | `false` | Allow publication with positive WBPS error |
| `pose_topic` | `/wapr_pose_tracker/pose` | Output topic |

For headless use set `display:=false initial_u:=320 initial_v:=240` to select
the object covering that pixel. A reset retries the configured initial click.
If automatic initialization fails without a display, correct the coordinates
and restart. When the scene or camera profile changes, reinitialize.

Images are decoded with row stride and byte order respected. Z16 ROS depth is
millimetres (the driver normalizes device depth units); 32FC1 depth is metres.
Invalid depth is zeroed and aligned depth is rectified together with RGB.
The ROS receiver uses sensor-data QoS, approximate timestamp synchronization,
and a one-frame mailbox so inference skips old frames rather than building a
queue. CUDA/OpenGL and the UI stay on the main thread. Publication timestamps
are never replaced by inference completion time. Positive WBPS group error
causes an orange overlay and, by default, suppresses publication; it is not a
calibrated confidence score. `PoseStamped` cannot carry uncertainty flags.

Run all geometry, message-publication, and standalone tests with ROS sourced:

```bash
source /opt/ros/jazzy/setup.bash
.venv/bin/python -m unittest discover -s tests -v
```

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
