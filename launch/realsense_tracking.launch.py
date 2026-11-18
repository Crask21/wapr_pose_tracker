"""Start RealSense RGB-D streams and WAPR in its CUDA Python environment."""

import os
from pathlib import Path
import shlex
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def setup(context):
    value = lambda name: context.launch_configurations[name]
    boolean = lambda name: value(name).lower() == "true"
    namespace, camera_name = value("camera_namespace").strip("/"), value("camera_name").strip("/")
    root = "/" + "/".join(part for part in (namespace, camera_name) if part)
    python = Path(value("python_executable")).expanduser()
    if not python.is_file():
        raise ValueError(f"python_executable not found: {python}")
    mesh = Path(value("mesh_path")).expanduser()
    if not mesh.is_file():
        raise ValueError(f"Supply mesh_path:=/absolute/path/to/object.obj (got {mesh})")
    profile = value("camera_profile")
    camera = Node(
        package="realsense2_camera", executable="realsense2_camera_node",
        name=camera_name, namespace=namespace, output="screen",
        parameters=[{
            "camera_name": camera_name,
            "serial_no": value("serial_no"), "enable_color": True, "enable_depth": True,
            "enable_infra1": False, "enable_infra2": False,
            "rgb_camera.color_profile": profile, "depth_module.depth_profile": profile,
            "rgb_camera.color_format": "RGB8", "enable_sync": True,
            "align_depth.enable": True,
            # Preserve factory sensor geometry during depth-to-color alignment.
            # Rotate the aligned observations in WAPR's decoder instead.
            "rotation_filter.enable": False,
            "rotation_filter.rotation": 180.0,
        }],
    )
    tracker = Node(
        package="wapr_pose_tracker", executable="wapr_pose_tracker_node",
        name="wapr_pose_tracker", output="screen",
        # ROS supplies rclpy/messages via setup.bash; CUDA/WAPR come from the venv.
        prefix=shlex.quote(str(python.absolute())),
        parameters=[{
            "mesh_path": str(mesh.resolve()), "mesh_scale": float(value("mesh_scale")),
            "diameter_m": float(value("diameter_m")), "device": value("device"),
            "calibration_file": value("calibration_file"),
            "calibration_direction": value("calibration_direction"), "world_frame": value("world_frame"),
            "images_rotated": False, "camera_info_rotated": False,
            "rotate_180": boolean("rotate_180"),
            "calibration_camera_rotated": boolean("calibration_camera_rotated"),
            "color_topic": root + "/color/image_raw",
            "depth_topic": root + "/aligned_depth_to_color/image_raw",
            "camera_info_topic": root + "/color/camera_info", "pose_topic": value("pose_topic"),
            "display": boolean("display"), "region_tracking": boolean("region_tracking"),
            "initial_click": [int(value("initial_u")), int(value("initial_v"))],
            "publish_uncertain": boolean("publish_uncertain"),
            "sync_slop": float(value("sync_slop")), "max_frame_age": float(value("max_frame_age")),
        }],
    )
    stop_camera = RegisterEventHandler(OnProcessExit(
        target_action=tracker, on_exit=[EmitEvent(event=Shutdown(reason="WAPR tracker exited"))]))
    stop_tracker = RegisterEventHandler(OnProcessExit(
        target_action=camera, on_exit=[EmitEvent(event=Shutdown(reason="RealSense camera exited"))]))
    return [stop_camera, stop_tracker, camera, tracker]


def generate_launch_description():
    share = Path(get_package_share_directory("wapr_pose_tracker"))
    venv = os.environ.get("VIRTUAL_ENV")
    default_python = str(Path(venv) / "bin/python") if venv else sys.executable
    arguments = {
        "mesh_path": ("", "Absolute path to the metric object model"),
        "mesh_scale": ("1.0", "Mesh units to metres; 0.001 for millimetres"),
        "diameter_m": ("0.0", "Zero uses the mesh bounding-box diagonal"),
        "python_executable": (default_python, "Absolute path to the WAPR venv Python"),
        "calibration_file": (str(share / "configs/hand_eye_cam_1.yaml"), "Hand-eye transform YAML"),
        "calibration_direction": ("from_yaml", "from_yaml, world_to_camera, or camera_to_world"),
        "world_frame": ("world", "Output PoseStamped frame_id"),
        "calibration_camera_rotated": ("true", "Calibration uses the 180-degree rotated optical frame"),
        "camera_namespace": ("camera", "RealSense namespace"),
        "camera_name": ("camera", "RealSense node name"),
        "serial_no": ("", "Optional RealSense device serial"),
        "camera_profile": ("640x480x30", "Requested RGB and depth profile"),
        "rotate_180": ("true", "Rotate aligned RGB/depth and intrinsics in the tracker after native alignment"),
        "pose_topic": ("/wapr_pose_tracker/pose", "World-frame geometry_msgs/PoseStamped"),
        "device": ("cuda:0", "WAPR CUDA device"), "display": ("true", "Enable click-selection preview"),
        "region_tracking": ("true", "Propagate DINOv2 regions"),
        "initial_u": ("-1", "Initial pixel u for headless mode; -1 waits for click"),
        "initial_v": ("-1", "Initial pixel v for headless mode; -1 waits for click"),
        "publish_uncertain": ("false", "Publish poses with positive WBPS group error"),
        "sync_slop": ("0.02", "Maximum color-depth timestamp difference in seconds"),
        "max_frame_age": ("0.5", "Reject observations/output older than this many seconds"),
    }
    actions = []
    for name, (default, description) in arguments.items():
        kwargs = {"choices": ["true", "false"]} if default in ("true", "false") else {}
        actions.append(DeclareLaunchArgument(name, default_value=default, description=description, **kwargs))
    return LaunchDescription(actions + [OpaqueFunction(function=setup)])
