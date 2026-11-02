"""Capture RealSense images and UR TCP poses: Space saves; Q/Esc quits.

Example: uv run python src/calibration/data_collection.py --save-dir data/hand_eye
Keep the robot stationary during capture; camera and RTDE are not synchronized.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

import cv2
import numpy as np
import pyrealsense2 as rs
from rtde_receive import RTDEReceiveInterface as RTDEReceive

WIDTH, HEIGHT, FPS = 1280, 720, 30
WINDOW = "Hand-eye calibration | Space: capture | Q/Esc: quit"
POSE_FIELDS = [
    "sample_index", "image", "captured_at_utc", "x", "y", "z", "rx", "ry", "rz",
    *[f"T_base_tcp_{row}{col}" for row in range(4) for col in range(4)],
    "camera_frame_number", "camera_timestamp_ms", "camera_timestamp_domain",
    "tcp_read_started_unix_ns", "tcp_read_finished_unix_ns",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")


def save_pair(session: Path, index: int, frame, robot) -> Path:
    """Save an image and publish its matching row in the session CSV."""
    read_started_ns = time.time_ns()
    if not robot.isConnected():
        raise RuntimeError("Robot RTDE connection lost; pair was not saved.")
    pose = np.asarray(robot.getActualTCPPose(), dtype=float)
    read_finished_ns = time.time_ns()
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid TCP pose received: {pose}")
    image = np.asanyarray(frame.get_data())
    if image.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"Expected a {WIDTH}x{HEIGHT} color frame, got {image.shape}.")
    rotation, _ = cv2.Rodrigues(pose[3:])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = pose[:3]
    record = {
        "sample_index": index,
        "captured_at_utc": utc_now(),
        "image": f"images/{index:06d}.png",
        **dict(zip(("x", "y", "z", "rx", "ry", "rz"), pose.tolist())),
        **{f"T_base_tcp_{row}{col}": float(transform[row, col])
           for row in range(4) for col in range(4)},
        "camera_frame_number": frame.get_frame_number(),
        "camera_timestamp_ms": frame.get_timestamp(),
        "camera_timestamp_domain": str(frame.get_frame_timestamp_domain()),
        "tcp_read_started_unix_ns": read_started_ns,
        "tcp_read_finished_unix_ns": read_finished_ns,
    }
    destination = session / record["image"]
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    staging_image = session / "images" / f".pending_{uuid4().hex}.png"
    staging_csv = session / f".pending_{uuid4().hex}.csv"
    published_image = False
    try:
        if not cv2.imwrite(str(staging_image), image):
            raise OSError("Could not write the captured image.")
        # Replace the CSV atomically so a failed write preserves earlier rows.
        with staging_csv.open("w", encoding="utf-8", newline="") as output:
            with (session / "poses.csv").open("r", encoding="utf-8", newline="") as source:
                shutil.copyfileobj(source, output)
            csv.DictWriter(output, fieldnames=POSE_FIELDS).writerow(record)
        staging_image.rename(destination)
        published_image = True
        staging_csv.replace(session / "poses.csv")
    except BaseException:
        if published_image:
            destination.unlink(missing_ok=True)
        raise
    finally:
        staging_image.unlink(missing_ok=True)
        staging_csv.unlink(missing_ok=True)
    return destination


def collect(args: argparse.Namespace) -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    if args.camera_serial:
        config.enable_device(args.camera_serial)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    # SDK post-processing: rotate color frames for the upside-down mounting.
    image_rotation = rs.rotation_filter([rs.stream.color])
    image_rotation.set_option(rs.option.rotation, 180)
    robot = None
    started = False
    session = None
    count = 0
    try:
        robot = RTDEReceive(args.robot_ip)
        if not robot.isConnected():
            raise RuntimeError(f"Could not connect to robot at {args.robot_ip}")
        profile = pipeline.start(config)
        started = True
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        device = profile.get_device()
        args.save_dir.mkdir(parents=True, exist_ok=True)
        session = args.save_dir / (
            datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%S_%fZ_")
            + uuid4().hex[:8]
        )
        (session / "images").mkdir(parents=True)
        with (session / "poses.csv").open("x", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=POSE_FIELDS).writeheader()
        write_json(session / "session.json", {
            "schema_version": 2,
            "images_directory": "images",
            "poses_file": "poses.csv",
            "created_at_utc": utc_now(),
            "robot_ip": args.robot_ip,
            "requested_num_samples": args.num_samples,
            "camera": {
                "serial": device.get_info(rs.camera_info.serial_number),
                "width": WIDTH, "height": HEIGHT, "fps": FPS,
                "image_format": "PNG", "stream_format": "bgr8",
                "image_rotation_degrees": 180,
                # Preserve factory intrinsics in the native optical frame.
                # Map detected saved-image pixels back before pose estimation.
                "intrinsics_pixel_frame": "native_unrotated_color",
                "saved_pixel_to_native_pixel": [
                    [-1, 0, WIDTH - 1],
                    [0, -1, HEIGHT - 1],
                    [0, 0, 1],
                ],
                "intrinsics": {
                    "fx": intrinsics.fx, "fy": intrinsics.fy,
                    "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
                    "distortion_model": str(intrinsics.model),
                    "distortion_coefficients": list(intrinsics.coeffs),
                },
            },
            "pose_convention": {
                "tcp_pose": "[x, y, z, rx, ry, rz] in robot base frame",
                "translation_unit": "metres",
                "orientation": "rotation vector (axis multiplied by angle in radians)",
                "T_base_tcp": "4x4 homogeneous matrix mapping TCP coordinates to robot base",
                "tool_frame": "TCP configured on the robot at collection time",
            },
            "synchronization": "Sequential camera/RTDE reads, not hardware synchronized. Keep robot stationary for capture.",
        })
        print(f"Saving pairs to: {session.resolve()}")
        print("Move the robot manually, stop, then press Space in the preview.")
        print("Q / Esc or Ctrl+C stops collection. No robot motion commands are sent.")
        # Allow automatic exposure to settle before the first capture.
        for _ in range(30):
            pipeline.wait_for_frames(timeout_ms=5000)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        while args.num_samples is None or count < args.num_samples:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            frame = frames.get_color_frame()
            if not frame:
                continue
            frame = image_rotation.process(frame)
            preview = np.asanyarray(frame.get_data()).copy()
            target = str(args.num_samples) if args.num_samples else "unlimited"
            cv2.putText(preview, f"Saved: {count}/{target} | Space: capture | Q/Esc: quit",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                # Acquire an image after the keypress and read TCP immediately.
                capture_frame = pipeline.wait_for_frames(timeout_ms=5000).get_color_frame()
                if not capture_frame:
                    print("No color frame available; try capturing again.")
                    continue
                capture_frame = image_rotation.process(capture_frame)
                destination = save_pair(session, count, capture_frame, robot)
                count += 1
                print(f"Saved pair {count}: {destination.name}")
    except KeyboardInterrupt:
        print("\nCollection stopped.")
    finally:
        # Release every resource even if another cleanup operation fails.
        try:
            if started:
                pipeline.stop()
        finally:
            try:
                if robot is not None:
                    robot.disconnect()
            finally:
                cv2.destroyAllWindows()
                if session is not None:
                    print(f"Saved {count} pair(s) in {session.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, required=True,
                        help="Parent directory for a new calibration session.")
    parser.add_argument("--robot-ip", default="192.168.1.130")
    parser.add_argument("--camera-serial", help="RealSense serial (optional for a single camera).")
    parser.add_argument("--num-samples", type=int,
                        help="Stop after this many pairs; omit to collect until quit.")
    args = parser.parse_args()
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be greater than zero")
    return args


if __name__ == "__main__":
    try:
        collect(parse_args())
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"Collection failed: {error}", file=sys.stderr)
        sys.exit(1)
