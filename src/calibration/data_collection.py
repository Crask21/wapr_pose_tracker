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
from uuid import uuid4

import cv2
import numpy as np
import pyrealsense2 as rs
from rtde_receive import RTDEReceiveInterface as RTDEReceive

WIDTH, HEIGHT, FPS = 1280, 720, 30
WINDOW = "Hand-eye calibration | Space: capture | Q/Esc: quit"


def save_camera_intrinsics(path: Path, intrinsics) -> None:
    """Export GUI-compatible YAML for the saved, 180-degree-rotated images."""
    coefficients = list(intrinsics.coeffs)
    # The GUI uses OpenCV's forward Brown-Conrady distortion model.
    if any(coefficients) and intrinsics.model != rs.distortion.brown_conrady:
        raise ValueError(
            f"Cannot export nonzero {intrinsics.model} coefficients as plumb_bob. "
            "Calibrate OpenCV intrinsics for this camera first."
        )
    # Rotating the optical frame around Z negates the tangential coefficients.
    coefficients[2] *= -1
    coefficients[3] *= -1
    cx = intrinsics.width - 1 - intrinsics.ppx
    cy = intrinsics.height - 1 - intrinsics.ppy
    fx, fy = intrinsics.fx, intrinsics.fy
    matrices = {
        "camera_matrix": (3, 3, [fx, 0., cx, 0., fy, cy, 0., 0., 1.]),
        "distortion_coefficients": (5, 1, coefficients),
        "rectification_matrix": (3, 3, [1., 0., 0., 0., 1., 0., 0., 0., 1.]),
        "projection_matrix": (3, 4, [fx, 0., cx, 0., 0., fy, cy, 0., 0., 0., 1., 0.]),
    }
    with path.open("w", encoding="utf-8") as file:
        file.write(f"camera_name: RealSense\nimage_width: {intrinsics.width}\n"
                   f"image_height: {intrinsics.height}\ndistortion_model: plumb_bob\n")
        for name, (rows, cols, data) in matrices.items():
            file.write(f"{name}:\n  rows: {rows}\n  cols: {cols}\n  dt: d\n"
                       f"  data: {json.dumps(data, allow_nan=False)}\n")


def save_pair(session: Path, index: int, frame, robot) -> Path:
    """Save an image and publish its matching row in the session CSV."""
    if not robot.isConnected():
        raise RuntimeError("Robot RTDE connection lost; pair was not saved.")
    pose = np.asarray(robot.getActualTCPPose(), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f"Invalid TCP pose received: {pose}")
    image = np.asanyarray(frame.get_data())
    if image.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(f"Expected a {WIDTH}x{HEIGHT} color frame, got {image.shape}.")
    # Row 1 corresponds to 1.png, row 2 to 2.png, etc.; no CSV header.
    destination = session / f"{index + 1}.png"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    staging_image = session / f".pending_{uuid4().hex}.png"
    staging_csv = session / f".pending_{uuid4().hex}.csv"
    published_image = False
    try:
        if not cv2.imwrite(str(staging_image), image):
            raise OSError("Could not write the captured image.")
        # Replace the CSV atomically so a failed write preserves earlier rows.
        with staging_csv.open("w", encoding="utf-8", newline="") as output:
            with (session / "robot_poses.csv").open("r", encoding="utf-8", newline="") as source:
                shutil.copyfileobj(source, output)
            csv.writer(output).writerow(pose.tolist())
        staging_image.rename(destination)
        published_image = True
        staging_csv.replace(session / "robot_poses.csv")
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
        args.save_dir.mkdir(parents=True, exist_ok=True)
        session = args.save_dir / (
            datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%S_%fZ_")
            + uuid4().hex[:8]
        )
        session.mkdir(parents=True)
        save_camera_intrinsics(session / "camera_intrinsics.yaml", intrinsics)
        (session / "robot_poses.csv").touch(exist_ok=False)
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
