"""Live RealSense single-object pose tracking."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .camera import RealSenseCamera
from .tracker import ObjectTracker
from .visualization import draw_pose

WINDOW = "WAPR | click object: initialize | R: reset | Q/Esc: quit"


def pose_record(frame, estimate, elapsed_ms):
    return {
        "frame_number": frame.frame_number,
        "camera_timestamp_ms": frame.camera_timestamp_ms,
        "timestamp_domain": frame.timestamp_domain,
        "received_unix_ns": frame.received_unix_ns,
        "parent_frame": "camera_color_optical_frame", "child_frame": "object",
        "transform": "T_camera_object", "translation_unit": "m",
        "status": estimate.status, "pose_4x4": estimate.pose_4x4.tolist(),
        "between_group": estimate.between_group, "hypothesis": estimate.hypothesis,
        "processing_ms": elapsed_ms,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="Triangle mesh (OBJ, PLY, STL, ...)")
    parser.add_argument("--mesh-scale", type=float, default=1.0,
                        help="Convert mesh units to metres: 1 for m, 0.001 for mm")
    parser.add_argument("--diameter-m", type=float,
                        help="Known object diameter in metres; default: mesh bounding-box diagonal")
    parser.add_argument("--camera-serial")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="Requested camera rate, not tracking rate")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-region-tracking", action="store_true",
                        help="Disable DINOv2 region propagation; use previous-pose refinement only")
    parser.add_argument("--pose-log", type=Path, help="Create a JSONL pose log; refuses to overwrite")
    args = parser.parse_args(argv)
    if not args.mesh.is_file():
        parser.error(f"Mesh not found: {args.mesh}")
    for name in ("mesh_scale", "diameter_m", "width", "height", "fps"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not args.device.startswith("cuda"):
        parser.error("--device must select a CUDA device")
    return args


def run(args):
    import torch

    print("Loading WAPR and mesh. First initialization can build engines/templates.", flush=True)
    tracker = ObjectTracker(args.mesh, args.mesh_scale, args.diameter_m, args.device,
                            region_tracking=not args.no_region_tracking)
    print(f"Mesh diameter used: {tracker.diameter:.4f} m. Check --mesh-scale.", flush=True)
    click = None

    def on_mouse(event, x, y, flags, userdata):
        nonlocal click
        if event == cv2.EVENT_LBUTTONDOWN:
            click = (x, y)

    with ExitStack() as stack:
        log = stack.enter_context(args.pose_log.open("x", encoding="utf-8")) if args.pose_log else None
        camera = stack.enter_context(RealSenseCamera(args.camera_serial, args.width, args.height, args.fps))
        stack.callback(cv2.destroyAllWindows)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, on_mouse)
        message = "Click inside the object to initialize"
        last_console = 0.0
        previous_frame_number = None
        while True:
            cycle_started = time.perf_counter()
            frame = camera.read()
            if frame.frame_number == previous_frame_number:
                continue
            previous_frame_number = frame.frame_number
            estimate = None
            elapsed_ms = 0.0
            if tracker.pose is not None:
                torch.cuda.synchronize(args.device)
                started = time.perf_counter()
                try:
                    estimate = tracker.update(frame)
                except ValueError as error:
                    tracker.reset()
                    message = str(error)
                    print(message, flush=True)
                torch.cuda.synchronize(args.device)
                elapsed_ms = (time.perf_counter() - started) * 1000
            preview = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            if estimate is not None:
                color = (0, 165, 255) if estimate.status == "uncertain" else (0, 255, 0)
                draw_pose(preview, estimate.pose_4x4, frame.K, tracker.bounds, tracker.diameter, color)
                message = f"{estimate.status} | processing {elapsed_ms:.1f} ms | click/R to reinitialize"
                translation = estimate.pose_4x4[:3, 3]
                cv2.putText(preview, f"XYZ [m]: {translation[0]:+.3f} {translation[1]:+.3f} {translation[2]:+.3f}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if log:
                    log.write(json.dumps(pose_record(frame, estimate, elapsed_ms), allow_nan=False) + "\n")
                    log.flush()
                if time.monotonic() - last_console > 1:
                    print(json.dumps(pose_record(frame, estimate, elapsed_ms), allow_nan=False), flush=True)
                    last_console = time.monotonic()
            cv2.putText(preview, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cycle_hz = 1 / max(time.perf_counter() - cycle_started, 1e-6)
            cv2.putText(preview, f"Loop before UI: {cycle_hz:.1f} Hz", (10, args.height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("r"), ord("R")):
                click = None
                tracker.reset()
                message = "Click inside the object to initialize"
            if click is not None:
                selected_click, click = click, None
                print("Initializing on displayed frame; keep object/camera stationary...", flush=True)
                torch.cuda.synchronize(args.device)
                started = time.perf_counter()
                try:
                    estimate = tracker.initialize(frame, selected_click)
                except ValueError as error:
                    tracker.reset()
                    message = str(error)
                    print(message, flush=True)
                    continue
                torch.cuda.synchronize(args.device)
                elapsed_ms = (time.perf_counter() - started) * 1000
                record = pose_record(frame, estimate, elapsed_ms)
                print(json.dumps(record, allow_nan=False), flush=True)
                if log:
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nTracking stopped.")
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        print(f"Tracking stopped: {error}", file=sys.stderr)
        print("For missing WAPR resources run: python -m wapr.bootstrap --feature core dinov2 det2d sam2",
              file=sys.stderr)
        raise SystemExit(1) from error
