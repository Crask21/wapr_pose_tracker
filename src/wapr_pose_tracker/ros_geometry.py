"""ROS image decoding and calibration math, usable without a running ROS graph."""

from pathlib import Path

import cv2
import numpy as np
import yaml

ROTATE_180 = np.diag([-1., -1., 1., 1.])


def rigid_transform(value):
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-5)):
        raise ValueError("Calibration must contain a finite rigid 4x4 transform")
    return matrix


def load_world_from_camera(path, world_frame="world", direction="from_yaml"):
    """Return T_world_calibrated_camera; YAML defines p_target = T p_source."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config.get("translation", {}).get("unit") != "meters":
        raise ValueError("Calibration translation unit must be meters")
    matrix = rigid_transform(config["transform"])
    if direction == "from_yaml":
        source, target = config["source_frame"], config["target_frame"]
        if source == world_frame and target != world_frame:
            direction = "world_to_camera"
        elif target == world_frame and source != world_frame:
            direction = "camera_to_world"
        else:
            raise ValueError(f"Calibration must have exactly one frame named {world_frame!r}")
    if direction == "world_to_camera":
        return np.linalg.inv(matrix)
    if direction == "camera_to_world":
        return matrix.copy()
    raise ValueError("calibration_direction must be from_yaml, world_to_camera, or camera_to_world")


def world_object_pose(world_from_camera, camera_object, images_rotated, calibration_rotated):
    basis_change = ROTATE_180 if images_rotated != calibration_rotated else np.eye(4)
    return rigid_transform(world_from_camera) @ basis_change @ rigid_transform(camera_object)


def quaternion_xyzw(rotation):
    vector, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    vector = vector.reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.array([0., 0., 0., 1.])
    return np.r_[vector * (np.sin(angle / 2) / angle), np.cos(angle / 2)]


def image_array(message):
    """Decode uncompressed sensor_msgs/Image, including padding and byte order."""
    formats = {"rgb8": ("u1", 3), "bgr8": ("u1", 3),
               "16UC1": ("u2", 1), "32FC1": ("f4", 1)}
    if message.encoding not in formats:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    kind, channels = formats[message.encoding]
    dtype = np.dtype(kind).newbyteorder(">" if message.is_bigendian else "<")
    row_bytes = message.width * channels * dtype.itemsize
    if (message.width <= 0 or message.height <= 0 or message.step < row_bytes
            or len(message.data) < message.step * message.height):
        raise ValueError("Malformed image dimensions/stride/data")
    if channels == 3:
        shape, strides = (message.height, message.width, 3), (message.step, 3, 1)
    else:
        shape, strides = (message.height, message.width), (message.step, dtype.itemsize)
    result = np.ndarray(shape, dtype=dtype, buffer=message.data, strides=strides).copy()
    return result[..., ::-1].copy() if message.encoding == "bgr8" else result


class FrameDecoder:
    """Convert aligned ROS images to rectified WAPR observations.

    The launch supplies native aligned images and CameraInfo. Optional output
    rotation happens after rectification, preserving color/depth registration.
    images_rotated/camera_info_rotated also support external pre-rotated inputs;
    the provided launch never enables the driver's pre-alignment rotation.
    """

    def __init__(self, images_rotated=False, camera_info_rotated=False, rotate_180=False):
        self.images_rotated = images_rotated
        self.camera_info_rotated = camera_info_rotated
        self.rotate_180 = rotate_180
        self.signature = None
        self.maps = None

    def decode(self, color, depth, info):
        from .frames import RGBDFrame
        import time

        if not color.header.frame_id or any(
                msg.header.frame_id != color.header.frame_id for msg in (depth, info)):
            raise ValueError("RGB, aligned depth, and CameraInfo must share the color optical frame")
        if (color.width, color.height) != (depth.width, depth.height) or (
                color.width, color.height) != (info.width, info.height):
            raise ValueError("Aligned RGB-D and CameraInfo dimensions differ")
        if color.encoding not in ("rgb8", "bgr8") or depth.encoding not in ("16UC1", "32FC1"):
            raise ValueError("Expected rgb8/bgr8 color and 16UC1/32FC1 aligned depth")
        rgb = image_array(color)
        depth_m = image_array(depth).astype(np.float32)
        if depth.encoding == "16UC1":
            # realsense2_camera normalizes Z16 output to millimetres.
            depth_m *= 0.001
        depth_m[~np.isfinite(depth_m) | (depth_m < 0)] = 0
        signature = (info.width, info.height, tuple(info.k), tuple(info.d), info.distortion_model)
        if signature != self.signature:
            K = np.asarray(info.k, dtype=np.float32).reshape(3, 3).copy()
            D = np.asarray(info.d, dtype=np.float64).copy()
            if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.isfinite(D).all():
                raise ValueError("Invalid CameraInfo intrinsics")
            if self.images_rotated != self.camera_info_rotated:
                K[0, 2] = info.width - 1 - K[0, 2]
                K[1, 2] = info.height - 1 - K[1, 2]
                if np.any(D):
                    if info.distortion_model not in ("plumb_bob", "rational_polynomial") or len(D) < 5:
                        raise ValueError("Unsupported distortion model for rotated images")
                    D[2:4] *= -1
            self.maps = None
            if np.any(D):
                if info.distortion_model not in ("plumb_bob", "rational_polynomial"):
                    raise ValueError(f"Unsupported distortion: {info.distortion_model}")
                self.maps = cv2.initUndistortRectifyMap(K, D, None, K, (info.width, info.height), cv2.CV_32FC1)
            self.K, self.signature = K, signature
        if self.maps is not None:
            rgb = cv2.remap(rgb, *self.maps, interpolation=cv2.INTER_LINEAR)
            depth_m = cv2.remap(depth_m, *self.maps, interpolation=cv2.INTER_NEAREST)
        K = self.K.copy()
        if self.rotate_180:
            # Rotate only AFTER native depth-to-color alignment and rectification.
            # Both observations and their projection model change together.
            rgb = np.ascontiguousarray(rgb[::-1, ::-1])
            depth_m = np.ascontiguousarray(depth_m[::-1, ::-1])
            K[0, 2] = color.width - 1 - K[0, 2]
            K[1, 2] = color.height - 1 - K[1, 2]
        stamp = color.header.stamp
        return RGBDFrame(rgb, depth_m, K, 0,
                         stamp.sec * 1000 + stamp.nanosec / 1e6, "ros", time.time_ns())
