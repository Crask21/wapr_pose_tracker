"""OpenCV overlays shared by the standalone and ROS runners."""

import itertools

import cv2
import numpy as np


def draw_pose(image, pose, K, bounds, diameter, color):
    """Draw the original mesh's bounding box and XYZ axes in BGR."""
    corners = np.array(list(itertools.product(*zip(bounds[0], bounds[1]))))
    camera_points = corners @ pose[:3, :3].T + pose[:3, 3]
    if np.any(camera_points[:, 2] <= 0.01):
        return
    projected = camera_points @ K.T
    pixels = projected[:, :2] / projected[:, 2:]
    if not np.isfinite(pixels).all() or np.max(np.abs(pixels)) > 1e6:
        return
    pixels = np.rint(pixels).astype(int)
    for a in range(8):
        for bit in (1, 2, 4):
            b = a ^ bit
            if a < b:
                cv2.line(image, tuple(pixels[a]), tuple(pixels[b]), color, 2)
    rvec, _ = cv2.Rodrigues(pose[:3, :3].astype(np.float64))
    if pose[2, 3] > diameter:
        cv2.drawFrameAxes(image, K, np.zeros(5), rvec, pose[:3, 3], diameter * 0.3, 2)
