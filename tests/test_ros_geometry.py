from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from wapr_pose_tracker.ros_geometry import (
    FrameDecoder, ROTATE_180, image_array, load_world_from_camera,
    quaternion_xyzw, rigid_transform, world_object_pose,
)

CALIBRATION = Path(__file__).resolve().parents[1] / "configs/hand_eye_cam_1.yaml"


def image(array, encoding, frame_id="camera_color_optical_frame"):
    return SimpleNamespace(width=array.shape[1], height=array.shape[0],
                           encoding=encoding, is_bigendian=False, step=array.strides[0], data=array.tobytes(),
                           header=SimpleNamespace(frame_id=frame_id, stamp=SimpleNamespace(sec=4, nanosec=500)))


class ROSGeometryTests(unittest.TestCase):
    def test_post_alignment_rotation_matches_native_rectification_exactly(self):
        height, width = 24, 32
        rng = np.random.default_rng(7)
        color = image(rng.integers(0, 255, (height, width, 3), dtype=np.uint8), "rgb8")
        depth = image(rng.integers(400, 1500, (height, width), dtype=np.uint16), "16UC1")
        K = np.array([[31., 0, 12.3], [0, 29., 9.7], [0, 0, 1]], dtype=np.float32)
        D = np.array([0.05, -0.02, 0.003, -0.002, 0.001])
        info = SimpleNamespace(width=width, height=height, header=color.header,
                               k=K.reshape(-1).tolist(), d=D.tolist(), distortion_model="plumb_bob")
        # This is the working standalone camera's native rectification recipe.
        maps = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), cv2.CV_32FC1)
        native_rgb = cv2.remap(image_array(color), *maps, interpolation=cv2.INTER_LINEAR)
        native_depth = cv2.remap(image_array(depth).astype(np.float32) * 0.001,
                                 *maps, interpolation=cv2.INTER_NEAREST)
        result = FrameDecoder(rotate_180=True).decode(color, depth, info)
        np.testing.assert_array_equal(result.rgb, native_rgb[::-1, ::-1])
        np.testing.assert_array_equal(result.depth_m, native_depth[::-1, ::-1])
        self.assertTrue(result.rgb.flags.c_contiguous)
        self.assertTrue(result.depth_m.flags.c_contiguous)
        # Off-axis 3D points must project to the exactly flipped pixel.
        point = np.array([0.13, -0.08, 0.7])
        native_pixel = K @ point
        native_pixel = native_pixel[:2] / native_pixel[2]
        rotated_pixel = result.K @ (ROTATE_180[:3, :3] @ point)
        rotated_pixel = rotated_pixel[:2] / rotated_pixel[2]
        np.testing.assert_allclose(rotated_pixel, [width - 1, height - 1] - native_pixel, atol=2e-6)

    def test_config_direction_and_rotated_frame_composition(self):
        world_camera = load_world_from_camera(CALIBRATION)
        camera_world = load_world_from_camera(CALIBRATION, direction="world_to_camera")
        np.testing.assert_allclose(world_camera @ camera_world, np.eye(4), atol=1e-12)
        camera_object = np.eye(4)
        camera_object[:3, 3] = [0.1, 0.2, 0.8]
        expected = world_camera @ camera_object
        np.testing.assert_allclose(world_object_pose(world_camera, camera_object, True, True), expected)
        # A native observation must first map into the rotated calibration frame.
        np.testing.assert_allclose(world_object_pose(world_camera, camera_object, False, True),
                                   world_camera @ ROTATE_180 @ camera_object)
        with self.assertRaises(ValueError):
            load_world_from_camera(CALIBRATION, world_frame="not_world")

    def test_measured_world_position_uses_camera_to_world_matrix_directly(self):
        world_camera = load_world_from_camera(CALIBRATION)
        old_world_position = np.array([0.3595592828381056, -0.9384731193571547,
                                       0.5045480600031037, 1.])
        # Recover the camera observation from the formerly inverted composition.
        camera_position = world_camera @ old_world_position
        camera_object = np.eye(4)
        camera_object[:3, 3] = camera_position[:3]
        corrected = world_object_pose(world_camera, camera_object, True, True)[:3, 3]
        np.testing.assert_allclose(corrected, [0.10470, -0.76358, 0.03711], atol=1e-5)
        self.assertLess(np.linalg.norm(corrected - [0.13, -0.76, 0.035]), 0.03)

    def test_invalid_calibration_rejected_and_negative_z_allowed(self):
        transform = np.eye(4)
        transform[2, 3] = -2
        rigid_transform(transform)  # World translations need not have positive Z.
        transform[0, 0] = 2
        with self.assertRaises(ValueError):
            rigid_transform(transform)

    def test_quaternion_at_identity_and_half_turn(self):
        np.testing.assert_allclose(quaternion_xyzw(np.eye(3)), [0, 0, 0, 1])
        q = quaternion_xyzw(ROTATE_180[:3, :3])
        np.testing.assert_allclose(np.abs(q), [0, 0, 1, 0], atol=1e-8)
        q = quaternion_xyzw(cv2.Rodrigues(np.array([0.3, -0.7, 0.1]))[0])
        self.assertAlmostEqual(np.linalg.norm(q), 1)

    def test_padded_and_big_endian_depth(self):
        msg = SimpleNamespace(width=2, height=2, step=6, is_bigendian=True,
                              encoding="16UC1", data=b"\x00\x01\x01\x00xx\x00\x02\x02\x00xx")
        np.testing.assert_array_equal(image_array(msg), [[1, 256], [2, 512]])
        msg.step = 2
        with self.assertRaises(ValueError):
            image_array(msg)

    def test_rotated_intrinsics_metric_depth_and_rgb_order(self):
        rgb = image(np.array([[[1, 2, 3]] * 4] * 3, dtype=np.uint8), "bgr8")
        depth = image(np.full((3, 4), 1200, dtype=np.uint16), "16UC1")
        info = SimpleNamespace(width=4, height=3, header=rgb.header,
                               k=[10., 0, 1.2, 0, 11., 0.7, 0, 0, 1], d=[0.] * 5, distortion_model="plumb_bob")
        result = FrameDecoder(images_rotated=True).decode(rgb, depth, info)
        np.testing.assert_array_equal(result.rgb[0, 0], [3, 2, 1])
        np.testing.assert_allclose(result.depth_m, 1.2)
        self.assertAlmostEqual(result.K[0, 2], 1.8, places=5)
        self.assertAlmostEqual(result.K[1, 2], 1.3, places=5)
        self.assertEqual(result.timestamp_domain, "ros")
        # If the driver supplies corrected CameraInfo, do not flip it twice.
        result = FrameDecoder(True, True).decode(rgb, depth, info)
        self.assertAlmostEqual(result.K[0, 2], 1.2, places=5)

    def test_float_depth_invalid_samples_and_alignment_rejection(self):
        color = image(np.zeros((2, 2, 3), dtype=np.uint8), "rgb8")
        depth = image(np.array([[np.nan, np.inf], [-1., 0.7]], dtype=np.float32), "32FC1")
        info = SimpleNamespace(width=2, height=2, header=color.header,
                               k=[10., 0, 1, 0, 10., 1, 0, 0, 1], d=[], distortion_model="plumb_bob")
        result = FrameDecoder(False).decode(color, depth, info)
        np.testing.assert_allclose(result.depth_m, [[0, 0], [0, 0.7]])
        depth.header.frame_id = "depth_optical_frame"
        with self.assertRaisesRegex(ValueError, "share the color optical frame"):
            FrameDecoder(False).decode(color, depth, info)


if __name__ == "__main__":
    unittest.main()
