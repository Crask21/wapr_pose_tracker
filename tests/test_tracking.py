"""Hardware-free checks for coordinate, selection, and acquisition contracts."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from wapr_pose_tracker.camera import RealSenseCamera
from wapr_pose_tracker.cli import draw_pose, pose_record
from wapr_pose_tracker.tracker import ObjectTracker, PoseEstimate, select_mask, validate_pose


class TrackingTests(unittest.TestCase):
    def test_initialization_unpacks_detector_instances_and_timing(self):
        tracker = ObjectTracker.__new__(ObjectTracker)
        tracker.mesh, tracker.device, tracker.diameter, tracker.dino = object(), "cuda:0", 0.1, None
        mask = np.ones((20, 20), dtype=bool)
        tracker.detector = Mock()
        tracker.detector.detect_many_categories_many_instances.return_value = (
            [{"mask": "encoded-mask", "score_2d": 0.8}], {"total_ms": 12.0},
        )
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 1
        tracker.estimator = Mock()
        tracker.estimator.estimate_one_category_one_instance.return_value = {"pose_4x4": pose}
        frame = SimpleNamespace(rgb=np.zeros((20, 20, 3), dtype=np.uint8),
                                depth_m=np.ones((20, 20)), K=np.eye(3))
        codec = SimpleNamespace(decode=Mock(return_value=mask))
        modules = {
            "pycocotools": SimpleNamespace(mask=codec),
            "wapr.det2d": SimpleNamespace(WAPRDet2D=Mock(), onboard_meshes=Mock()),
            "wapr.region_tracking": SimpleNamespace(dino_tokens=Mock(), one_instance_mask_to_patches=Mock()),
        }
        with patch.dict("sys.modules", modules):
            result = tracker.initialize(frame, (10, 10))
            self.assertEqual(result.status, "initialized")
            np.testing.assert_array_equal(result.pose_4x4, pose)
            codec.decode.assert_called_once_with("encoded-mask")
            passed_mask = tracker.estimator.estimate_one_category_one_instance.call_args.kwargs["mask"]
            np.testing.assert_array_equal(passed_mask, np.full((20, 20), 255, dtype=np.uint8))
            tracker.detector.detect_many_categories_many_instances.return_value = ([], {"total_ms": 1})
            with self.assertRaisesRegex(ValueError, "No CAD-matched mask"):
                tracker.initialize(frame, (10, 10))
            self.assertIsNone(tracker.pose)

    def test_region_translation_is_passed_to_refinement_with_previous_rotation(self):
        tracker = ObjectTracker.__new__(ObjectTracker)
        tracker.pose = np.eye(4, dtype=np.float32)
        tracker.pose[2, 3] = 1
        tracker.mesh, tracker.estimator, tracker.device = object(), object(), "cuda:0"
        tracker.vertices = np.zeros((8, 3))
        tracker.diameter, tracker.dino = 0.2, object()
        tracker.prev_tokens = tracker.prev_patch = object()
        tracker.template_tokens = tracker.template_patch = object()
        frame = SimpleNamespace(rgb=np.zeros((20, 20, 3)), depth_m=np.ones((20, 20)), K=np.eye(3))
        selected = tracker.pose.copy()
        selected[:3, 3] = [0.1, 0.2, 1.1]
        tracking = SimpleNamespace(track_many_categories_many_instances=Mock(return_value=[{
            "pose_4x4": selected, "between_group": 1.5, "hypothesis": "center+WAPR",
        }]))
        regions = SimpleNamespace(
            dino_tokens=Mock(return_value=("tokens", {})),
            propagate_one_instance_region=Mock(return_value=(np.ones((20, 20)), (10, 10), 12)),
            one_instance_mask_to_patches=Mock(return_value="patch"),
            translation_from_one_instance=Mock(return_value=np.array([0.1, 0.2, 1.1])),
        )
        with patch.dict("sys.modules", {"wapr.tracking": tracking, "wapr.region_tracking": regions}):
            result = tracker.update(frame)
            call = tracking.track_many_categories_many_instances.call_args.args[-1][0]
            np.testing.assert_array_equal(call["center_pose_4x4"][:3, :3], np.eye(3))
            np.testing.assert_array_equal(call["pose_4x4"][:3, 3], [0, 0, 1])
            self.assertEqual(result.status, "uncertain")
            np.testing.assert_array_equal(result.pose_4x4, selected)
            regions.propagate_one_instance_region.return_value = None
            with self.assertRaisesRegex(ValueError, "region lost"):
                tracker.update(frame)
            self.assertIsNone(tracker.pose)
            self.assertIsNone(tracker.template_tokens)
            # No stale refinement/output after a missing region.
            tracking.track_many_categories_many_instances.assert_called_once()

    def test_rejects_bad_transforms(self):
        pose = np.eye(4)
        pose[2, 3] = 1
        np.testing.assert_array_equal(validate_pose(pose), pose)
        for change in (lambda p: p.__setitem__((2, 3), -1),
                       lambda p: p.__setitem__((0, 0), -1),
                       lambda p: p.__setitem__((0, 1), np.nan),
                       lambda p: p.__setitem__((3, 3), 0)):
            invalid = pose.copy()
            change(invalid)
            with self.assertRaises(ValueError):
                validate_pose(invalid)

    def test_selects_highest_score_at_click_not_highest_in_image(self):
        mask = np.zeros((30, 40), dtype=bool)
        mask[:20, :20] = True
        other = np.zeros_like(mask)
        other[10:, 20:] = True
        detections = [{"mask": other, "score_2d": 1},
                      {"mask": mask, "score_2d": 0.5}]
        chosen = select_mask(detections, (5, 5), mask.shape, lambda x: x)
        np.testing.assert_array_equal(chosen, mask)
        with self.assertRaisesRegex(ValueError, "No CAD-matched mask"):
            select_mask(detections, (39, 0), mask.shape, lambda x: x)

    def test_camera_converts_depth_and_owns_buffers(self):
        rgb = np.full((2, 3, 3), 12, dtype=np.uint8)
        depth = np.array([[0, 1000, 2000], [500, 600, 700]], dtype=np.uint16)
        color_frame = Mock()
        color_frame.get_data.return_value = rgb
        color_frame.get_frame_number.return_value = 42
        color_frame.get_timestamp.return_value = 123.5
        color_frame.get_frame_timestamp_domain.return_value = "hardware_clock"
        depth_frame = Mock()
        depth_frame.get_data.return_value = depth
        frames = Mock()
        frames.get_color_frame.return_value = color_frame
        frames.get_depth_frame.return_value = depth_frame
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.pipeline = Mock()
        camera.align = Mock()
        camera.align.process.return_value = frames
        camera.maps, camera.depth_scale, camera.K = None, 0.001, np.eye(3)
        result = camera.read()
        np.testing.assert_allclose(result.depth_m, depth * 0.001)
        rgb[:] = 0
        self.assertEqual(int(result.rgb[0, 0, 0]), 12)
        self.assertEqual(result.frame_number, 42)
        self.assertEqual(result.timestamp_domain, "hardware_clock")

    def test_camera_stops_after_setup_failure(self):
        pipeline = Mock()
        pipeline.start.return_value.get_device.side_effect = RuntimeError("disconnected")
        with patch("wapr_pose_tracker.camera.rs.pipeline", return_value=pipeline):
            camera = RealSenseCamera()
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                camera.__enter__()
        pipeline.stop.assert_called_once()

    def test_pose_log_preserves_object_origin_and_capture_time(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.1, -0.2, 1.2]
        frame = SimpleNamespace(frame_number=7, camera_timestamp_ms=50,
                                timestamp_domain="hardware_clock", received_unix_ns=100)
        record = pose_record(frame, PoseEstimate(pose, "uncertain", 2.0), 30)
        self.assertEqual(record["pose_4x4"], pose.tolist())
        self.assertEqual(record["received_unix_ns"], 100)
        self.assertEqual(record["status"], "uncertain")
        self.assertEqual(record["transform"], "T_camera_object")

    def test_overlay_projects_original_mesh_bounds(self):
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 1
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
        bounds = np.array([[0.1, -0.05, -0.05], [0.2, 0.05, 0.05]])
        with patch("wapr_pose_tracker.cli.cv2.line") as line:
            draw_pose(image, pose, K, bounds, 0.2, (0, 255, 0))
        self.assertEqual(line.call_count, 12)
        # The mesh is offset along X in its file: box must not be recentered.
        self.assertGreater(line.call_args_list[0].args[1][0], 320)


if __name__ == "__main__":
    unittest.main()
