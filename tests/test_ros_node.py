"""Uses real generated ROS messages; no camera, GPU, or ROS graph is required."""

from threading import Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

try:
    from builtin_interfaces.msg import Time
    from wapr_pose_tracker.ros_node import WAPRPoseTrackerNode
except ImportError:
    WAPRPoseTrackerNode = None


@unittest.skipIf(WAPRPoseTrackerNode is None, "Source /opt/ros/jazzy/setup.bash for ROS message tests")
class ROSNodeTests(unittest.TestCase):
    def node(self):
        pose = np.eye(4)
        pose[2, 3] = 0.5
        tracker = Mock()
        tracker.pose = pose
        tracker.update.return_value = SimpleNamespace(pose_4x4=pose, status="tracking")
        return SimpleNamespace(tracker=tracker, p={"display": False, "images_rotated": True,
                                                 "calibration_camera_rotated": True, "rotate_180": False,
                                                 "publish_uncertain": False, "world_frame": "world"},
                               world_from_camera=np.eye(4), fresh=lambda stamp: True,
                               publisher=Mock(), get_logger=Mock(), message="")

    def test_world_pose_and_source_timestamp_published(self):
        node = self.node()
        node.world_from_camera[:3, 3] = [1, 2, 3]
        stamp = Time(sec=42, nanosec=123456)
        WAPRPoseTrackerNode.infer(node, object(), stamp)
        msg = node.publisher.publish.call_args.args[0]
        self.assertEqual(msg.header.frame_id, "world")
        self.assertEqual(msg.header.stamp, stamp)
        self.assertEqual([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], [1, 2, 3.5])
        self.assertEqual(msg.pose.orientation.w, 1)

    def test_no_stale_uncertain_or_lost_pose_publication(self):
        node = self.node()
        node.fresh = lambda stamp: False
        WAPRPoseTrackerNode.infer(node, object(), Time())
        node.publisher.publish.assert_not_called()

        node.fresh = lambda stamp: True
        node.tracker.update.return_value.status = "uncertain"
        WAPRPoseTrackerNode.infer(node, object(), Time())
        node.publisher.publish.assert_not_called()
        node.tracker.update.side_effect = ValueError("region lost")
        WAPRPoseTrackerNode.infer(node, object(), Time())
        node.tracker.reset.assert_called_once()
        node.publisher.publish.assert_not_called()

    def test_post_alignment_rotation_preserves_calibrated_pose_basis(self):
        node = self.node()
        node.p["images_rotated"] = False
        node.p["rotate_180"] = True
        node.tracker.update.return_value.pose_4x4[:3, 3] = [0.1, 0.2, 0.5]
        WAPRPoseTrackerNode.infer(node, object(), Time())
        pose = node.publisher.publish.call_args.args[0].pose
        np.testing.assert_allclose([pose.position.x, pose.position.y, pose.position.z], [0.1, 0.2, 0.5])

    def test_receiver_overwrites_pending_frame(self):
        node = SimpleNamespace(lock=Lock(), info="intrinsics", latest=None)
        WAPRPoseTrackerNode.receive_images(node, "old color", "old depth")
        WAPRPoseTrackerNode.receive_images(node, "new color", "new depth")
        self.assertEqual(node.latest, ("new color", "new depth", "intrinsics"))


if __name__ == "__main__":
    unittest.main()
