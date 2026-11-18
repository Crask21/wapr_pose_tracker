"""ROS 2 RGB-D subscriber and world-frame object pose publisher.

    CUDA/OpenGL and GUI run on the main thread. The ROS executor only receives
    synchronized observations into a single-slot mailbox, never an inference FIFO.
"""

from pathlib import Path
from threading import Lock, Thread
import time

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

from .visualization import draw_pose
from .ros_geometry import FrameDecoder, load_world_from_camera, quaternion_xyzw, world_object_pose
from .tracker import ObjectTracker

WINDOW = "WAPR ROS 2 | click: initialize | R: reset | Q/Esc: quit"


class WAPRPoseTrackerNode(Node):
    def __init__(self):
        super().__init__("wapr_pose_tracker")
        defaults = {
            "mesh_path": "", "mesh_scale": 1.0, "diameter_m": 0.0, "device": "cuda:0",
            "calibration_file": "", "calibration_direction": "from_yaml", "world_frame": "world",
            "calibration_camera_rotated": True, "images_rotated": False, "camera_info_rotated": False,
            "rotate_180": True,
            "color_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "pose_topic": "~/pose", "region_tracking": True, "display": True,
            "initial_click": [-1, -1], "sync_slop": 0.02, "max_frame_age": 0.5,
            "publish_uncertain": False,
        }
        self.p = {key: self.declare_parameter(key, value).value for key, value in defaults.items()}
        mesh = Path(self.p["mesh_path"]).expanduser()
        if not mesh.is_file():
            raise ValueError(f"mesh_path is not a file: {mesh}")
        if not Path(self.p["calibration_file"]).is_file():
            raise ValueError("calibration_file must name the hand-eye YAML")
        for name in ("mesh_scale", "max_frame_age", "sync_slop"):
            if not np.isfinite(self.p[name]) or self.p[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.p["diameter_m"]) or self.p["diameter_m"] < 0:
            raise ValueError("diameter_m must be zero (automatic) or positive")
        if len(self.p["initial_click"]) != 2:
            raise ValueError("initial_click must contain [u, v] pixel coordinates")
        self.world_from_camera = load_world_from_camera(
            self.p["calibration_file"], self.p["world_frame"], self.p["calibration_direction"])
        self.decoder = FrameDecoder(self.p["images_rotated"], self.p["camera_info_rotated"], self.p["rotate_180"])
        self.lock = Lock()
        self.latest = self.info = None
        self.reset_requested = False
        self.initial_click_pending = True
        self.last_stamp = None
        self.last_profile = None
        self.display_frame = None
        self.pending_click = None
        self.message = "Waiting for synchronized camera topics; click object to initialize"
        self.publisher = self.create_publisher(PoseStamped, self.p["pose_topic"], 1)
        self.create_service(Trigger, "~/reset", self.request_reset)
        self.create_subscription(CameraInfo, self.p["camera_info_topic"], self.receive_info, qos_profile_sensor_data)
        self.color_sub = message_filters.Subscriber(self, Image, self.p["color_topic"], qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, self.p["depth_topic"], qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=5, slop=self.p["sync_slop"])
        self.sync.registerCallback(self.receive_images)
        self.get_logger().info("Loading WAPR; CUDA/OpenGL initialization may take a while")
        self.tracker = ObjectTracker(mesh, self.p["mesh_scale"], self.p["diameter_m"] or None,
                                     self.p["device"], region_tracking=self.p["region_tracking"])
        if self.p["display"]:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, self.mouse)
        self.get_logger().info(
            f"Publishing {self.p['pose_topic']} in {self.p['world_frame']}; "
            f"calibration_direction={self.p['calibration_direction']}, "
            f"images_rotated={self.p['images_rotated']}, "
            f"post_alignment_rotate_180={self.p['rotate_180']}, "
            f"calibration_camera_rotated={self.p['calibration_camera_rotated']}")
        self.get_logger().info(
            f"Listening for RGB {self.p['color_topic']}, aligned depth {self.p['depth_topic']}, "
            f"CameraInfo {self.p['camera_info_topic']}")

    def receive_info(self, message):
        with self.lock:
            self.info = message

    def receive_images(self, color, depth):
        with self.lock:
            if self.info is not None:
                self.latest = (color, depth, self.info)

    def request_reset(self, request, response):
        with self.lock:
            self.reset_requested = True
        response.success, response.message = True, "Reset queued; click or configured initial_click will initialize"
        return response

    def mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and self.display_frame is not None:
            # Capture the exact observation shown when the click was made.
            self.pending_click = (self.display_frame, (x, y))

    def fresh(self, stamp):
        age = (self.get_clock().now().nanoseconds - (stamp.sec * 10**9 + stamp.nanosec)) / 1e9
        return -self.p["sync_slop"] <= age <= self.p["max_frame_age"]

    def process_latest(self):
        with self.lock:
            messages, self.latest = self.latest, None
            reset, self.reset_requested = self.reset_requested, False
        if reset:
            self.tracker.reset()
            self.initial_click_pending = True
            self.pending_click = None
            self.message = "Reset: click object to initialize"
        selected = self.pending_click
        self.pending_click = None
        if selected is not None:
            (frame, stamp), click = selected
            self.infer(frame, stamp, click)
        elif messages is not None:
            color, depth, info = messages
            stamp = color.header.stamp
            stamp_key = (stamp.sec, stamp.nanosec)
            if stamp_key != self.last_stamp and self.fresh(stamp):
                self.last_stamp = stamp_key
                try:
                    frame = self.decoder.decode(color, depth, info)
                    profile = (color.header.frame_id, self.decoder.signature)
                    if profile != self.last_profile:
                        self.get_logger().info(
                            f"Tracking geometry {color.width}x{color.height}: "
                            f"fx={frame.K[0, 0]:.3f}, fy={frame.K[1, 1]:.3f}, "
                            f"cx={frame.K[0, 2]:.3f}, cy={frame.K[1, 2]:.3f}")
                    if self.last_profile is not None and profile != self.last_profile:
                        self.tracker.reset()
                        self.message = "Camera calibration/profile changed; click to initialize"
                    self.last_profile = profile
                    click = None
                    if self.initial_click_pending:
                        self.initial_click_pending = False
                        if min(self.p["initial_click"]) >= 0:
                            click = tuple(self.p["initial_click"])
                    self.infer(frame, stamp, click)
                except ValueError as error:
                    self.tracker.reset()
                    self.message = str(error)
                    self.get_logger().warning(self.message, throttle_duration_sec=2)
        if self.p["display"]:
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False
            if key in (ord("r"), ord("R")):
                with self.lock:
                    self.reset_requested = True
        return True

    def infer(self, frame, stamp, click=None):
        estimate = None
        if click is not None or self.tracker.pose is not None:
            try:
                if click is not None:
                    self.get_logger().info("Initializing on selected image; keep object/camera stationary")
                    estimate = self.tracker.initialize(frame, click)
                else:
                    estimate = self.tracker.update(frame)
            except ValueError as error:
                self.tracker.reset()
                self.message = str(error)
                self.get_logger().warning(self.message)
        if estimate is not None:
            self.message = estimate.status
            if estimate.status == "uncertain" and not self.p["publish_uncertain"]:
                self.get_logger().warning("Uncertain pose withheld; inspect overlay or reinitialize", throttle_duration_sec=2)
            elif self.fresh(stamp):
                transform = world_object_pose(self.world_from_camera, estimate.pose_4x4,
                                              self.p["images_rotated"] != self.p["rotate_180"],
                                              self.p["calibration_camera_rotated"])
                message = PoseStamped()
                message.header.stamp = stamp
                message.header.frame_id = self.p["world_frame"]
                message.pose.position.x, message.pose.position.y, message.pose.position.z = map(float, transform[:3, 3])
                q = quaternion_xyzw(transform[:3, :3])
                message.pose.orientation.x, message.pose.orientation.y, message.pose.orientation.z, message.pose.orientation.w = map(float, q)
                self.publisher.publish(message)
            else:
                self.get_logger().warning("Pose exceeded max_frame_age; withholding stale output", throttle_duration_sec=2)
        if self.p["display"]:
            preview = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            if estimate is not None:
                color = (0, 165, 255) if estimate.status == "uncertain" else (0, 255, 0)
                draw_pose(preview, estimate.pose_4x4, frame.K, self.tracker.bounds, self.tracker.diameter, color)
            cv2.putText(preview, self.message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            self.display_frame = (frame, stamp)
            cv2.imshow(WINDOW, preview)


def main(args=None):
    rclpy.init(args=args)
    node = executor = thread = None
    try:
        node = WAPRPoseTrackerNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def spin():
            try:
                executor.spin()
            except ExternalShutdownException:
                pass

        thread = Thread(target=spin, name="ros-rgbd-receiver")
        thread.start()
        while rclpy.ok() and thread.is_alive() and node.process_latest():
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join()
        if node is not None:
            node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
