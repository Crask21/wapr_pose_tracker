"""RealSense acquisition, independent of WAPR and the preview UI."""

import time

import cv2
import numpy as np
import pyrealsense2 as rs


from .frames import RGBDFrame


class RealSenseCamera:
    """Aligned, rectified RGB-D in the native color optical frame.

    Depth is axial Z in metres. Frames are owned copies, safe after SDK reuse.
    The SDK queue holds only the latest frames when inference is slower than capture.
    """

    def __init__(self, serial=None, width=640, height=480, fps=30):
        self.serial, self.width, self.height, self.fps = serial, width, height, fps
        self.pipeline = rs.pipeline()
        self.started = False

    def __enter__(self):
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        try:
            profile = self.pipeline.start(config)
            self.started = True
            for sensor in profile.get_device().query_sensors():
                if sensor.supports(rs.option.frames_queue_size):
                    sensor.set_option(rs.option.frames_queue_size, 1)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
            self.maps = None
            if any(intr.coeffs):
                if intr.model != rs.distortion.brown_conrady:
                    raise ValueError(f"Unsupported color distortion: {intr.model}; requires Brown-Conrady or zero coefficients")
                self.maps = cv2.initUndistortRectifyMap(
                    self.K, np.array(intr.coeffs), None, self.K,
                    (intr.width, intr.height), cv2.CV_32FC1,
                )
            self.align = rs.align(rs.stream.color)
            for _ in range(15):
                self.pipeline.wait_for_frames(timeout_ms=5000)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def read(self):
        frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=5000))
        received = time.time_ns()
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("RealSense returned an incomplete RGB-D frameset")
        rgb = np.asanyarray(color.get_data()).copy()
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        if self.maps is not None:
            rgb = cv2.remap(rgb, *self.maps, interpolation=cv2.INTER_LINEAR)
            # Do not interpolate depth across object boundaries.
            depth_m = cv2.remap(depth_m, *self.maps, interpolation=cv2.INTER_NEAREST)
        return RGBDFrame(rgb, depth_m, self.K.copy(), color.get_frame_number(),
                         color.get_timestamp(), str(color.get_frame_timestamp_domain()), received)

    def __exit__(self, *_):
        if self.started:
            self.started = False
            self.pipeline.stop()
