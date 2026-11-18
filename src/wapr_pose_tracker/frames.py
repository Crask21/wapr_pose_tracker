"""Source-independent RGB-D observation passed to WAPR."""

from dataclasses import dataclass

import numpy as np


@dataclass
class RGBDFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    frame_number: int
    camera_timestamp_ms: float
    timestamp_domain: str
    received_unix_ns: int
