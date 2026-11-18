"""One mesh/instance pose tracker; accepts RGB-D frames from any source."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float32)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1]) or pose[2, 3] <= 0):
        raise ValueError("Pose must be finite, homogeneous, and in front of the camera")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.01) or not np.isclose(np.linalg.det(rotation), 1, atol=0.01):
        raise ValueError("Pose rotation is not a proper rotation matrix")
    return pose.copy()


def select_mask(detections, click, shape, decode):
    """Select the highest-scoring CAD detection containing the user's click."""
    u, v = click
    if not (0 <= u < shape[1] and 0 <= v < shape[0]):
        raise ValueError("Click outside image")
    candidates = []
    for detection in detections:
        mask = np.asarray(decode(detection["mask"])) > 0
        if mask.shape != shape:
            raise ValueError("Detector mask dimensions differ from RGB-D frame")
        if mask[v, u] and np.count_nonzero(mask) >= 150:
            candidates.append((float(detection["score_2d"]), mask))
    if not candidates:
        raise ValueError("No CAD-matched mask at click; click inside the object and retry")
    return max(candidates, key=lambda item: item[0])[1]


@dataclass
class PoseEstimate:
    pose_4x4: np.ndarray
    status: str
    between_group: float | None = None
    hypothesis: str | None = None


class ObjectTracker:
    def __init__(self, mesh_path: Path, mesh_scale=1.0, diameter_m=None,
                 device="cuda:0", region_tracking=True):
        import torch
        import trimesh
        from wapr import WAPREstimator, recipe
        from wapr.estimator import prepare_mesh

        if not torch.cuda.is_available():
            raise RuntimeError("WAPR requires CUDA-enabled PyTorch and an NVIDIA GPU")
        recipe.visualize = False
        raw = trimesh.load(str(mesh_path), force="mesh", process=False)
        if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) == 0:
            raise ValueError("Model must contain a triangle mesh")
        raw.apply_scale(mesh_scale)
        self.vertices = np.asarray(raw.vertices, dtype=np.float64).copy()
        if not np.isfinite(self.vertices).all():
            raise ValueError("Mesh contains nonfinite coordinates")
        self.bounds = np.asarray(raw.bounds).copy()
        # Conservative diameter bound avoids a quadratic all-vertex distance calculation.
        self.diameter = float(diameter_m or np.linalg.norm(raw.extents))
        if not np.isfinite(self.diameter) or self.diameter <= 0:
            raise ValueError("Mesh diameter must be finite and positive")
        mesh = prepare_mesh(raw)
        mesh.metadata["name"] = mesh_path.stem
        self.estimator = WAPREstimator(device=device)
        self.mesh = self.estimator.prepare_meshes([mesh])[0]
        self.device = device
        self.detector = None
        self.dino = None
        if region_tracking:
            from wapr.region_tracking import load_tracker_dino
            self.dino = load_tracker_dino(device, dino_name="vits14")
        self.reset()

    def reset(self):
        self.pose = None
        self.prev_tokens = self.prev_patch = None
        self.template_tokens = self.template_patch = None

    def initialize(self, frame, click):
        from pycocotools import mask as mask_utils
        from wapr.det2d import WAPRDet2D, onboard_meshes
        from wapr.region_tracking import dino_tokens, one_instance_mask_to_patches

        self.reset()
        if self.detector is None:
            bank = onboard_meshes({1: self.mesh}, device=self.device)
            self.detector = WAPRDet2D(bank, device=self.device, backend="trt")
            from .sam_compat import PromptAlignedSAM
            self.detector.proposals.segmentor = PromptAlignedSAM(self.detector.proposals.segmentor)
        detections, _timing = self.detector.detect_many_categories_many_instances(frame.rgb)
        mask = select_mask(detections, click, frame.depth_m.shape, mask_utils.decode)
        if np.count_nonzero(mask & (frame.depth_m > 0)) < 75:
            raise ValueError("Insufficient valid depth on the selected object")
        result = self.estimator.estimate_one_category_one_instance(
            frame.rgb, frame.depth_m, frame.K, self.mesh, self.diameter,
            mask=mask.astype(np.uint8) * 255,
        )
        pose = validate_pose(result["pose_4x4"])
        if self.dino is not None:
            tokens, grid = dino_tokens(self.dino, frame.rgb, self.device)
            patch = one_instance_mask_to_patches(mask, grid)
            if np.count_nonzero(patch) < 8:
                raise ValueError("Object too small for DINOv2 region tracking; move closer")
            self.prev_tokens = self.template_tokens = tokens
            self.prev_patch = self.template_patch = patch
        self.pose = pose
        return PoseEstimate(pose.copy(), "initialized")

    def update(self, frame):
        from wapr.tracking import track_many_categories_many_instances
        from wapr.region_tracking import (dino_tokens, propagate_one_instance_region,
                                          one_instance_mask_to_patches, translation_from_one_instance)

        if self.pose is None:
            raise ValueError("Click to initialize before tracking")
        center_pose = self.pose.copy()
        if self.dino is not None:
            tokens, grid = dino_tokens(self.dino, frame.rgb, self.device)
            region = propagate_one_instance_region(
                self.prev_tokens, self.prev_patch, self.template_tokens, self.template_patch,
                tokens, grid, self.diameter * frame.K[0, 0] / self.pose[2, 3],
            )
            if region is None:
                self.reset()
                raise ValueError("Object region lost; click to reinitialize")
            mask, uv, _ = region
            translation = translation_from_one_instance(
                mask, uv, frame.depth_m, frame.K, self.pose[:3, :3], self.vertices,
            )
            if translation is None or not np.isfinite(translation).all():
                self.reset()
                raise ValueError("Object depth lost; click to reinitialize")
            center_pose[:3, 3] = translation
            self.prev_tokens, self.prev_patch = tokens, one_instance_mask_to_patches(mask, grid)
        row = track_many_categories_many_instances(
            self.estimator, frame.rgb, frame.depth_m, frame.K,
            [{"track_id": "object", "mesh": self.mesh, "diameter_m": self.diameter,
              "pose_4x4": self.pose, "center_pose_4x4": center_pose}],
        )[0]
        self.pose = validate_pose(row["pose_4x4"])
        error = float(row["between_group"])
        if not np.isfinite(error):
            self.reset()
            raise ValueError("Invalid pose quality; click to reinitialize")
        # WBPS group error is diagnostic, not a calibrated probability.
        status = "uncertain" if error > 0 else "tracking"
        return PoseEstimate(self.pose.copy(), status, error, row.get("hypothesis"))
