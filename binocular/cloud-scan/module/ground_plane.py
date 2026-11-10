"""Constrained floor fitting from confidence-2 stereo points."""

from dataclasses import dataclass

import numpy as np

from .map_geometry import gravity_rotation, sample_points


@dataclass(frozen=True)
class FloorEstimate:
    normal_cam: np.ndarray | None
    height_m: float | None
    rotation_grav_cam: np.ndarray | None
    fresh: bool
    usable: bool
    stale_s: float
    inliers: int
    candidates: int


class GroundPlaneEstimator:
    def __init__(self, config):
        self.settings = config["map"]
        self._rng = np.random.default_rng(17)
        self._normal = None
        self._height = None
        self._last_good_s = None

    def _estimate(self, points):
        s = self.settings
        if len(points) < s["floor_min_inliers"]:
            return None
        if len(points) > s["floor_ransac_samples"]:
            indices = self._rng.choice(len(points), s["floor_ransac_samples"], replace=False)
            points = points[indices]
        triplets = self._rng.integers(0, len(points), (s["floor_iterations"], 3))
        triples = points[triplets].astype(np.float64)
        normals = np.cross(triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-8
        normals[valid] /= lengths[valid, None]
        normals[normals[:, 1] > 0] *= -1
        prior = self._normal if self._normal is not None else np.array([0., -1., 0.])
        cos_gate = np.cos(np.deg2rad(s["floor_max_tilt_deg"]))
        valid &= (normals @ prior) >= cos_gate
        heights = -np.einsum("ij,ij->i", normals, triples[:, 0])
        valid &= (heights >= s["floor_min_height_m"]) & (heights <= s["floor_max_height_m"])
        if not np.any(valid):
            return None
        normals = normals[valid]
        heights = heights[valid]
        distance = np.abs(points.astype(np.float64) @ normals.T + heights)
        counts = np.count_nonzero(distance <= s["floor_inlier_m"], axis=0)
        best = int(np.argmax(counts + 0.01 * heights))
        inliers = distance[:, best] <= s["floor_inlier_m"]
        count = int(np.count_nonzero(inliers))
        if count < s["floor_min_inliers"] or count < s["floor_min_fraction"] * len(points):
            return None
        subset = points[inliers].astype(np.float64)
        center = subset.mean(axis=0)
        _, _, vt = np.linalg.svd(subset - center, full_matrices=False)
        normal = vt[-1]
        if normal[1] > 0:
            normal = -normal
        height = float(-normal @ center)
        if (normal @ prior < cos_gate or
                not s["floor_min_height_m"] <= height <= s["floor_max_height_m"]):
            return None
        refined_count = int(np.count_nonzero(np.abs(points @ normal + height) <= s["floor_inlier_m"]))
        return normal, height, refined_count

    def fit(self, result, timestamp_s):
        s = self.settings
        row_start = int(result.disparity.shape[0] * s["floor_image_start_fraction"])
        points, _, _ = sample_points(result, s["sample_stride"],
                                     s["floor_min_range_m"], s["floor_max_range_m"],
                                     row_start=row_start)
        estimate = self._estimate(points)
        fresh = estimate is not None
        inliers = 0
        if fresh:
            normal, height, inliers = estimate
            alpha = s["floor_smoothing"]
            if self._normal is not None:
                normal = alpha * normal + (1 - alpha) * self._normal
                normal /= np.linalg.norm(normal)
                height = alpha * height + (1 - alpha) * self._height
            self._normal, self._height = normal, height
            self._last_good_s = timestamp_s
        stale_s = (float("inf") if self._last_good_s is None else
                   max(0.0, timestamp_s - self._last_good_s))
        usable = self._normal is not None and stale_s <= s["floor_stale_s"]
        rotation = gravity_rotation(self._normal) if usable else None
        return FloorEstimate(self._normal if usable else None,
                             self._height if usable else None, rotation,
                             fresh, usable, stale_s, inliers, len(points))
