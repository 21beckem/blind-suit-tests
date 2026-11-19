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

    def _estimate_from_prior(self, points):
        """Refit a spatially spread floor patch despite periodic stereo aliases."""
        s = self.settings
        if self._normal is None or self._height is None:
            return None
        near = points[np.abs(points @ self._normal + self._height) <=
                      s["floor_prior_band_m"]].astype(np.float64)
        if len(near) < s["floor_prior_min_inliers"]:
            return None
        if (np.ptp(near[:, 0]) < s["floor_prior_min_span_x_m"] or
                np.ptp(near[:, 2]) < s["floor_prior_min_span_z_m"]):
            return None
        center = near.mean(axis=0)
        _, _, vt = np.linalg.svd(near - center, full_matrices=False)
        normal = vt[-1]
        if normal[1] > 0:
            normal = -normal
        height = float(-normal @ center)
        inlier = np.abs(points @ normal + height) <= s["floor_inlier_m"]
        count = int(np.count_nonzero(inlier))
        if count < s["floor_prior_min_inliers"] or count < (
                s["floor_prior_min_fraction"] * len(points)):
            return None
        support = points[inlier]
        if (np.ptp(support[:, 0]) < s["floor_prior_min_span_x_m"] or
                np.ptp(support[:, 2]) < s["floor_prior_min_span_z_m"]):
            return None
        if (normal @ self._normal < np.cos(np.deg2rad(s["floor_prior_max_tilt_deg"])) or
                -normal[1] < np.cos(np.deg2rad(s["floor_max_absolute_tilt_deg"])) or
                abs(height - self._height) > s["floor_prior_max_height_change_m"] or
                not s["floor_min_height_m"] <= height <= s["floor_max_height_m"]):
            return None
        return normal, height, count

    def _estimate(self, points):
        s = self.settings
        if len(points) < s["floor_min_inliers"]:
            return None
        if len(points) > s["floor_ransac_samples"]:
            indices = self._rng.choice(len(points), s["floor_ransac_samples"], replace=False)
            points = points[indices]
        guided = self._estimate_from_prior(points)
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
        # A wall can rotate the previous-normal prior a little every frame.
        # Keep an independent upright bound so this cannot accumulate to 90°.
        upright_cos = np.cos(np.deg2rad(s["floor_max_absolute_tilt_deg"]))
        valid &= -normals[:, 1] >= upright_cos
        heights = -np.einsum("ij,ij->i", normals, triples[:, 0])
        valid &= (heights >= s["floor_min_height_m"]) & (heights <= s["floor_max_height_m"])
        if not np.any(valid):
            return guided
        normals = normals[valid]
        heights = heights[valid]
        distance = np.abs(points.astype(np.float64) @ normals.T + heights)
        counts = np.count_nonzero(distance <= s["floor_inlier_m"], axis=0)
        # Stair treads can make a mixed, slanted hypothesis score higher than
        # the true ground. Try the next supported candidate if refinement
        # rejects one, rather than losing gravity for the whole frame.
        scores = np.zeros(len(counts), np.float64)
        for index, count in enumerate(counts):
            if count >= s["floor_min_inliers"]:
                residual = float(np.median(distance[:, index][
                    distance[:, index] <= s["floor_inlier_m"]]))
                scores[index] = count / (1 + residual / s[
                    "floor_plane_precision_m"])
        for best in np.argsort(scores)[::-1]:
            count = int(counts[best])
            if count < s["floor_min_inliers"] or count < s["floor_min_fraction"] * len(points):
                break
            coarse = distance[:, best] <= s["floor_inlier_m"]
            median_residual = float(np.median(distance[:, best][coarse]))
            trim = min(s["floor_inlier_m"], max(
                s["floor_refine_trim_m"], 2 * median_residual))
            subset = points[distance[:, best] <= trim].astype(np.float64)
            if len(subset) < s["floor_min_inliers"]:
                continue
            center = subset.mean(axis=0)
            _, _, vt = np.linalg.svd(subset - center, full_matrices=False)
            normal = vt[-1]
            if normal[1] > 0:
                normal = -normal
            height = float(-normal @ center)
            if (normal @ prior < cos_gate or -normal[1] < upright_cos or
                    not s["floor_min_height_m"] <= height <= s["floor_max_height_m"]):
                continue
            refined_count = int(np.count_nonzero(
                np.abs(points @ normal + height) <= s["floor_inlier_m"]))
            return normal, height, refined_count
        return guided

    def _tread_orientation(self, points):
        """Keep gravity observable when no individual stair tread is a broad floor.

        This is available only after a conventional floor has established the
        camera-height reference. It updates orientation, never that reference.
        """
        s = self.settings
        if self._normal is None or len(points) < s["stair_orientation_min_points"]:
            return None
        if len(points) > s["floor_ransac_samples"]:
            points = points[self._rng.choice(
                len(points), s["floor_ransac_samples"], replace=False)]
        triplets = self._rng.integers(0, len(points), (s["floor_iterations"], 3))
        triples = points[triplets].astype(np.float64)
        normals = np.cross(triples[:, 1] - triples[:, 0],
                           triples[:, 2] - triples[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-8
        normals[valid] /= lengths[valid, None]
        normals[normals[:, 1] > 0] *= -1
        valid &= normals @ self._normal >= np.cos(np.deg2rad(
            s["floor_prior_max_tilt_deg"]))
        valid &= -normals[:, 1] >= np.cos(np.deg2rad(
            s["floor_max_absolute_tilt_deg"]))
        if not np.any(valid):
            return None
        normals = normals[valid]
        offsets = -np.einsum("ij,ij->i", normals, triples[valid, 0])
        inlier = np.abs(points.astype(np.float64) @ normals.T + offsets) <= s[
            "floor_inlier_m"]
        counts = inlier.sum(axis=0)
        for index in np.argsort(counts)[::-1]:
            if counts[index] < s["stair_orientation_min_points"]:
                break
            support = points[inlier[:, index]]
            if (np.ptp(support[:, 0]) >= s["stair_min_tread_width_m"] and
                    np.ptp(support[:, 2]) >= s["stair_min_tread_depth_m"]):
                return normals[index], int(counts[index])
        return None

    def fit(self, result, timestamp_s):
        s = self.settings
        row_start = int(result.disparity.shape[0] * s["floor_image_start_fraction"])
        points, _, _ = sample_points(result, s["sample_stride"],
                                     s["floor_min_range_m"], s["floor_max_range_m"],
                                     row_start=row_start)
        estimate = self._estimate(points)
        orientation_only = False
        if estimate is None:
            tread = self._tread_orientation(points)
            if tread is not None:
                estimate = (tread[0], self._height, tread[1])
                orientation_only = True
        fresh = estimate is not None
        inliers = 0
        if fresh:
            normal, height, inliers = estimate
            alpha = s["floor_smoothing"]
            if self._normal is not None:
                normal = alpha * normal + (1 - alpha) * self._normal
                normal /= np.linalg.norm(normal)
                height = (self._height if orientation_only else
                          alpha * height + (1 - alpha) * self._height)
            self._normal, self._height = normal, height
            self._last_good_s = timestamp_s
        stale_s = (float("inf") if self._last_good_s is None else
                   max(0.0, timestamp_s - self._last_good_s))
        usable = self._normal is not None and stale_s <= s["floor_stale_s"]
        rotation = gravity_rotation(self._normal) if usable else None
        return FloorEstimate(self._normal if usable else None,
                             self._height if usable else None, rotation,
                             fresh, usable, stale_s, inliers, len(points))
