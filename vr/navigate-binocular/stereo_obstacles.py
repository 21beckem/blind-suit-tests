"""Convert calibrated stereo depth into a metric walking map.

Other sensors can produce the same LocalMap without using this adapter.
"""

import time

import numpy as np

from navigation import LocalMap


class StereoObstacles:
    def __init__(self, *, fx, fy, cx, cy, width, height,
                 near=0.20, far=3.0, x_range=1.8, cell_size=0.12,
                 min_hazard_points=3):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = width, height
        self.near, self.far = near, far
        self.min_hazard_points = min_hazard_points
        self.x_edges = np.arange(-x_range, x_range + cell_size, cell_size)
        self.z_edges = np.arange(near, far + cell_size, cell_size)
        self.debug = {}

    @staticmethod
    def _floor_plane(points):
        """Fit a plausible floor with bounded, sampled RANSAC work."""
        if len(points) < 80:
            return None, 0
        rng = np.random.default_rng(42)
        best = None
        best_count = 0
        for _ in range(48):
            a, b, c = points[rng.choice(len(points), 3, replace=False)]
            n = np.cross(b - a, c - a)
            length = np.linalg.norm(n)
            if length < 1e-6:
                continue
            n /= length
            if n[1] < 0:
                n = -n
            if n[1] < 0.70:
                continue
            d = -np.dot(n, a)
            if not 0.6 <= -d <= 2.2:
                continue
            count = np.count_nonzero(np.abs(points @ n + d) < 0.06)
            if count > best_count:
                best_count, best = count, (n, d)
        if best_count < max(80, 0.18 * len(points)):
            return None, best_count
        return best, best_count

    def observe(self, depth, *, timestamp=None):
        if depth.shape != (self.height, self.width):
            raise ValueError("depth shape does not match camera geometry")
        shape = (len(self.z_edges) - 1, len(self.x_edges) - 1)
        empty = np.zeros(shape, dtype=bool)
        stamp = time.monotonic() if timestamp is None else timestamp
        valid = np.isfinite(depth) & (depth >= self.near) & (depth <= self.far)
        labels = np.zeros(depth.shape, dtype=np.uint8)
        labels[valid] = 3  # valid depth, neither floor nor hazard
        self.debug = {
            "labels": labels,
            "valid_fraction": float(np.mean(valid)),
            "sample_count": 0,
            "floor_inliers": 0,
            "floor_found": False,
            "floor_status": "no valid depth",
            "camera_height": None,
            "floor_tilt_deg": None,
            "floor_cells": 0,
            "hazard_cells": 0,
        }
        if not np.any(valid):
            return LocalMap(self.x_edges, self.z_edges, empty.copy(), empty.copy(), stamp)

        v, u = np.nonzero(valid)
        z = depth[v, u]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points = np.column_stack((x, y, z))
        sample = (v > self.height * 0.38) & (u % 7 == 0) & (v % 7 == 0)
        self.debug["sample_count"] = int(np.count_nonzero(sample))
        self.debug["floor_status"] = "too few floor samples" if self.debug["sample_count"] < 80 else "candidate rejected"
        plane, inliers = self._floor_plane(points[sample])
        self.debug["floor_inliers"] = int(inliers)
        if plane is None:
            if self.debug["sample_count"] >= 80:
                self.debug["floor_status"] = "no plausible plane" if inliers == 0 else "low plane support"
            return LocalMap(self.x_edges, self.z_edges, empty.copy(), empty.copy(), stamp)

        n, d = plane
        self.debug["floor_found"] = True
        self.debug["floor_status"] = "floor fitted"
        self.debug["camera_height"] = float(-d)
        self.debug["floor_tilt_deg"] = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
        height_above_floor = -(points @ n + d)
        floor = np.abs(height_above_floor) < 0.075
        hazard = ((height_above_floor > 0.10) & (height_above_floor < 2.1)) | (height_above_floor < -0.15)
        labels[v[floor], u[floor]] = 1
        labels[v[hazard], u[hazard]] = 2
        xi = np.searchsorted(self.x_edges, x, side="right") - 1
        zi = np.searchsorted(self.z_edges, z, side="right") - 1
        inside = (xi >= 0) & (xi < shape[1]) & (zi >= 0) & (zi < shape[0])
        index = zi[inside] * shape[1] + xi[inside]
        floor_count = np.bincount(index[floor[inside]], minlength=empty.size).reshape(shape)
        hazard_count = np.bincount(index[hazard[inside]], minlength=empty.size).reshape(shape)
        self.debug["floor_cells"] = int(np.count_nonzero(floor_count >= 3))
        blocked = hazard_count >= self.min_hazard_points
        self.debug["hazard_cells"] = int(np.count_nonzero(blocked))
        return LocalMap(self.x_edges, self.z_edges, floor_count >= 3, blocked, stamp)
