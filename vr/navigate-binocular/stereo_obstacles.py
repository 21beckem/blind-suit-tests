"""Convert calibrated stereo depth into a metric walking map.

Other sensors can produce the same LocalMap without using this adapter.
"""

import time

import numpy as np

from navigation import LocalMap


class StereoObstacles:
    def __init__(self, *, fx, fy, cx, cy, width, height,
                 near=0.20, far=3.0, x_range=1.8, cell_size=0.12):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = width, height
        self.near, self.far = near, far
        self.x_edges = np.arange(-x_range, x_range + cell_size, cell_size)
        self.z_edges = np.arange(near, far + cell_size, cell_size)

    @staticmethod
    def _floor_plane(points):
        """Fit a plausible floor with bounded, sampled RANSAC work."""
        if len(points) < 80:
            return None
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
            return None
        return best

    def observe(self, depth, *, timestamp=None):
        if depth.shape != (self.height, self.width):
            raise ValueError("depth shape does not match camera geometry")
        shape = (len(self.z_edges) - 1, len(self.x_edges) - 1)
        empty = np.zeros(shape, dtype=bool)
        stamp = time.monotonic() if timestamp is None else timestamp
        valid = np.isfinite(depth) & (depth >= self.near) & (depth <= self.far)
        if not np.any(valid):
            return LocalMap(self.x_edges, self.z_edges, empty.copy(), empty.copy(), stamp)

        v, u = np.nonzero(valid)
        z = depth[v, u]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points = np.column_stack((x, y, z))
        sample = (v > self.height * 0.38) & (u % 7 == 0) & (v % 7 == 0)
        plane = self._floor_plane(points[sample])
        if plane is None:
            return LocalMap(self.x_edges, self.z_edges, empty.copy(), empty.copy(), stamp)

        n, d = plane
        height_above_floor = -(points @ n + d)
        floor = np.abs(height_above_floor) < 0.075
        hazard = ((height_above_floor > 0.10) & (height_above_floor < 2.1)) | (height_above_floor < -0.15)
        xi = np.searchsorted(self.x_edges, x, side="right") - 1
        zi = np.searchsorted(self.z_edges, z, side="right") - 1
        inside = (xi >= 0) & (xi < shape[1]) & (zi >= 0) & (zi < shape[0])
        index = zi[inside] * shape[1] + xi[inside]
        floor_count = np.bincount(index[floor[inside]], minlength=empty.size).reshape(shape)
        hazard_count = np.bincount(index[hazard[inside]], minlength=empty.size).reshape(shape)
        return LocalMap(self.x_edges, self.z_edges, floor_count >= 3, hazard_count >= 3, stamp)
