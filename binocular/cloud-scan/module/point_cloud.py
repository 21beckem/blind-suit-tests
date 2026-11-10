"""Build a point cloud from a DepthResult and optionally export binary PLY."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PointCloud:
    """Points in metres: X right, Y down, Z forward; grayscale uint8 intensity."""

    points: np.ndarray
    intensity: np.ndarray

    @classmethod
    def from_depth(cls, result, min_confidence=2):
        """Use solid pixels by default; pass 1 to also include weak matches."""
        if min_confidence not in (1, 2):
            raise ValueError("min_confidence must be 1 (weak + solid) or 2 (solid)")
        rows, cols = np.nonzero((result.disparity > 0) & (result.confidence >= min_confidence))
        disparity = result.disparity[rows, cols].astype(np.float64)
        homogeneous = result.Q @ np.stack((cols, rows, disparity, np.ones_like(disparity)))
        valid = np.isfinite(homogeneous).all(axis=0) & (homogeneous[3] > 1e-9)
        points = (homogeneous[:3, valid] / homogeneous[3, valid]).T.astype(np.float32)
        forward = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        return cls(points[forward], result.gray[rows, cols][valid][forward])

    def downsample(self, voxel_size):
        """Return a new cloud averaging points inside each voxel (metres)."""
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size must be positive and finite")
        if len(self.points) == 0:
            return PointCloud(self.points.copy(), self.intensity.copy())
        cells = np.floor(self.points / voxel_size).astype(np.int64)
        _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        points = np.stack([np.bincount(inverse, self.points[:, axis]) / counts
                           for axis in range(3)], axis=1).astype(np.float32)
        intensity = (np.bincount(inverse, self.intensity) / counts).astype(np.uint8)
        return PointCloud(points, intensity)

    def export(self, path):
        """Write binary little-endian PLY, including grayscale RGB per vertex."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        vertices = np.empty(len(self.points), dtype=dtype)
        for axis, name in enumerate(("x", "y", "z")):
            vertices[name] = self.points[:, axis]
        for name in ("red", "green", "blue"):
            vertices[name] = self.intensity
        with path.open("wb") as file:
            file.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n"
                        "property float x\nproperty float y\nproperty float z\n"
                        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
            vertices.tofile(file)
        return path
