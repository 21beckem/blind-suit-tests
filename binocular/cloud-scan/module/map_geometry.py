"""Sparse stereo reprojection and explicitly named coordinate frames.

OpenCV camera coordinates are X right, Y down, Z forward. Gravity coordinates
are X right, Y up, Z forward with the observed floor at Y=0. This intuitive
gravity basis has determinant -1 relative to the OpenCV camera basis. Yaw is
positive toward camera-right; world X/Z use the same horizontal convention.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x_m: float = 0.0
    z_m: float = 0.0
    yaw_rad: float = 0.0
    tracking_ok: bool = True


def sample_points(result, stride, min_range_m, max_range_m, row_start=0,
                  min_confidence=2):
    """Return camera points and source rows/columns above a confidence threshold."""
    if stride < 1:
        raise ValueError("sample stride must be positive")
    mask = (result.confidence[row_start::stride, ::stride] >= min_confidence) & (
        result.disparity[row_start::stride, ::stride] > 0)
    rows, cols = np.nonzero(mask)
    rows = row_start + rows * stride
    cols = cols * stride
    if rows.size == 0:
        return np.empty((0, 3), np.float32), rows, cols
    disp = result.disparity[rows, cols].astype(np.float64)
    pixels = np.stack((cols, rows, disp, np.ones_like(disp)), axis=0)
    homogeneous = np.asarray(result.Q, np.float64) @ pixels
    w = homogeneous[3]
    valid = np.isfinite(homogeneous).all(axis=0) & (w > 1e-9)
    p_cam = np.empty((len(rows), 3), np.float64)
    p_cam[:] = np.nan
    p_cam[valid] = (homogeneous[:3, valid] / w[valid]).T
    valid &= (p_cam[:, 2] >= min_range_m) & (p_cam[:, 2] <= max_range_m)
    return p_cam[valid].astype(np.float32), rows[valid], cols[valid]


def gravity_rotation(normal_cam):
    """Map camera XYZ into right/up/forward gravity coordinates."""
    up_cam = np.asarray(normal_cam, np.float64)
    up_cam = up_cam / np.linalg.norm(up_cam)
    right_cam = np.array([1.0, 0.0, 0.0])
    right_cam -= up_cam * np.dot(right_cam, up_cam)
    right_cam /= np.linalg.norm(right_cam)
    forward_cam = -np.cross(right_cam, up_cam)
    return np.stack((right_cam, up_cam, forward_cam))


def camera_to_gravity(p_cam, rotation_grav_cam, camera_height_m):
    p_grav = np.asarray(p_cam) @ rotation_grav_cam.T
    p_grav[..., 1] += camera_height_m
    return p_grav


def gravity_to_world_xz(p_grav, pose):
    c, s = np.cos(pose.yaw_rad), np.sin(pose.yaw_rad)
    x_world = pose.x_m + c * p_grav[..., 0] + s * p_grav[..., 2]
    z_world = pose.z_m - s * p_grav[..., 0] + c * p_grav[..., 2]
    return x_world, z_world
