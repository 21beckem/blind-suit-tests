"""Deterministic multi-elevation geometry and safety regressions."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module.config import load_config
from module.depth_mapper import DepthResult
from module.ground_plane import FloorEstimate, GroundPlaneEstimator
from module.map_geometry import Pose2D
from module.pose_tracker import PoseTracker
from module.terrain_map import TerrainMap


def stair_scene(treads=(0.17, 0.34, 0.51), landing=True,
                front_m=1.65, pitch_deg=20):
    """Render the closest ray intersection with treads and vertical risers."""
    width, height = 320, 240
    focal, baseline, camera_y = 369.81, 0.06046, 1.1
    cx, cy = 169.03, 113.64
    angle = math.radians(pitch_deg)
    rotation = np.array([[1, 0, 0], [0, -math.cos(angle), -math.sin(angle)],
                         [0, -math.sin(angle), math.cos(angle)]])
    v, u = np.indices((height, width), dtype=np.float64)
    ray_cam = np.stack(((u - cx) / focal, (v - cy) / focal,
                        np.ones_like(u)), axis=-1)
    ray = ray_cam @ rotation.T
    best = np.full((height, width), np.inf)
    bands = [(0.0, 0.0, front_m)]
    front = front_m
    for elevation in treads:
        bands.append((elevation, front, front + 0.30))
        front += 0.30
    if landing:
        bands.append((treads[-1], front, 3.5))
    for elevation, z0, z1 in bands:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (elevation - camera_y) / ray[..., 1]
        x, z = t * ray[..., 0], t * ray[..., 2]
        valid = ((t > 0) & (z >= z0) & (z < z1) &
                 (np.abs(x) < 0.9))
        best = np.where(valid & (t < best), t, best)
    prior = 0.0
    for index, elevation in enumerate(treads):
        edge_z = front_m + index * 0.30
        with np.errstate(divide="ignore", invalid="ignore"):
            t = edge_z / ray[..., 2]
        y = camera_y + t * ray[..., 1]
        x = t * ray[..., 0]
        valid = ((t > 0) & (y >= min(prior, elevation)) &
                 (y <= max(prior, elevation)) &
                 (np.abs(x) < 0.9))
        best = np.where(valid & (t < best), t, best)
        prior = elevation
    disparity = np.where(np.isfinite(best), focal * baseline / best, 0)
    disparity = disparity.astype(np.float32)
    confidence = np.where(disparity > 0, 2, 0).astype(np.uint8)
    q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy],
                  [0, 0, 0, focal], [0, 0, 1 / baseline, 0]], np.float64)
    gray = np.random.default_rng(42).integers(20, 235, (height, width),
                                              dtype=np.uint8)
    return DepthResult(disparity, confidence, gray, q, 48)


class TerrainMapTests(unittest.TestCase):
    def test_stairs_have_separate_surfaces_and_confirmed_links(self):
        config = load_config()
        result = stair_scene()
        estimator = GroundPlaneEstimator(config)
        grid = TerrainMap(config)
        for frame in range(12):
            t = frame / 25
            floor = estimator.fit(result, t)
            self.assertTrue(floor.fresh)
            grid.integrate(result, floor, Pose2D(y_m=1.1), t)
        self.assertGreaterEqual(grid.surface_count, 3)
        self.assertGreaterEqual(grid.confirmed_step_count, 1)
        self.assertGreater(np.count_nonzero(grid.state_grid() == grid.STEP_UP), 0)
        self.assertIsNotNone(grid.nearest_step())
        self.assertEqual(grid.surface_state(0, 1.5, 0), "free")

    def test_partial_staircase_does_not_invent_a_landing(self):
        config = load_config()
        result = stair_scene(landing=False)
        estimator = GroundPlaneEstimator(config)
        grid = TerrainMap(config)
        for frame in range(12):
            t = frame / 25
            grid.integrate(result, estimator.fit(result, t),
                           Pose2D(y_m=1.1), t)
        self.assertTrue(any(link["open_end"] for link in grid.transitions()))
        self.assertEqual(grid.surface_state(0, 3.2, 0.51), "unknown")

    def test_unverified_descending_tread_remains_a_hazard(self):
        config = load_config()
        result = stair_scene(treads=(-0.17, -0.34, -0.51),
                             landing=False, front_m=1.4, pitch_deg=25)
        estimator = GroundPlaneEstimator(config)
        grid = TerrainMap(config)
        for frame in range(12):
            t = frame / 25
            grid.integrate(result, estimator.fit(result, t),
                           Pose2D(y_m=1.1), t)
        self.assertGreaterEqual(grid.surface_count, 2)
        self.assertGreater(np.count_nonzero(grid._levels[1].grid.state_grid() ==
                                            grid.FREE), 0)
        self.assertEqual(grid.confirmed_step_count, 0)
        self.assertGreater(np.count_nonzero(grid.state_grid() == grid.DANGER), 0)
        self.assertIsNone(grid.nearest_step())

    def test_current_level_changes_only_after_motion_through_a_link(self):
        config = load_config()
        result = stair_scene()
        estimator = GroundPlaneEstimator(config)
        grid = TerrainMap(config)
        for frame in range(12):
            t = frame / 25
            grid.integrate(result, estimator.fit(result, t),
                           Pose2D(y_m=1.1), t)
        self.assertEqual(grid.current_level_height_m, 0)
        for frame in range(2):
            grid.pose = Pose2D(z_m=1.8, y_m=1.27)
            grid._now_s = 0.8 + frame / 25
            grid._select_current_level()
        self.assertAlmostEqual(grid.current_level_height_m, 0.17, delta=0.02)

    def test_height_layers_do_not_clear_each_other(self):
        config = load_config()
        grid = TerrainMap(config)
        lower = grid._new_level(0.0)
        upper = grid._new_level(1.0)
        grid._base_id = lower.id
        grid._active_id = lower.id
        for level in (lower, upper):
            level.grid.floor_hits[60, 60] = 3
            level.grid.last_floor_seen[60, 60] = 0
            level.grid._now_s = 0
        self.assertEqual(grid.surface_state(0, 0, 0), "free")
        self.assertEqual(grid.surface_state(0, 0, 1), "free")
        lower.grid.floor_hits[60, 60] = 0
        self.assertEqual(grid.surface_state(0, 0, 0), "unknown")
        self.assertEqual(grid.surface_state(0, 0, 1), "free")

    def test_reset_keeps_old_elevation_graph_without_inventing_live_floor(self):
        config = load_config()
        result = stair_scene()
        estimator = GroundPlaneEstimator(config)
        grid = TerrainMap(config)
        for frame in range(12):
            t = frame / 25
            grid.integrate(result, estimator.fit(result, t),
                           Pose2D(y_m=1.1), t)
        old_surfaces = grid.surface_count
        old_steps = grid.confirmed_step_count
        self.assertGreater(old_steps, 0)
        grid.reset_local(1)
        self.assertEqual(grid.surface_count, 0)
        grid.reset_local(0)
        self.assertEqual(grid.surface_count, old_surfaces)
        self.assertEqual(grid.confirmed_step_count, old_steps)
        self.assertEqual(grid.surface_state(0, 1.5, 0), "unknown")

    def test_single_box_does_not_create_a_step_link(self):
        from tests.test_obstacle_map import synthetic_scene

        config = load_config()
        result, _, _ = synthetic_scene(1.5)
        estimator = GroundPlaneEstimator(config)
        grid = TerrainMap(config)
        for frame in range(6):
            t = frame / 25
            grid.integrate(result, estimator.fit(result, t),
                           Pose2D(y_m=1.1), t)
        self.assertEqual(grid.confirmed_step_count, 0)
        self.assertIsNone(grid.nearest_step())


class VerticalPoseTests(unittest.TestCase):
    def test_pnp_camera_rise_updates_world_y(self):
        config = load_config()
        tracker = PoseTracker(config)
        result = stair_scene()
        normal = np.array([0., -1., 0.])
        floor = FloorEstimate(normal, 1.1, np.diag([1., -1., 1.]),
                              True, True, 0.0, 300, 300)
        source_pose = Pose2D(y_m=1.1)
        source = np.array([(x, y) for y in (20, 100, 180, 220)
                           for x in (20, 65, 110, 155, 200, 245, 300)],
                          np.float32)
        points = np.ones((len(source), 3), np.float32)
        inliers = np.arange(len(source), dtype=np.int32).reshape(-1, 1)
        with patch("module.pose_tracker.cv2.solvePnPRansac",
                   return_value=(True, np.zeros((3, 1)),
                                 np.array([[0.], [0.17], [0.]]), inliers)):
            solved = tracker._solve((points, source, source), result, floor,
                                    0.04, floor, source_pose,
                                    check_motion=False)
        self.assertIsNotNone(solved)
        self.assertAlmostEqual(solved[0].y_m, 1.27, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
