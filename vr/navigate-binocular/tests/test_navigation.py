"""Synthetic checks for the binary HUD and sensor-independent route policy."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grid import generate_grid
from navigation import LocalMap, Navigator, fuse_maps
from stereo_obstacles import StereoObstacles


class NavigationTests(unittest.TestCase):
    def scene(self, *, observed=True):
        x_edges = np.arange(-1.8, 1.81, 0.12)
        z_edges = np.arange(0.2, 3.01, 0.12)
        shape = (len(z_edges) - 1, len(x_edges) - 1)
        return LocalMap(x_edges, z_edges, np.full(shape, observed), np.zeros(shape, bool), 0.0)

    def block(self, scene, x, z):
        xi = np.searchsorted(scene.x_edges, x, side="right") - 1
        zi = np.searchsorted(scene.z_edges, z, side="right") - 1
        scene.blocked[zi, xi] = True

    def test_clear_path_and_binary_code(self):
        decision = Navigator().decide(self.scene())
        grid = generate_grid(decision)
        self.assertEqual(decision.direction, 2)
        self.assertEqual(grid.shape, (4, 5))
        self.assertEqual(grid[0].tolist(), [0, 0, 1, 0, 0])
        self.assertTrue(np.isin(grid, [0, 1]).all())

    def test_imminent_centre_obstacle_stops(self):
        scene = self.scene()
        self.block(scene, 0, 1.0)
        decision = Navigator().decide(scene)
        self.assertEqual(decision.action, "stop")
        self.assertEqual(generate_grid(decision)[0].tolist(), [1] * 5)

    def test_ahead_obstacle_selects_side(self):
        scene = self.scene()
        self.block(scene, 0, 2.6)
        decision = Navigator().decide(scene)
        self.assertEqual(decision.action, "direction")
        self.assertNotEqual(decision.direction, 2)
        self.assertTrue(decision.ahead[2])

    def test_missing_depth_is_unknown(self):
        decision = Navigator().decide(self.scene(observed=False))
        self.assertEqual(decision.action, "unknown")
        self.assertEqual(generate_grid(decision)[0].tolist(), [1, 0, 1, 0, 1])

    def test_corridor_width_rejects_narrow_gap(self):
        scene = self.scene()
        self.block(scene, 0.25, 2.0)
        self.assertTrue(Navigator().decide(scene).ahead[2])

    def test_fusion_keeps_hazard_and_drops_stale_clear(self):
        clear = self.scene()
        clear.timestamp = 8.0
        blocked = self.scene(observed=False)
        blocked.timestamp = 10.0
        self.block(blocked, 0, 1.0)
        merged = fuse_maps([clear, blocked], now=10.1)
        self.assertFalse(merged.observed.any())
        self.assertTrue(merged.blocked.any())
        self.assertEqual(Navigator().decide(merged).action, "stop")

    def test_stereo_floor_and_object(self):
        width, height, focal = 320, 240, 350.0
        cy, cx = 120.0, 160.0
        pitch, camera_height = np.deg2rad(32), 1.1
        rows = np.arange(height)[:, None]
        denom = np.cos(pitch) * (rows - cy) / focal + np.sin(pitch)
        floor_z = np.where(denom > 0, camera_height / denom, 0)
        depth = np.broadcast_to(floor_z, (height, width)).copy().astype(np.float32)
        depth[(depth < 0.2) | (depth > 3.0)] = 0
        observer = StereoObstacles(fx=focal, fy=focal, cx=cx, cy=cy,
                                   width=width, height=height)
        clear = observer.observe(depth)
        self.assertGreater(np.count_nonzero(clear.observed), 10)
        self.assertEqual(np.count_nonzero(clear.blocked), 0)
        depth[80:160, 145:175] = 1.6
        with_box = observer.observe(depth)
        self.assertGreater(np.count_nonzero(with_box.blocked), 0)


if __name__ == "__main__":
    unittest.main()
