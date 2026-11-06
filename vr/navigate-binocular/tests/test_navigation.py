"""Synthetic checks for the binary HUD and sensor-independent route policy."""

import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grid import generate_grid
from config import V_ON, V_OFF, ACTION_ROW_STOP, ACTION_ROW_UNKNOWN
from navigation import LocalMap, Navigator, fuse_maps
from stereo_obstacles import StereoObstacles
from debug_view import render_dashboard
from tuning_controls import NavigationControls, TuningValues, load_settings, save_settings, settings_from_dict


class NavigationTests(unittest.TestCase):
    def scene(self, *, observed=True, far=3.0):
        x_edges = np.arange(-1.8, 1.81, 0.12)
        z_edges = np.arange(0.2, far + 0.01, 0.12)
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
        self.assertEqual(grid[0].tolist(), [V_OFF, V_OFF, V_ON, V_OFF, V_OFF])
        self.assertTrue(np.isin(grid, [V_OFF, V_ON]).all())

    def test_every_hud_cell_uses_configured_brightness(self):
        scene = self.scene()
        self.block(scene, 0, 1.0)
        navigator = Navigator()
        navigator.decide(scene, now=0)
        decision = navigator.decide(scene, now=0.06)
        with patch("grid.V_ON", 0.3), patch("grid.V_OFF", 0.1), \
             patch("grid.ACTION_ROW_STOP", [0.3] * 5):
            grid = generate_grid(decision)
        self.assertTrue(np.isin(grid, [np.float32(0.3), np.float32(0.1)]).all())
        self.assertTrue(np.allclose(grid[0], 0.3))
        self.assertTrue(np.isclose(grid[2, 2], 0.1))

    def test_imminent_centre_obstacle_stops(self):
        scene = self.scene()
        self.block(scene, 0, 1.0)
        navigator = Navigator()
        self.assertEqual(navigator.decide(scene, now=0).action, "unknown")
        decision = navigator.decide(scene, now=0.06)
        self.assertEqual(decision.action, "stop")
        self.assertEqual(generate_grid(decision)[0].tolist(), ACTION_ROW_STOP)

    def test_stop_distance_is_separate_from_near_band(self):
        scene = self.scene()
        self.block(scene, 0, 1.7)
        self.assertNotEqual(Navigator(stop_distance=1.2).decide(scene).action, "stop")
        navigator = Navigator(stop_distance=1.8, near_limit=2.0)
        navigator.decide(scene, now=0)
        decision = navigator.decide(scene, now=0.06)
        self.assertEqual(decision.action, "stop")
        self.assertEqual(decision.reason, "hazard within stop distance")

    def test_ahead_limit_excludes_more_distant_hazards(self):
        scene = self.scene(far=4.0)
        self.block(scene, 0, 3.4)
        self.assertEqual(Navigator(ahead_limit=3.0).decide(scene).direction, 2)
        self.assertTrue(Navigator(ahead_limit=4.0).decide(scene).ahead[2])

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
        self.assertEqual(generate_grid(decision)[0].tolist(), ACTION_ROW_UNKNOWN)

    def test_action_rows_come_from_config(self):
        unknown = Navigator().decide(self.scene(observed=False))
        stop_scene = self.scene()
        self.block(stop_scene, 0, 1.0)
        stop_navigator = Navigator()
        stop_navigator.decide(stop_scene, now=0)
        stop = stop_navigator.decide(stop_scene, now=0.06)
        with patch("grid.ACTION_ROW_UNKNOWN", [V_OFF, V_ON, V_OFF, V_ON, V_OFF]), \
             patch("grid.ACTION_ROW_STOP", [V_ON, V_OFF, V_ON, V_OFF, V_ON]):
            self.assertEqual(generate_grid(unknown)[0].tolist(),
                             [V_OFF, V_ON, V_OFF, V_ON, V_OFF])
            self.assertEqual(generate_grid(stop)[0].tolist(),
                             [V_ON, V_OFF, V_ON, V_OFF, V_ON])

    def test_obstacle_and_path_changes_use_elapsed_milliseconds(self):
        scene = self.scene()
        navigator = Navigator(debounce_add_obstacle_ms=100,
                              debounce_remove_obstacle_ms=400,
                              debounce_add_path_ms=250,
                              debounce_path_choice_ms=350)
        self.assertEqual(navigator.decide(scene, now=0).direction, 2)
        self.block(scene, 0, 2.6)
        self.assertFalse(navigator.decide(scene, now=0.01).ahead[2])
        self.assertFalse(navigator.decide(scene, now=0.09).ahead[2])
        blocked = navigator.decide(scene, now=0.12)
        self.assertTrue(blocked.ahead[2])
        self.assertNotEqual(blocked.direction, 2)
        side = blocked.direction
        scene.blocked[:] = False
        self.assertTrue(navigator.decide(scene, now=0.2).ahead[2])
        self.assertTrue(navigator.decide(scene, now=0.59).ahead[2])
        self.assertFalse(navigator.decide(scene, now=0.61).ahead[2])
        self.assertEqual(navigator.decide(scene, now=0.87).direction, side)
        self.assertEqual(navigator.decide(scene, now=1.21).direction, side)
        self.assertEqual(navigator.decide(scene, now=1.23).direction, 2)

    def test_one_brief_obstacle_does_not_change_hud(self):
        scene = self.scene()
        navigator = Navigator(debounce_add_obstacle_ms=100)
        navigator.decide(scene, now=0)
        self.block(scene, 0, 2.6)
        self.assertFalse(navigator.decide(scene, now=0.01).ahead[2])
        scene.blocked[:] = False
        self.assertFalse(navigator.decide(scene, now=0.05).ahead[2])
        self.block(scene, 0, 2.6)
        self.assertFalse(navigator.decide(scene, now=0.09).ahead[2])
        self.assertFalse(navigator.decide(scene, now=0.17).ahead[2])
        self.assertTrue(navigator.decide(scene, now=0.20).ahead[2])

    def test_add_stop_waits_for_persistent_center_hazard(self):
        scene = self.scene()
        navigator = Navigator(debounce_add_stop_ms=50)
        self.assertEqual(navigator.decide(scene, now=0).direction, 2)
        self.block(scene, 0, 0.4)
        self.assertEqual(navigator.decide(scene, now=0.01).action, "unknown")
        scene.blocked[:] = False
        self.assertEqual(navigator.decide(scene, now=0.03).action, "direction")
        self.block(scene, 0, 0.4)
        self.assertEqual(navigator.decide(scene, now=0.10).action, "unknown")
        self.assertEqual(navigator.decide(scene, now=0.14).action, "unknown")
        self.assertEqual(navigator.decide(scene, now=0.16).action, "stop")

    def test_add_stop_also_covers_all_blocked_corridors(self):
        scene = self.scene()
        navigator = Navigator(debounce_add_obstacle_ms=0, debounce_add_stop_ms=50)
        navigator.decide(scene, now=0)
        zi = np.searchsorted(scene.z_edges, 2.6, side="right") - 1
        scene.blocked[zi, :] = True
        self.assertEqual(navigator.decide(scene, now=0.01).action, "unknown")
        self.assertEqual(navigator.decide(scene, now=0.07).action, "stop")

    def test_unknown_and_stop_release_are_debounced(self):
        scene = self.scene()
        navigator = Navigator(debounce_add_unknown_ms=0,
                              debounce_remove_unknown_ms=300,
                              debounce_add_path_ms=250,
                              debounce_release_stop_ms=400)
        self.assertEqual(navigator.decide(scene, now=0).direction, 2)
        scene.observed[:] = False
        self.assertEqual(navigator.decide(scene, now=0.1).action, "unknown")
        scene.observed[:] = True
        self.assertEqual(navigator.decide(scene, now=0.2).action, "unknown")
        self.assertEqual(navigator.decide(scene, now=0.49).action, "unknown")
        self.assertEqual(navigator.decide(scene, now=0.51).action, "unknown")
        self.assertEqual(navigator.decide(scene, now=0.77).action, "direction")
        self.block(scene, 0, 0.4)
        self.assertEqual(navigator.decide(scene, now=0.78).action, "unknown")
        self.assertEqual(navigator.decide(scene, now=0.84).action, "stop")
        scene.blocked[:] = False
        self.assertEqual(navigator.decide(scene, now=0.85).reason,
                         "waiting for stop condition to clear")
        self.assertEqual(navigator.decide(scene, now=1.24).action, "stop")
        self.assertEqual(navigator.decide(scene, now=1.26).action, "direction")

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
        navigator = Navigator()
        self.assertEqual(navigator.decide(merged, now=0).action, "unknown")
        self.assertEqual(navigator.decide(merged, now=0.06).action, "stop")

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
        self.assertTrue(observer.debug["floor_found"])
        depth[80:160, 145:175] = 1.6
        with_box = observer.observe(depth)
        self.assertGreater(np.count_nonzero(with_box.blocked), 0)
        self.assertGreater(observer.debug["hazard_cells"], 0)

    def test_dashboard_explains_missing_floor(self):
        depth = np.zeros((120, 160), np.float32)
        observer = StereoObstacles(fx=200, fy=200, cx=80, cy=60,
                                   width=160, height=120)
        scene = observer.observe(depth)
        decision = Navigator().decide(scene)
        stats = dict(camera_fps=0, frame_age_ms=0, stereo_ms=0,
                     observe_ms=0, total_ms=0, message_gap_ms=0)
        image = render_dashboard(np.zeros_like(depth, np.uint8), depth,
                                 scene, decision, generate_grid(decision),
                                 observer.debug, stats, near=0.2, far=4.0,
                                 settings=TuningValues())
        self.assertEqual(image.shape, (720, 1200, 3))
        self.assertFalse(observer.debug["floor_found"])
        self.assertEqual(observer.debug["floor_status"], "no valid depth")
        self.assertEqual(decision.reason, "no verified clear corridor")

    def test_tuning_values_save_and_validate(self):
        TuningValues(stop_distance=0.4, near_distance=0.6, ahead_distance=0.8)
        values = TuningValues(stop_distance=1.4, near_distance=2.0,
                              ahead_distance=3.5, hazard_points=7)
        with tempfile.NamedTemporaryFile(dir=Path(__file__).resolve().parents[1],
                                         suffix=".json", delete=False) as temporary:
            path = Path(temporary.name)
        try:
            save_settings(values, path)
            self.assertEqual(load_settings(path), values)
        finally:
            path.unlink(missing_ok=True)
        with self.assertRaises(ValueError):
            TuningValues(stop_distance=2.0, near_distance=1.5)
        self.assertEqual(settings_from_dict({"switch_frames": 10}), TuningValues())
        with self.assertRaises(ValueError):
            TuningValues(debounce_add_obstacle_ms=-1)

    def test_controls_keep_distance_bands_ordered(self):
        positions = {
            "Stop distance cm": 220, "Near distance cm": 180,
            "Ahead distance cm": 200, "Path width cm": 65,
            "Min coverage percent": 12, "Obstacle points per cell": 3,
            "Add obstacle ms": 100, "Remove obstacle ms": 400,
            "Add unknown ms": 0, "Remove unknown ms": 300,
            "Add path ms": 250, "Path choice ms": 350, "Add stop ms": 50,
            "Release stop ms": 400,
        }
        def set_position(label, _window, position):
            positions[label] = position
        controls = NavigationControls.__new__(NavigationControls)
        with patch("tuning_controls.cv2.getTrackbarPos", side_effect=lambda label, _window: positions[label]), \
             patch("tuning_controls.cv2.setTrackbarPos", side_effect=set_position):
            values = controls.read()
        self.assertEqual((values.stop_distance, values.near_distance, values.ahead_distance),
                         (2.2, 2.4, 2.6))
        self.assertEqual(positions["Near distance cm"], 240)
        self.assertEqual(positions["Ahead distance cm"], 260)


if __name__ == "__main__":
    unittest.main()
