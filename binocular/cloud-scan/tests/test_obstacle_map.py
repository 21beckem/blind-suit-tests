"""Deterministic metric geometry and rolling-map fixtures."""

import math
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from threading import Event
from unittest.mock import patch
import unittest

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from module.config import load_config
from module.depth_mapper import DepthResult
from module.ground_plane import FloorEstimate, GroundPlaneEstimator
from module.map_geometry import camera_to_gravity, gravity_rotation, sample_points
from module.map_geometry import Pose2D
from module.obstacle_map import ObstacleMap
from module.pose_tracker import PoseTracker
from module.latest_camera_feed import LatestCameraFeed
from map_offline import discover_recordings, run_recording
import map_live


def synthetic_scene(box_z=None, pitch_deg=20.0, box_x=0.25, dropoff_m=0.0):
    """Render exact rectified depth of a floor and optional 0.4 m high box face."""
    width, height = 320, 240
    focal, baseline, camera_height = 369.81, 0.06046, 1.1
    cx, cy = 169.03, 113.64
    q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy],
                  [0, 0, 0, focal], [0, 0, 1 / baseline, 0]], np.float64)
    angle = math.radians(pitch_deg)
    rotation = np.array([[1, 0, 0], [0, -math.cos(angle), -math.sin(angle)],
                         [0, -math.sin(angle), math.cos(angle)]])
    v, u = np.indices((height, width), dtype=np.float64)
    ray_cam = np.stack(((u - cx) / focal, (v - cy) / focal, np.ones_like(u)), axis=-1)
    ray_grav = ray_cam @ rotation.T
    floor_t = np.where(ray_grav[..., 1] < -1e-8,
                       -camera_height / np.minimum(ray_grav[..., 1], -1e-8), np.inf)
    t = floor_t.copy()
    if dropoff_m:
        lower_t = -(camera_height + dropoff_m) / np.minimum(ray_grav[..., 1], -1e-8)
        lower_x = lower_t * ray_grav[..., 0]
        lower_z = lower_t * ray_grav[..., 2]
        patch = ((lower_x >= 0.3) & (lower_x <= 0.7) &
                 (lower_z >= 1.2) & (lower_z <= 2.0) &
                 np.isfinite(floor_t))
        t[patch] = lower_t[patch]
    if box_z is not None:
        box_t = box_z / np.maximum(ray_grav[..., 2], 1e-8)
        box_x_hit = box_t * ray_grav[..., 0]
        box_y_hit = camera_height + box_t * ray_grav[..., 1]
        face = ((np.abs(box_x_hit - box_x) <= 0.18) &
                (box_y_hit >= 0) & (box_y_hit <= 0.4) &
                (box_t > 0) & (box_t < floor_t))
        t[face] = box_t[face]
    disparity = np.where(np.isfinite(t) & (t > 0), focal * baseline / t, 0).astype(np.float32)
    confidence = np.where(disparity > 0, 2, 0).astype(np.uint8)
    gray = np.random.default_rng(42).integers(20, 235, (height, width), dtype=np.uint8)
    return DepthResult(disparity, confidence, gray, q, 48), rotation, camera_height


def panorama_frame(yaw_deg):
    """Deterministic textured 3D cylinder for a continuous 180 degree turn."""
    rng = np.random.default_rng(77)
    count = 7000
    angles = rng.uniform(-math.pi, math.pi, count)
    radii = rng.uniform(1.5, 2.5, count)
    world = np.column_stack((radii * np.sin(angles),
                             rng.uniform(-0.7, 0.7, count),
                             radii * np.cos(angles)))
    yaw = math.radians(yaw_deg)
    rotation = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                         [0, 1, 0], [math.sin(yaw), 0, math.cos(yaw)]])
    p_cam = world @ rotation.T
    focal, baseline, cx, cy = 369.81, 0.06046, 169.03, 113.64
    positive = p_cam[:, 2] > 0.6
    u = np.rint(focal * p_cam[:, 0] / np.maximum(p_cam[:, 2], 0.6) + cx).astype(int)
    v = np.rint(focal * p_cam[:, 1] / np.maximum(p_cam[:, 2], 0.6) + cy).astype(int)
    visible = np.nonzero(positive & (u >= 3) & (u < 317) & (v >= 3) & (v < 237))[0]
    gray = np.full((240, 320), 20, np.uint8)
    disparity = np.zeros((240, 320), np.float32)
    confidence = np.zeros((240, 320), np.uint8)
    shades = rng.integers(90, 245, count)
    for index in visible:
        x, y = u[index], v[index]
        cv2.circle(gray, (x, y), 2, int(shades[index]), -1)
        cv2.circle(disparity, (x, y), 2, float(focal * baseline / p_cam[index, 2]), -1)
        cv2.circle(confidence, (x, y), 2, 2, -1)
    q = np.array([[1, 0, 0, -cx], [0, 1, 0, -cy],
                  [0, 0, 0, focal], [0, 0, 1 / baseline, 0]], np.float64)
    return DepthResult(disparity, confidence, gray, q, 48)


class GroundGeometryTests(unittest.TestCase):
    def test_pitched_floor_height_and_gravity_frame(self):
        result, expected_rotation, camera_height = synthetic_scene(1.5)
        floor = GroundPlaneEstimator(load_config()).fit(result, 0.0)
        self.assertTrue(floor.fresh)
        self.assertTrue(floor.usable)
        self.assertAlmostEqual(floor.height_m, camera_height, delta=0.05)
        np.testing.assert_allclose(floor.rotation_grav_cam, expected_rotation, atol=0.05)
        points, _, _ = sample_points(result, 3, 0.5, 3.0)
        grav = camera_to_gravity(points, floor.rotation_grav_cam, floor.height_m)
        self.assertTrue(np.isfinite(grav).all())
        self.assertGreater(np.count_nonzero(np.abs(grav[:, 1]) < 0.05), 100)

    def test_upright_axis_signs(self):
        rotation = gravity_rotation(np.array([0., -1., 0.]))
        np.testing.assert_array_equal(rotation, np.diag([1., -1., 1.]))

    def test_stale_floor_expires(self):
        config = load_config()
        estimator = GroundPlaneEstimator(config)
        result, _, _ = synthetic_scene()
        self.assertTrue(estimator.fit(result, 0.0).fresh)
        result.confidence[:] = 0
        near = estimator.fit(result, 0.5)
        late = estimator.fit(result, 1.1)
        self.assertFalse(near.fresh)
        self.assertTrue(near.usable)
        self.assertFalse(late.usable)

    def test_vertical_wall_is_not_a_floor(self):
        result, _, _ = synthetic_scene()
        result.disparity[:] = result.Q[2, 3] / result.Q[3, 2] / 1.5
        result.confidence[:] = 2
        floor = GroundPlaneEstimator(load_config()).fit(result, 0)
        self.assertFalse(floor.usable)


class SingleFrameMapTests(unittest.TestCase):
    def test_small_pitch_change_keeps_box_edge_within_one_cell(self):
        config = load_config()
        bounds = []
        for pitch in (20.0, 20.5):
            result, _, _ = synthetic_scene(1.5, pitch_deg=pitch)
            floor = GroundPlaneEstimator(config).fit(result, 0)
            grid = ObstacleMap(config)
            grid.integrate(result, floor, Pose2D(), 0)
            rows, cols = np.nonzero(np.isfinite(grid.max_height))
            self.assertGreater(len(rows), 0)
            bounds.append(np.array([cols.min(), cols.max(), rows.min(), rows.max()]))
        self.assertTrue(np.all(np.abs(bounds[0] - bounds[1]) <= 1))

    def test_pitched_boxes_land_in_expected_cells(self):
        config = load_config()
        for distance in (1.0, 1.5, 2.0, 2.5):
            with self.subTest(distance=distance):
                result, _, _ = synthetic_scene(distance)
                floor_estimator = GroundPlaneEstimator(config)
                grid = ObstacleMap(config)
                for frame in range(3):
                    t = frame / 25
                    floor = floor_estimator.fit(result, t)
                    stats = grid.integrate(result, floor, Pose2D(), t)
                self.assertGreater(stats["obstacle_cells"], 0)
                row, col = np.nonzero(grid.state_grid() == grid.OBSTACLE)
                self.assertGreater(len(col), 0)
                x = grid.origin_x_m + (col + 0.5) * grid.cell_m
                z = grid.origin_z_m + (row + 0.5) * grid.cell_m
                near = np.argmin(np.abs(x - 0.25) + np.abs(z - distance))
                tolerance = 0.1 if distance <= 2 else 0.2
                self.assertAlmostEqual(x[near], 0.25, delta=tolerance)
                self.assertAlmostEqual(z[near], distance, delta=tolerance)

    def test_flat_floor_and_weak_depth_do_not_make_obstacles_or_free_space(self):
        config = load_config()
        result, _, _ = synthetic_scene()
        estimator = GroundPlaneEstimator(config)
        grid = ObstacleMap(config)
        for frame in range(5):
            t = frame / 25
            grid.integrate(result, estimator.fit(result, t), Pose2D(), t)
        self.assertEqual(np.count_nonzero(grid.state_grid() == grid.OBSTACLE), 0)
        self.assertGreater(np.count_nonzero(grid.state_grid() == grid.FREE), 0)
        weak = synthetic_scene()[0]
        weak.confidence[weak.confidence == 2] = 1
        empty = ObstacleMap(config)
        floor = GroundPlaneEstimator(config).fit(weak, 0)
        empty.integrate(weak, floor, Pose2D(), 0)
        self.assertTrue(np.all(empty.state_grid() == empty.UNKNOWN))


class TemporalMapTests(unittest.TestCase):
    def test_obstacle_survives_recent_turn_then_fades_without_new_hits(self):
        grid = ObstacleMap(load_config())
        grid.hits[60, 60] = grid.settings["obstacle_evidence_cap"]
        grid.min_height[60, 60] = 0.2
        grid.max_height[60, 60] = 0.8
        grid.last_seen[60, 60] = 0
        grid._decay(0)
        grid._decay(3)
        self.assertEqual(grid.state_grid()[60, 60], grid.OBSTACLE)
        grid._decay(6)
        self.assertEqual(grid.state_grid()[60, 60], grid.UNKNOWN)

    def test_forward_coverage_counts_unknown_without_inferred_free(self):
        grid = ObstacleMap(load_config())
        grid.pose = Pose2D()
        coverage = grid.forward_coverage(1.0, 1.0)
        self.assertGreater(coverage["cells"], 0)
        self.assertEqual(coverage["floor_evidence_cells"], 0)
        self.assertEqual(coverage["unknown"], coverage["cells"])

    def test_traversed_footprint_clears_to_unknown_without_inventing_floor(self):
        config = load_config()
        grid = ObstacleMap(config)
        row, col = 60, 60
        grid.hits[row, col] = 6
        grid.danger_hits[row, col] = 5
        grid.last_seen[row, col] = 0
        grid.last_danger_seen[row, col] = 0
        grid.min_height[row, col] = 0.1
        grid.max_height[row, col] = 0.7
        result, _, _ = synthetic_scene()
        result.confidence[:] = 0
        normal = np.array([0., -1., 0.])
        floor = FloorEstimate(normal, 1.1, gravity_rotation(normal),
                              False, True, 0, 0, 0)
        grid.integrate(result, floor, Pose2D(), 0)
        self.assertEqual(grid.state_grid()[row, col], grid.UNKNOWN)
        self.assertEqual(grid.floor_hits[row, col], 0)

    def test_low_flat_depth_artifact_stays_unknown(self):
        grid = ObstacleMap(load_config())
        row, col = 60, 60
        grid.hits[row, col] = 6
        grid.last_seen[row, col] = 0
        grid.min_height[row, col] = 0.10
        grid.max_height[row, col] = 0.16
        self.assertEqual(grid.state_grid()[row, col], grid.UNKNOWN)
        grid.max_height[row, col] = 0.40
        self.assertEqual(grid.state_grid()[row, col], grid.OBSTACLE)
        grid.min_height[row, col] = grid.max_height[row, col] = 0.70
        self.assertEqual(grid.state_grid()[row, col], grid.OBSTACLE)

    def test_unobserved_dropoff_warning_fades_to_unknown(self):
        grid = ObstacleMap(load_config())
        grid.danger_hits[60, 60] = 5
        grid.last_danger_seen[60, 60] = 0
        grid._now_s = 0
        self.assertEqual(grid.state_grid()[60, 60], grid.DANGER)
        grid._decay(3.0)
        self.assertEqual(grid.state_grid()[60, 60], grid.UNKNOWN)

    def test_floor_ray_clears_crossed_wall_but_not_low_box(self):
        grid = ObstacleMap(load_config())
        grid.hits[:] = 6
        grid.min_height[:] = 0.1
        grid.max_height[:] = 0.3
        low_x, low_z = int((0.125 + 3) / 0.05), int((1.0 + 3) / 0.05)
        tall_x, tall_z = int((0.1875 + 3) / 0.05), int((1.5 + 3) / 0.05)
        grid.max_height[tall_z - 1:tall_z + 2, tall_x - 1:tall_x + 2] = 1.2
        carved = grid._carve_obstacle_rays(
            np.array([[0.25, 0., 2.]], np.float32),
            np.array([True]), np.array([160]), 320, Pose2D(), 1.1,
            np.empty(0, np.int64))
        self.assertGreater(carved, 0)
        self.assertEqual(grid.hits[low_z, low_x], 6)
        self.assertLess(np.min(grid.hits[tall_z - 1:tall_z + 2,
                                          tall_x - 1:tall_x + 2]), 6)

    def test_repeated_solid_floor_replaces_obsolete_obstacle(self):
        config = load_config()
        box, _, _ = synthetic_scene(1.5)
        bare, _, _ = synthetic_scene()
        estimator = GroundPlaneEstimator(config)
        grid = ObstacleMap(config)
        for frame in range(3):
            t = frame / 25
            grid.integrate(box, estimator.fit(box, t), Pose2D(), t)
        self.assertEqual(grid.cell_state(0.25, 1.5), "obstacle")
        for frame in range(3, 10):
            t = frame / 25
            grid.integrate(bare, estimator.fit(bare, t), Pose2D(), t)
        self.assertEqual(grid.cell_state(0.25, 1.5), "free")

    def test_dropoff_is_unknown_danger_not_free(self):
        config = load_config()
        result, _, _ = synthetic_scene(dropoff_m=0.35)
        estimator = GroundPlaneEstimator(config)
        grid = ObstacleMap(config)
        for frame in range(4):
            t = frame / 25
            grid.integrate(result, estimator.fit(result, t), Pose2D(), t)
        self.assertGreater(np.count_nonzero(grid.state_grid() == grid.DANGER), 0)
        self.assertTrue(np.all(grid.state_grid()[grid.danger_hits >= 2.5] != grid.FREE))

    def test_three_of_five_then_decay_to_unknown(self):
        config = load_config()
        box, _, _ = synthetic_scene(1.5)
        bare, _, _ = synthetic_scene()
        estimator = GroundPlaneEstimator(config)
        grid = ObstacleMap(config)
        for frame, result in enumerate((box, bare, box, bare, box)):
            t = frame / 25
            grid.integrate(result, estimator.fit(result, t), Pose2D(), t)
            if frame < 4:
                self.assertEqual(np.count_nonzero(grid.state_grid() == grid.OBSTACLE), 0)
        self.assertGreater(np.count_nonzero(grid.state_grid() == grid.OBSTACLE), 0)
        self.assertIsNotNone(grid.nearest_obstacle(0, 60))
        self.assertEqual(grid.cell_state(0.25, 1.5), "obstacle")
        self.assertEqual(grid.clearance_sectors()["front"] is None, False)
        grid.integrate(bare, estimator.fit(bare, 21.0), Pose2D(tracking_ok=False), 21.0)
        self.assertFalse(grid.tracking_ok)
        self.assertEqual(np.count_nonzero(grid.state_grid() == grid.OBSTACLE), 0)

    def test_one_frame_noise_never_confirms(self):
        config = load_config()
        box, _, _ = synthetic_scene(1.5)
        bare, _, _ = synthetic_scene()
        estimator = GroundPlaneEstimator(config)
        grid = ObstacleMap(config)
        for frame, result in enumerate((box, bare, bare, bare, bare)):
            t = frame / 25
            grid.integrate(result, estimator.fit(result, t), Pose2D(), t)
        self.assertEqual(np.count_nonzero(grid.state_grid() == grid.OBSTACLE), 0)

    def test_scroll_preserves_old_cells_and_clears_exposed_area(self):
        config = load_config()
        box, _, _ = synthetic_scene(1.5)
        estimator = GroundPlaneEstimator(config)
        grid = ObstacleMap(config)
        for frame in range(3):
            t = frame / 25
            grid.integrate(box, estimator.fit(box, t), Pose2D(), t)
        self.assertEqual(grid.cell_state(0.25, 1.5), "obstacle")
        old_origin = grid.origin_x_m
        grid.integrate(box, estimator.fit(box, 0.2), Pose2D(x_m=2.0), 0.2)
        self.assertGreater(grid.origin_x_m, old_origin)
        self.assertEqual(grid.cell_state(0.25, 1.5), "obstacle")
        self.assertEqual(grid.cell_state(4.0, 0), "unknown")


class PoseTrackerTests(unittest.TestCase):
    def test_long_tracking_loss_requests_new_local_map(self):
        config = load_config()
        level_normal = np.array([0., -1., 0.])
        floor = FloorEstimate(level_normal, 1.1, gravity_rotation(level_normal),
                              True, True, 0, 500, 500)
        tracker = PoseTracker(config)
        textured = panorama_frame(0)
        self.assertTrue(tracker.update(textured, floor, 0).tracking_ok)
        blank = DepthResult(textured.disparity, textured.confidence,
                            np.zeros_like(textured.gray), textured.Q,
                            textured.max_disparity)
        self.assertFalse(tracker.update(blank, floor, 0.04).tracking_ok)
        self.assertFalse(tracker.update(blank, floor, 2.1).tracking_ok)
        recovered = tracker.update(textured, floor, 2.14)
        self.assertTrue(recovered.tracking_ok)
        self.assertTrue(tracker.reset_map)

    def test_full_turn_retains_obstacle_behind_camera(self):
        config = load_config()
        grid = ObstacleMap(config)
        box, _, _ = synthetic_scene(1.5)
        estimator = GroundPlaneEstimator(config)
        for frame in range(3):
            t = frame / 25
            grid.integrate(box, estimator.fit(box, t), Pose2D(), t)
        self.assertEqual(grid.cell_state(0.25, 1.5), "obstacle")

        level_normal = np.array([0., -1., 0.])
        level_floor = FloorEstimate(level_normal, 1.1, gravity_rotation(level_normal),
                                    True, True, 0, 500, 500)
        tracker = PoseTracker(config)
        first = tracker.update(panorama_frame(0), level_floor, 0.12)
        self.assertTrue(first.tracking_ok)
        for degrees in range(3, 181, 3):
            pose = tracker.update(panorama_frame(degrees), level_floor,
                                  0.12 + degrees / 75)
            self.assertTrue(pose.tracking_ok, f"tracking lost at {degrees} degrees")
        self.assertAlmostEqual(math.degrees(pose.yaw_rad), 180, delta=10)
        self.assertLess(math.hypot(pose.x_m, pose.z_m), 0.3)
        grid.integrate(panorama_frame(180), level_floor, pose, 2.52)
        self.assertEqual(grid.cell_state(0.25, 1.5), "obstacle")
        self.assertIsNotNone(grid.nearest_obstacle(180, 40))

    def test_known_translation_and_turn_keep_previous_obstacle(self):
        config = load_config()
        result, _, _ = synthetic_scene(1.5)
        floor_estimator = GroundPlaneEstimator(config)
        tracker = PoseTracker(config)
        grid = ObstacleMap(config)
        for frame in range(3):
            t = frame / 25
            floor = floor_estimator.fit(result, t)
            pose = tracker.update(result, floor, t)
            self.assertTrue(pose.tracking_ok)
            grid.integrate(result, floor, pose, t)
        self.assertEqual(grid.cell_state(0.25, 1.5), "obstacle")
        source = tracker._anchor_points_3d
        camera = tracker._camera_matrix(result)
        target, _ = cv2.projectPoints(source, np.zeros(3),
                                      np.array([-0.1, 0., 0.]), camera, None)
        solved = tracker._solve((source, target.reshape(-1, 2),
                                tracker._anchor_points_2d), result, floor, 1.0)
        self.assertIsNotNone(solved)
        self.assertAlmostEqual(solved[0].x_m, 0.1, delta=0.02)

        yaw = math.radians(5)
        rotation_cur_anchor = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                                        [0, 1, 0],
                                        [math.sin(yaw), 0, math.cos(yaw)]])
        homography = camera @ rotation_cur_anchor @ np.linalg.inv(camera)
        turned_gray = cv2.warpPerspective(result.gray, homography,
                                          (result.gray.shape[1], result.gray.shape[0]))
        turned = DepthResult(np.zeros_like(result.disparity),
                             np.zeros_like(result.confidence), turned_gray,
                             result.Q, result.max_disparity)
        normal_cur = rotation_cur_anchor @ floor.normal_cam
        turned_floor = FloorEstimate(normal_cur, floor.height_m,
                                     gravity_rotation(normal_cur), True, True, 0, 500, 500)
        pose = tracker.update(turned, turned_floor, 0.12)
        self.assertTrue(pose.tracking_ok)
        self.assertAlmostEqual(math.degrees(pose.yaw_rad), 5, delta=2)
        grid.integrate(turned, turned_floor, pose, 0.12)
        self.assertEqual(grid.cell_state(0.25, 1.5), "obstacle")

        blank = DepthResult(result.disparity, result.confidence,
                            np.zeros_like(result.gray), result.Q, result.max_disparity)
        lost = tracker.update(blank, turned_floor, 0.16)
        self.assertFalse(lost.tracking_ok)
        integrated = grid.integrated_frames
        grid.integrate(blank, turned_floor, lost, 0.16)
        self.assertEqual(grid.integrated_frames, integrated)
        recovered = tracker.update(turned, turned_floor, 0.20)
        self.assertTrue(recovered.tracking_ok)
        self.assertEqual(grid.cell_state(0.25, 1.5), "obstacle")


class OfflineRunnerTests(unittest.TestCase):
    def test_discovery_and_explicit_unsuitable_reports(self):
        self.assertIn(Path(__file__).resolve().parents[1] /
                      "module/recorded_camera_feed.avi", discover_recordings())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports"
            for name, size in (("wrong_size.avi", (128, 48)),
                               ("blank.avi", (1280, 480))):
                path = root / name
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                                         25, size)
                self.assertTrue(writer.isOpened())
                try:
                    for _ in range(5):
                        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
                finally:
                    writer.release()
                report = run_recording(path, load_config(), output, debug_interval_s=1000)
                self.assertEqual(report["status"], "unsuitable")
                self.assertTrue(report["reason"])
                saved = json.loads((output / path.stem / "metrics.json").read_text())
                self.assertEqual(saved["status"], "unsuitable")
                if name == "blank.avi":
                    self.assertEqual(saved["integrated_frames"], 0)
                    self.assertFalse(saved["suitability"]["solid_depth_coverage"])


class LatestFrameTests(unittest.TestCase):
    def test_sequential_recording_is_paced_by_source_time(self):
        class SequentialFeed:
            def __init__(self, config):
                self._capture = type("Capture", (), {"set": lambda *_: True})()
                self._frame_index = 0
                self._fps = 25.0

            def __enter__(self):
                return self

            def read(self):
                if self._frame_index == 3:
                    raise RuntimeError("end")
                value = self._frame_index
                self._frame_index += 1
                return value, value

            def close(self):
                pass

        with LatestCameraFeed(SequentialFeed, {}) as feed:
            values = [feed.read() for _ in range(3)]
        self.assertEqual([pair[0] for pair in values], [0, 1, 2])
        np.testing.assert_allclose([pair[2] for pair in values], [0, .04, .08])

    def test_realtime_recording_publishes_each_source_frame_once(self):
        ready = Event()
        closed = Event()

        class FakeCapture:
            position = 0

            def set(self, *_):
                return True

            def get(self, *_):
                return self.position

        class RepeatingRecording:
            def __init__(self, config):
                self._capture = FakeCapture()
                self._start_time = 0.0
                self._fps = 25.0
                self.sequence = [0, 0, 0, 1, 1, 2]
                self.index = 0

            def __enter__(self):
                return self

            def read(self):
                if self.index == len(self.sequence):
                    ready.set()
                    closed.wait(1)
                    raise RuntimeError("closed")
                value = self.sequence[self.index]
                self.index += 1
                self._capture.position = value + 1
                return value, value

            def close(self):
                closed.set()

        with LatestCameraFeed(RepeatingRecording, {}) as feed:
            self.assertTrue(ready.wait(1))
            with feed._condition:
                while feed._serial < 3:
                    feed._condition.wait(1)
            self.assertEqual(feed._serial, 3)
            left, right, timestamp_s = feed.read()
            self.assertEqual((left, right), (2, 2))
            self.assertAlmostEqual(timestamp_s, 2 / 25)

    def test_capture_thread_keeps_only_latest_pair(self):
        ready = Event()
        closed = Event()

        class FakeFeed:
            def __init__(self, config):
                self._capture = type("Capture", (), {"set": lambda *_: True})()
                self.index = 0

            def __enter__(self):
                return self

            def read(self):
                if self.index == 5:
                    closed.wait(1)
                    raise RuntimeError("closed")
                value = self.index
                self.index += 1
                if self.index == 5:
                    ready.set()
                return value, value

            def close(self):
                closed.set()

        with LatestCameraFeed(FakeFeed, {}) as feed:
            self.assertTrue(ready.wait(1))
            with feed._condition:
                while feed._serial < 5:
                    feed._condition.wait(1)
            left, right, timestamp_s = feed.read()
            self.assertEqual((left, right), (4, 4))
            self.assertGreater(timestamp_s, 0)


class LiveEntryTests(unittest.TestCase):
    def test_recording_end_exits_live_view_cleanly(self):
        config = load_config()
        config["benchmark"]["output"] = None
        with patch("map_live.load_config", return_value=config), \
                patch("map_live.DepthMapper") as mapper, \
                patch("map_live.LatestCameraFeed") as camera, \
                patch("map_live.MapWindow") as window_class:
            window_class.return_value.__enter__.return_value.is_open = True
            camera.return_value.__enter__.return_value.read.side_effect = RuntimeError(
                "End of stereo recording")
            with redirect_stdout(io.StringIO()):
                map_live.main()
            mapper.return_value.compute.assert_not_called()
            window_class.return_value.__exit__.assert_called_once()

    def test_one_live_frame_runs_depth_floor_tracker_and_map(self):
        config = load_config()
        config["benchmark"]["output"] = None
        result, _, _ = synthetic_scene(1.5)
        with patch("map_live.load_config", return_value=config), \
                patch("map_live.DepthMapper") as mapper, \
                patch("map_live.LatestCameraFeed") as camera, \
                patch("map_live.MapWindow") as window_class:
            mapper.return_value.compute.return_value = result
            camera.return_value.__enter__.return_value.read.return_value = ("left", "right", 0.0)
            window = window_class.return_value.__enter__.return_value
            window.is_open = True
            window.update.side_effect = lambda *_: setattr(window, "is_open", False)
            with redirect_stdout(io.StringIO()):
                map_live.main()
            mapper.return_value.compute.assert_called_once_with("left", "right")
            self.assertEqual(window.update.call_count, 1)
            grid = window.update.call_args.args[1]
            self.assertEqual(grid.integrated_frames, 1)


if __name__ == "__main__":
    unittest.main()
