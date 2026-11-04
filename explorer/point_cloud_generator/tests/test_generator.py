import unittest

import numpy as np

from explorer.point_cloud_generator import (
    generate_point_cloud,
    generate_scan,
    generate_scene,
    scan_scene,
)
from explorer.point_cloud_generator.generator import _distance_to_other_surfaces


TEST_CONFIG = {
    "seed": 9,
    "room": {"width_m": 3.5, "depth_m": 4.0, "height_m": 2.4},
    "objects": {"count": 3},
    "scan": {
        "points_per_m2": 120,
        "max_points": 20_000,
        "noise_sigma_m": 0.0,
        "dropout": 0.0,
    },
}


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scan = generate_scan(TEST_CONFIG)

    def test_detector_shape_and_types(self):
        self.assertEqual(self.scan.points.ndim, 2)
        self.assertEqual(self.scan.points.shape[1], 3)
        self.assertEqual(self.scan.points.dtype, np.float64)
        self.assertTrue(self.scan.points.flags.c_contiguous)
        self.assertTrue(np.isfinite(self.scan.points).all())
        self.assertEqual(len(self.scan.points), len(self.scan.object_ids))

    def test_scene_has_room_and_objects(self):
        ids = set(self.scan.object_ids.tolist())
        self.assertIn(0, ids)
        self.assertEqual(len(self.scan.objects), 3)
        self.assertTrue(any(item.object_id in ids for item in self.scan.objects))

    def test_room_has_front_and_back_walls(self):
        scene = generate_scene(TEST_CONFIG)
        part_kinds = {part.kind for part in scene.parts}
        self.assertIn("front_wall", part_kinds)
        self.assertIn("back_wall", part_kinds)

    def test_primary_api_returns_only_points(self):
        points = generate_point_cloud(TEST_CONFIG)
        self.assertIsInstance(points, np.ndarray)
        np.testing.assert_array_equal(points, self.scan.points)

    def test_seed_is_deterministic(self):
        again = generate_scan(TEST_CONFIG)
        np.testing.assert_array_equal(self.scan.points, again.points)
        np.testing.assert_array_equal(self.scan.object_ids, again.object_ids)

    def test_scene_can_be_rescanned_from_another_pose(self):
        scene = generate_scene(TEST_CONFIG)
        first = scan_scene(
            scene, sensor_position_m=(0.0, 0.0, 0.0), sensor_yaw_deg=0.0, omniscient_scan=True
        )
        moved = scan_scene(
            scene, sensor_position_m=(0.5, 0.0, -1.0), sensor_yaw_deg=90.0, omniscient_scan=True
        )
        self.assertEqual(first.objects, moved.objects)
        self.assertEqual(len(first.points), len(moved.points))
        self.assertFalse(np.array_equal(first.points, moved.points))
        self.assertEqual(moved.sensor.world_position_m, (0.5, 0.0, -1.0))

    def test_omniscient_scan_contains_more_than_single_view(self):
        scene = generate_scene(TEST_CONFIG)
        complete = scan_scene(scene, omniscient_scan=True)
        single = scan_scene(scene, omniscient_scan=False)
        self.assertGreater(len(complete.points), len(single.points))

    def test_single_view_uses_a_forward_frustum_and_keeps_room_planes(self):
        scene = generate_scene(TEST_CONFIG)
        scan = scan_scene(
            scene,
            omniscient_scan=False,
            scan_overrides={
                "noise_sigma_m": 0.0,
                "dropout": 0.0,
                "outlier_fraction": 0.0,
                "flat_plane_degradation": False,
            },
        )
        self.assertTrue(np.all(scan.points[:, 2] < 0.0))
        horizontal = np.abs(np.degrees(np.arctan2(scan.points[:, 0], -scan.points[:, 2])))
        vertical = np.abs(
            np.degrees(np.arctan2(scan.points[:, 1], np.hypot(scan.points[:, 0], scan.points[:, 2])))
        )
        self.assertLessEqual(horizontal.max(), scan.config["scan"]["single_view_horizontal_fov_deg"] / 2)
        self.assertLessEqual(vertical.max(), scan.config["scan"]["single_view_vertical_fov_deg"] / 2)
        self.assertGreater(np.isclose(scan.points[:, 1], -1.4).sum(), 10)  # floor
        self.assertGreater(np.isclose(scan.points[:, 2], -3.5).sum(), 10)  # back wall

    def test_flat_plane_effect_is_tunable(self):
        scene = generate_scene(TEST_CONFIG)
        realistic = scan_scene(scene)
        ideal = scan_scene(
            scene,
            scan_overrides={"flat_plane_degradation": False},
        )
        room_realistic = int((realistic.object_ids == 0).sum())
        room_ideal = int((ideal.object_ids == 0).sum())
        self.assertLess(room_realistic, room_ideal)

    def test_flat_plane_effect_applies_to_cabinets_and_tables(self):
        scene = generate_scene(
            {
                "seed": 31,
                "room": {"width_m": 4.0, "depth_m": 5.0},
                "objects": {
                    "count": 2,
                    "allowed": ["cabinet", "table"],
                    "weights": {"cabinet": 1, "table": 1},
                },
                "scan": {
                    "points_per_m2": 500,
                    "max_points": 50_000,
                    "noise_sigma_m": 0.0,
                    "dropout": 0.0,
                    "outlier_fraction": 0.0,
                },
            }
        )
        realistic = scan_scene(scene)
        ideal = scan_scene(scene, scan_overrides={"flat_plane_degradation": False})
        for object_id in (1, 2):
            self.assertLess(
                int((realistic.object_ids == object_id).sum()),
                int((ideal.object_ids == object_id).sum()),
            )

    def test_empty_floor_under_table_is_not_marked_near_the_table_box(self):
        scene = generate_scene(
            {
                "seed": 31,
                "room": {"width_m": 4.0, "depth_m": 5.0},
                "objects": {"count": 1, "allowed": ["table"]},
                "scan": {
                    "points_per_m2": 500,
                    "max_points": 50_000,
                    "noise_sigma_m": 0.0,
                    "dropout": 0.0,
                    "outlier_fraction": 0.0,
                    "flat_plane_degradation": False,
                },
            }
        )
        scan = scan_scene(scene)
        item = scan.objects[0]
        yaw = np.radians(item.yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        local_rotation = np.asarray(
            [
                [np.cos(np.radians(scan.sensor.world_yaw_deg)), 0, np.sin(np.radians(scan.sensor.world_yaw_deg))],
                [0, 1, 0],
                [-np.sin(np.radians(scan.sensor.world_yaw_deg)), 0, np.cos(np.radians(scan.sensor.world_yaw_deg))],
            ]
        )
        floor_y = -scene.config["room"]["sensor_height_m"]
        center_world = np.asarray([item.center_m[0], floor_y, item.center_m[2]])
        near_leg_world = center_world + np.asarray(
            [c * (item.size_m[0] / 2 - 0.04) + s * (item.size_m[1] / 2 - 0.04), 0,
             -s * (item.size_m[0] / 2 - 0.04) + c * (item.size_m[1] / 2 - 0.04)]
        )
        floor_samples = np.vstack(
            [
                (center_world - np.asarray(scan.sensor.world_position_m)) @ local_rotation,
                (near_leg_world - np.asarray(scan.sensor.world_position_m)) @ local_rotation,
            ]
        )
        object_points = scan.points[scan.object_ids > 0]
        distances = _distance_to_other_surfaces(
            floor_samples,
            np.zeros(2, dtype=np.int32),
            object_points,
            scan.object_ids[scan.object_ids > 0],
        )
        self.assertGreater(distances[0], distances[1] * 3)


if __name__ == "__main__":
    unittest.main()

