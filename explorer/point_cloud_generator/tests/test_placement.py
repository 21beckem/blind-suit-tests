import math
import unittest

from explorer.point_cloud_generator.config import resolve_config
from explorer.point_cloud_generator.geometry import footprint_extents, footprints_overlap, place_objects
import numpy as np


class PlacementTests(unittest.TestCase):
    def test_footprints_do_not_overlap(self):
        config = resolve_config({"seed": 4, "objects": {"count": 7}})
        objects = place_objects(config, np.random.default_rng(config.seed))
        for index, left in enumerate(objects):
            for right in objects[index + 1 :]:
                self.assertFalse(
                    footprints_overlap(left, right, config.objects.object_margin_m)
                )

    def test_cabinets_touch_a_wall_and_chairs_are_near_tables(self):
        config = resolve_config(
            {
                "seed": 12,
                "objects": {
                    "count": 5,
                    "allowed": ["cabinet", "table", "chair"],
                    "weights": {"cabinet": 2, "table": 2, "chair": 4},
                },
            }
        )
        objects = place_objects(config, np.random.default_rng(config.seed))
        cabinets = [item for item in objects if item.kind == "cabinet"]
        tables = [item for item in objects if item.kind == "table"]
        chairs = [item for item in objects if item.kind == "chair"]
        self.assertTrue(cabinets)
        self.assertTrue(tables)
        self.assertTrue(chairs)
        for item in cabinets:
            ex, ez = footprint_extents(item.size_m, math.radians(item.yaw_deg))
            wall_gap = min(
                config.room.width_m / 2 - (abs(item.center_m[0]) + ex),
                item.center_m[2] - ez + config.room.depth_m,
            )
            self.assertAlmostEqual(wall_gap, config.objects.wall_margin_m, places=6)
        for chair in chairs:
            nearest = min(
                math.hypot(chair.center_m[0] - table.center_m[0], chair.center_m[2] - table.center_m[2])
                for table in tables
            )
            self.assertLess(nearest, 1.5)


if __name__ == "__main__":
    unittest.main()

